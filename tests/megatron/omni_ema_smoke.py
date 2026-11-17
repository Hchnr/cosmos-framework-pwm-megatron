"""Real OmniMoT construction, action forward/backward, EMA scope and checkpoint."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from async_checkpoint_smoke import writer
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec
from cosmos_framework.model.generator.mot.unified_mot import (
    Nemotron3DenseVLMoTConfig,
    Nemotron3DenseVLTextForCausalLM,
    Qwen3VLMoeMoTConfig,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLMoTConfig,
    Qwen3VLTextForCausalLM,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.training_backends.checkpoint_metadata import extract_chunk_metadata
from cosmos_framework.training_backends.megatron import get_runtime
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor
from cosmos_framework.utils.generator.optimizer import OptimizersContainer
from cosmos_framework.utils.lazy_config import LazyCall


def tiny_config(**kwargs):
    return Qwen3VLMoTConfig(
        {
            "text_config": dict(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                rms_norm_eps=1e-6,
            )
        },
        **kwargs,
    )


def tiny_nemotron_config(**kwargs):
    kwargs.setdefault("qk_norm_for_text", False)
    return Nemotron3DenseVLMoTConfig(
        dict(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            mrope_section=[3, 3, 2],
        ),
        **kwargs,
    )


def tiny_moe_config(**kwargs):
    return Qwen3VLMoeMoTConfig(
        {
            "text_config": dict(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                rms_norm_eps=1e-6,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=64,
            )
        },
        **kwargs,
    )


class TinyOmni(OmniMoTModel):
    def set_up_tokenizers(self):
        self.vlm_config = self.config.vlm_config
        self.vlm_checkpoint_path = ""
        self.tokenizer_vision_gen = None
        self.tokenizer_sound_gen = None


def batch():
    indexes = torch.arange(2, 5, device="cuda")
    action = ModalityData(
        sequence_indexes=indexes,
        timesteps=torch.ones(3, device="cuda"),
        mse_loss_indexes=indexes,
        token_shapes=[(3,)],
        tokens=[torch.randn(3, 128, device="cuda")],
        condition_mask=[torch.zeros(3, device="cuda")],
        noisy_frame_indexes=[torch.arange(3, device="cuda")],
        domain_id=[torch.tensor([0], device="cuda")],
    )
    return PackedSequence(
        sample_lens=[5],
        split_lens=[2, 3],
        attn_modes=["causal", "full"],
        sequence_length=5,
        text_ids=torch.tensor([1, 2], device="cuda"),
        text_indexes=torch.arange(2, device="cuda"),
        position_ids=torch.arange(5, device="cuda").expand(3, -1),
        action=action,
    )


def tiny_visual_config(**kwargs):
    config = tiny_config(**kwargs)
    config.config_dict.update(
        vision_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            depth=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[0, 1],
            num_position_embeddings=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
        ),
        image_token_id=4,
        video_token_id=7,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    return config


def visual_batch(config=None):
    data = batch()
    data.sample_lens, data.split_lens, data.sequence_length = [6], [3, 3], 6
    token_ids = (
        [5, 4, 6]
        if config is None
        else [config.vision_start_token_id, config.image_token_id, config.vision_end_token_id]
    )
    patch_features = 24
    if config is not None:
        vision = config.vision_config
        patch_features = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    data.text_ids = torch.tensor(token_ids, device="cuda")
    data.text_indexes = torch.arange(3, device="cuda")
    data.position_ids = torch.arange(6, device="cuda").expand(3, -1).clone()
    data.action.sequence_indexes = data.action.sequence_indexes + 1
    data.action.mse_loss_indexes = data.action.mse_loss_indexes + 1
    return data, [
        dict(
            pixel_values=torch.randn(4, patch_features, device="cuda"),
            image_grid_thw=torch.tensor([[1, 2, 2]], device="cuda"),
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--family", choices=("qwen", "nemotron", "moe"), default="qwen")
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    config = copy.deepcopy(OmniMoTModelConfig())
    config.parallelism.backend = "megatron"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_shard_degree = args.cp
    config.parallelism.megatron_sharding_strategy = args.sharding_strategy
    config.compile.enabled = args.compile
    config.compile.compile_dynamic = False
    config.ema.enabled = True
    config.lora_enabled = args.lora
    config.lora_rank, config.lora_alpha = 4, 8
    config.vision_gen = True
    config.action_gen = True
    config.action128_spec = tiny_action128_spec()
    config.tower_structure = "two_tower"
    config.load_vision_tokenizer = False
    config.load_vlm_visual = args.visual
    config.tokenizer = SimpleNamespace(temporal_compression_factor=4)
    model_class, config_factory = {
        "qwen": (Qwen3VLTextForCausalLM, tiny_config),
        "nemotron": (Nemotron3DenseVLTextForCausalLM, tiny_nemotron_config),
        "moe": (Qwen3VLMoeTextForCausalLM, tiny_moe_config),
    }[args.family]
    if args.visual:
        assert args.family == "qwen"
        config_factory = tiny_visual_config
    config.vlm_config.model_instance = LazyCall(model_class)(config=LazyCall(config_factory)())
    model = TinyOmni(config)
    meta_initialization = os.environ.get("COSMOS_MEGATRON_OMNI_META_INITIALIZATION", "0") == "1"
    assert all(
        p.device.type in (("cpu", "meta") if meta_initialization else ("cpu",)) for p in model.net_ema.parameters()
    )
    model.to("cuda")
    assert all(
        p.device.type in (("cpu", "meta") if meta_initialization else ("cpu",)) for p in model.net_ema.parameters()
    )
    expected = None if meta_initialization else {n: v.cuda().clone() for n, v in model.net_ema.state_dict().items()}
    optimizer = OptimizersContainer(
        model,
        "AdamW",
        fused=True,
        lr=1e-3,
        keys_to_select=(
            ["lora_"]
            if args.lora
            else ["_moe_gen", "action_modality_embed", "action_timestep", "action2llm", "llm2action"]
        ),
    )
    assert all(p.device.type == "cuda" for p in model.net_ema.parameters())
    runtime = get_runtime(model)
    if expected is None:
        expected = {n: materialize_tensor(v).clone() for n, v in model.net_ema.state_dict().items()}
    replica_initialization_verified = False
    if meta_initialization and args.lora:
        digest = hashlib.sha256()
        for name, parameter in model.net.named_parameters():
            if "lora_A" not in name:
                continue
            full = materialize_tensor(parameter).float()
            assert torch.linalg.matrix_rank(full).item() == full.shape[0], name
            digest.update(name.encode())
            digest.update(full.cpu().contiguous().numpy().tobytes())
        hashes = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, digest.hexdigest())
        assert len(set(hashes)) == 1, hashes
        replica_initialization_verified = True
    parameter_names = {
        n.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
        for n, _ in model.net_ema.named_parameters()
    }
    initial_frozen = (
        {
            name: parameter.to_local().detach().cpu().clone()
            for name, parameter in model.net.named_parameters()
            if not parameter.requires_grad
        }
        if args.lora
        else {}
    )
    adapter_gradients = []
    output_projection_gradients = []
    metadata_parity_verified = False
    for step in range(2):
        optimizer.zero_grad()
        for micro in range(3):
            torch.manual_seed(step * 100 + micro + dist.get_rank() // args.cp)
            with ddp_sync_grad(model, enabled=micro == 2):
                data, visual_inputs = visual_batch() if args.visual else (batch(), None)
                output = model.net(data, vlm_inputs=visual_inputs)
                if args.visual and step == 0 and micro == 0:
                    data.prepare_sequence_pack_metadata()
                    cached_output = model.net(copy.deepcopy(data), vlm_inputs=visual_inputs)
                    for a, b in zip(output["preds_action"], cached_output["preds_action"], strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    metadata_parity_verified = True
                loss = sum(t.float().square().mean() for t in output["preds_action"]) / 3
                assert torch.isfinite(loss)
                loss.backward()
        if args.lora:
            maxima = torch.zeros(3, device="cuda")
            for name, parameter in model.net.named_parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.to_local()
                if gradient.numel() and "lora_" in name:
                    index = 0 if "lora_A" in name else 1
                    maximum = gradient.float().abs().max()
                    maxima[index] = torch.maximum(maxima[index], maximum)
                    if "o_proj_moe_gen.lora_B" in name:
                        maxima[2] = torch.maximum(maxima[2], maximum)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            assert maxima[1].item() > 0, "LoRA B must receive learning gradients"
            if step > 0:
                assert maxima[0].item() > 0, "LoRA A must learn after the first B update"
            assert maxima[2].item() > 0, "Output-projection LoRA B must receive gradients"
            adapter_gradients.append(maxima[:2].tolist())
            output_projection_gradients.append(maxima[2].item())
        optimizer.step()
        beta = model.ema_beta(step)
        model.on_before_zero_grad(optimizer, None, step)
        for name, value in model.net.state_dict().items():
            if name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "") in parameter_names:
                expected[name].mul_(beta).add_(materialize_tensor(value).float(), alpha=1 - beta)
        for name, value in model.net_ema.state_dict().items():
            if hasattr(value, "to_local"):
                torch.testing.assert_close(materialize_tensor(value), expected[name], rtol=0, atol=0)
        original = {n: materialize_tensor(v).clone() for n, v in model.net.state_dict().items()}
        torch.manual_seed(999 + dist.get_rank() // args.cp)
        with torch.no_grad(), model.ema_scope(is_cpu=True):
            data, visual_inputs = visual_batch() if args.visual else (batch(), None)
            evaluation = model.net(data, vlm_inputs=visual_inputs)
            assert all(torch.isfinite(t).all() for t in evaluation["preds_action"])
        for name, value in model.net.state_dict().items():
            torch.testing.assert_close(materialize_tensor(value), original[name], rtol=0, atol=0)
    if args.lora:
        for name, parameter in model.net.named_parameters():
            if name in initial_frozen:
                torch.testing.assert_close(parameter.to_local().cpu(), initial_frozen[name], rtol=0, atol=0)
    checkpoint = Path("/tmp") / f"cosmos_megatron_omni_ema_{os.environ['MASTER_PORT']}"
    saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    metadata = extract_chunk_metadata(saved)
    staged = _create_cpu_state_dict(saved, pin_memory=True, share_memory=True)
    _copy_state_dict(saved, staged)
    torch.cuda.synchronize()
    context = torch.multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=writer,
        args=(queue, dist.get_rank(), dist.get_world_size(), int(os.environ["MASTER_PORT"]) + 2, checkpoint),
    )
    process.start()
    queue.put((staged, metadata))
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    assert process.exitcode == 0, process.exitcode
    with torch.no_grad():
        for parameter in list(model.net.parameters()) + list(model.net_ema.parameters()):
            parameter.to_local().zero_()
    restored = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.load(restored, checkpoint_id=checkpoint)
    model.load_state_dict(restored["model"], strict=False)
    optimizer.load_state_dict(restored["optimizer"])
    for name, value in model.net.state_dict().items():
        torch.testing.assert_close(materialize_tensor(value), original[name], rtol=0, atol=0)
    for name, value in model.net_ema.state_dict().items():
        torch.testing.assert_close(materialize_tensor(value), expected[name], rtol=0, atol=0)
    assert runtime is get_runtime(model)
    result = dict(
        passed=True,
        model=f"OmniMoTModel with real {args.family} MoT and Action128 heads",
        family=args.family,
        visual=args.visual,
        cached_and_uncached_metadata_bitwise_equal=metadata_parity_verified,
        meta_initialization=meta_initialization,
        full_rank_adapters_and_replica_initialization_equal=replica_initialization_verified,
        lora=args.lora,
        lora_gradient_maxima_A_B=adapter_gradients,
        output_projection_lora_B_gradient_maxima=output_projection_gradients,
        frozen_base_bitwise_unchanged=True if args.lora else None,
        steps=2,
        compiled=args.compile,
        cp=args.cp,
        sharding_strategy=args.sharding_strategy,
        cpu_ema_cache=True,
        gradient_accumulation=3,
        shards=2,
        replicas=2,
        ema_bitwise_equal=True,
        ema_scope_finite=True,
        training_weights_restored_bitwise=True,
        checkpoint_roundtrip=True,
        async_cpu_staging_spawn_queue=True,
    )
    if dist.get_rank() == 0:
        suffix = "_cp2" if args.cp > 1 else "_compiled" if args.compile else ""
        if meta_initialization:
            suffix = "_meta" + suffix
        if args.lora:
            suffix = "_lora" + suffix
        if args.visual:
            suffix = "_visual" + suffix
        if args.family != "qwen":
            suffix = "_" + args.family + suffix
        Path(f"artifacts/megatron/omni_ema_integration{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
