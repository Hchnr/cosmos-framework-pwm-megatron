"""Real OmniMoT construction, action forward/backward, EMA scope and checkpoint."""

import argparse
import copy
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
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    config = copy.deepcopy(OmniMoTModelConfig())
    config.parallelism.backend = "megatron"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.megatron_sharding_strategy = args.sharding_strategy
    config.compile.enabled = args.compile
    config.compile.compile_dynamic = False
    config.ema.enabled = True
    config.vision_gen = True
    config.action_gen = True
    config.action128_spec = tiny_action128_spec()
    config.tower_structure = "two_tower"
    config.load_vision_tokenizer = False
    config.load_vlm_visual = False
    config.tokenizer = SimpleNamespace(temporal_compression_factor=4)
    config.vlm_config.model_instance = LazyCall(Qwen3VLTextForCausalLM)(config=LazyCall(tiny_config)())
    model = TinyOmni(config)
    assert all(p.device.type == "cpu" for p in model.net_ema.parameters())
    model.to("cuda")
    assert all(p.device.type == "cpu" for p in model.net_ema.parameters())
    expected = {n: v.cuda().clone() for n, v in model.net_ema.state_dict().items()}
    optimizer = OptimizersContainer(
        model,
        "AdamW",
        fused=True,
        lr=1e-3,
        keys_to_select=["_moe_gen", "action_modality_embed", "action_timestep", "action2llm", "llm2action"],
    )
    assert all(p.device.type == "cuda" for p in model.net_ema.parameters())
    runtime = get_runtime(model)
    parameter_names = {
        n.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
        for n, _ in model.net_ema.named_parameters()
    }
    for step in range(2):
        optimizer.zero_grad()
        for micro in range(3):
            with ddp_sync_grad(model, enabled=micro == 2):
                output = model.net(batch())
                loss = sum(t.float().square().mean() for t in output["preds_action"]) / 3
                assert torch.isfinite(loss)
                loss.backward()
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
        with torch.no_grad(), model.ema_scope(is_cpu=True):
            evaluation = model.net(batch())
            assert all(torch.isfinite(t).all() for t in evaluation["preds_action"])
        for name, value in model.net.state_dict().items():
            torch.testing.assert_close(materialize_tensor(value), original[name], rtol=0, atol=0)
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
        model="OmniMoTModel with real Qwen MoT and Action128 heads",
        steps=2,
        compiled=args.compile,
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
        Path(
            "artifacts/megatron/omni_ema_integration_compiled.json"
            if args.compile
            else "artifacts/megatron/omni_ema_integration.json"
        ).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
