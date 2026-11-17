"""Fresh Omni HF startup hook, uneven UND→GEN copies, EMA and full math parity."""

import argparse
import copy
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omni_ema_smoke import TinyOmni, batch, tiny_config, tiny_nemotron_config
from safetensors.torch import load_file, save_file

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec
from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLTextForCausalLM, Qwen3VLTextForCausalLM
from cosmos_framework.training_backends.megatron import get_runtime
from cosmos_framework.utils.distributed import ddp_sync_grad
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor
from cosmos_framework.utils.generator.optimizer import OptimizersContainer
from cosmos_framework.utils.lazy_config import LazyCall


def checkpoint_key(name, family):
    if family == "qwen":
        return "model.language_model." + name[6:] if name.startswith("model.") else name
    if name == "model.embed_tokens.weight":
        return "model.language_model.embeddings.weight"
    if name == "model.norm.weight":
        return "model.language_model.norm_f.weight"
    if name == "lm_head.weight":
        return name
    match = re.fullmatch(r"model.layers.(\d+).(input_layernorm|post_attention_layernorm).weight", name)
    if match:
        block = 2 * int(match[1]) + (match[2] == "post_attention_layernorm")
        return f"model.language_model.layers.{block}.norm.weight"
    match = re.fullmatch(r"model.layers.(\d+).(self_attn|mlp).(.*)", name)
    assert match, name
    block = 2 * int(match[1]) + (match[2] == "mlp")
    return f"model.language_model.layers.{block}.mixer.{match[3]}"


def construct(backend, family, path, optimizer_type, sharding_strategy, partial_freeze, lora):
    config = copy.deepcopy(OmniMoTModelConfig())
    config.parallelism.backend = backend
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.megatron_sharding_strategy = sharding_strategy
    config.compile.enabled, config.compile.compile_dynamic = True, False
    config.ema.enabled = True
    config.lora_enabled = lora
    config.lora_rank, config.lora_alpha = 4, 8
    config.vision_gen = config.action_gen = True
    config.action128_spec = tiny_action128_spec()
    config.tower_structure = "two_tower"
    config.load_vision_tokenizer = config.load_vlm_visual = False
    config.tokenizer = SimpleNamespace(temporal_compression_factor=4)
    model_class, factory = (
        (Nemotron3DenseVLTextForCausalLM, tiny_nemotron_config)
        if family == "nemotron"
        else (Qwen3VLTextForCausalLM, tiny_config)
    )
    config.vlm_config.model_instance = LazyCall(model_class)(config=LazyCall(factory)())
    pretrain = config.vlm_config.pretrained_weights
    pretrain.enabled, pretrain.backbone_path, pretrain.credentials_path = True, str(path), ""
    pretrain.checkpoint_format = "nemotron_3_dense_vl" if family == "nemotron" else "qwen3"
    config.diffusion_expert_config.load_weights_from_pretrained = True
    model = TinyOmni(config).to("cuda")
    optimizer = OptimizersContainer(
        model,
        optimizer_type,
        fused=True,
        lr=1e-3,
        keys_to_select=(
            ["lora_"]
            if lora
            else (
                [
                    "q_proj_moe_gen",
                    "k_proj_moe_gen",
                    "v_proj_moe_gen",
                    "mlp_moe_gen",
                    "norm_moe_gen",
                    "action_modality_embed",
                    "action_timestep",
                    "action2llm",
                    "llm2action",
                ]
                if partial_freeze
                else ["_moe_gen", "action_modality_embed", "action_timestep", "action2llm", "llm2action"]
            )
        ),
    )
    return model, optimizer


def canonical(name):
    return name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")


def compare(source, target):
    expected = {canonical(n): value for n, value in source.state_dict().items()}
    maximum = 0.0
    for name, value in target.state_dict().items():
        a, b = materialize_tensor(expected[canonical(name)]), materialize_tensor(value)
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        if a.is_floating_point():
            maximum = max(maximum, (a - b).abs().max().item())
    return maximum


def verify_hf_loaded(language, before, weights, family, copy_generation):
    checked = 0
    for raw_name, value in language.state_dict().items():
        name = canonical(raw_name)
        original = name.replace("_moe_gen_action", "").replace("_moe_gen", "")
        key = checkpoint_key(original, family)
        if "_moe_gen" not in name or (copy_generation and key in weights):
            expected = weights[key]
        else:
            expected = before[name]
        actual = materialize_tensor(value)
        torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype), rtol=0, atol=0, msg=name)
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--family", choices=("qwen", "nemotron"), default="qwen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-generation-freeze", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--optimizer", choices=("AdamW", "FusedAdam"), default="AdamW")
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    assert not args.lora or args.warm_start, "LoRA startup test uses a pretrained generation checkpoint"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    path = Path("/tmp") / f"cosmos_megatron_fresh_omni_{os.environ['MASTER_PORT']}"
    if dist.get_rank() == 0:
        model_class, factory = (
            (Nemotron3DenseVLTextForCausalLM, tiny_nemotron_config)
            if args.family == "nemotron"
            else (Qwen3VLTextForCausalLM, tiny_config)
        )
        template = model_class(factory())
        weights = {
            checkpoint_key(name, args.family): torch.rand_like(value.float()) * 0.05 + 0.1
            for name, value in template.state_dict().items()
            if "_moe_gen" not in name
        }
        path.mkdir()
        save_file(weights, path / "model.safetensors")
    dist.barrier()
    source, source_opt = construct(
        "pytorch", args.family, path, args.optimizer, args.sharding_strategy, args.partial_generation_freeze, args.lora
    )
    target, target_opt = construct(
        "megatron", args.family, path, args.optimizer, args.sharding_strategy, args.partial_generation_freeze, args.lora
    )
    checkpoint = path / "aligned_initial_state"
    dcp.save({"model": source.state_dict()}, checkpoint_id=checkpoint)
    restored = {"model": target.state_dict()}
    dcp.load(restored, checkpoint_id=checkpoint)
    target.load_state_dict(restored["model"], strict=False)
    compare(source, target)
    before = [
        {canonical(n): materialize_tensor(value).cpu().clone() for n, value in language.state_dict().items()}
        for language in (source.net.language_model, source.net_ema.language_model)
    ]
    for model in (source, target):
        model.load_pretrained_model_if_needed(has_resumable_checkpoint=False, has_load_path=args.warm_start)
    weights = load_file(path / "model.safetensors")
    verified = [
        verify_hf_loaded(language, initial, weights, args.family, not args.warm_start)
        for language, initial in zip((source.net.language_model, source.net_ema.language_model), before, strict=True)
    ]
    compare(source, target)
    runtime = get_runtime(target)
    language_state = {canonical(n): value for n, value in target.net.language_model.state_dict().items()}
    runtime.prepare_state_dict(language_state)
    different_owners = 0
    for name, value in language_state.items():
        original_name = name.replace("_moe_gen_action", "").replace("_moe_gen", "")
        if "_moe_gen" not in name or original_name not in language_state:
            continue
        source_value = language_state[original_name]
        source_chunk = source_value._local_tensor.__create_chunk_list__()[0]
        destination_chunk = value._local_tensor.__create_chunk_list__()[0]
        different_owners += (source_chunk.offsets, source_chunk.sizes) != (
            destination_chunk.offsets,
            destination_chunk.sizes,
        )
    owner_cases = [None] * dist.get_world_size()
    dist.all_gather_object(owner_cases, different_owners)
    if args.partial_generation_freeze:
        assert sum(owner_cases) > 0, "Fixture must exercise different UND/GEN ownership"
    for optimizer in (source_opt, target_opt):
        optimizer.zero_grad()
    maximum_output_error = 0.0
    for micro in range(3):
        torch.manual_seed(100 + micro + dist.get_rank())
        data, outputs = batch(), []
        for model in (source, target):
            with ddp_sync_grad(model, enabled=micro == 2):
                output = model.net(copy.deepcopy(data))["preds_action"][0]
                (output.float().square().mean() / 3).backward()
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-4, atol=2e-5)
        maximum_output_error = max(maximum_output_error, (outputs[0] - outputs[1]).abs().max().item())
    runtime = get_runtime(target)
    reference = {canonical(n): p for n, p in source.net.named_parameters()}
    gradients = {
        n: p.grad if p.grad is not None else torch.zeros_like(p)
        for n, p in target.net.named_parameters()
        if p.requires_grad
    }
    runtime.prepare_state_dict(gradients)
    maximum_gradient_error = 0.0
    for name, gradient in gradients.items():
        a, b = materialize_tensor(reference[canonical(name)].grad), materialize_tensor(gradient)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=name)
        maximum_gradient_error = max(maximum_gradient_error, (a - b).abs().max().item())
    source_opt.step()
    target_opt.step()
    for model, optimizer in ((source, source_opt), (target, target_opt)):
        model.on_before_zero_grad(optimizer, None, 0)
    compare(source, target)
    result = dict(
        passed=True,
        family=args.family,
        actual_startup_hook=True,
        optimizer=args.optimizer,
        partial_generation_freeze=args.partial_generation_freeze,
        lora=args.lora,
        different_und_gen_owner_cases_per_rank=owner_cases,
        sharding_strategy=args.sharding_strategy,
        fresh_und_to_gen_copy=not args.warm_start,
        warm_start=args.warm_start,
        independently_verified_language_tensors_online_and_ema=verified,
        ema=True,
        compile=True,
        gradient_accumulation=3,
        initialized_weights_bitwise_equal=True,
        optimizer_update=True,
        updated_model_and_ema_bitwise_equal=True,
        maximum_output_error=maximum_output_error,
        maximum_gradient_error=maximum_gradient_error,
    )
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
