#!/usr/bin/env python3
"""Replay explicit Nano denoiser inputs, with no data/noise/timestep sampling.

FP32 reference runs replace the unsupported native FP32 FlashAttention with
independent PyTorch math SDPA. The three-tower joint path can use either math
SDPA or production FlexAttention, separating routing from backend precision.
Use torchrun with two ranks for the complete FP32 three-tower model.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tools.validate_action_tower import cpu


def math_attention(q, k, v, *, is_causal=False, scale=None, **kwargs):
    if kwargs.get("cumulative_seqlen_Q") is not None:
        raise ValueError("FP32 replay reference expects one unpadded sample per call")
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,
        ).transpose(1, 2)


def joint_math(q, k, v, *, block_mask, scale, **kwargs):
    qi = torch.arange(q.shape[2], device=q.device)[:, None]
    ki = torch.arange(k.shape[2], device=k.device)[None, :]
    mask = block_mask.mask_mod(0, 0, qi, ki)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)


def to_device(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [to_device(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item) for item in value)
    if isinstance(value, dict):
        return {key: to_device(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value, **{f.name: to_device(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
        )
    return value


def install_routing_audit(model):
    """Assert actual rows entering every GEN module, including final norms."""
    expected, observed = {}, {}

    def layer_pre(module, inputs):
        pack = inputs[0]
        expected["action"] = pack["_action_indices"].numel()
        expected["video"] = pack["_video_indices"].numel()

    def module_pre(name):
        def hook(module, inputs):
            branch = "action" if "_moe_gen_action" in name else "video"
            rows = inputs[0].shape[0]
            assert rows == expected[branch], f"{name}: received {rows} rows, expected {expected[branch]}"
            stats = observed.setdefault(name, {"branch": branch, "calls": 0, "row_counts": []})
            stats["calls"] += 1
            if rows not in stats["row_counts"]:
                stats["row_counts"].append(rows)

        return hook

    for layer in model.net.language_model.model.layers:
        layer.register_forward_pre_hook(layer_pre)
    for name, module in model.net.language_model.named_modules():
        if "_moe_gen" in name:
            module.register_forward_pre_hook(module_pre(name))
    return observed


def sample_frozen(model, fixture, args, rank, routing_audit):
    # Match the three-tower execution boundary even for a one-GPU reference.
    model._can_reuse_inference_text_kv = lambda *a, **kw: False
    outputs, results, inputs = [], [], []
    for sample, capture in enumerate(fixture["captures"]):
        assert len(capture["prepared"]) == 1, "Recapture fixture with --freeze-inputs to save preparation"
        prepared = to_device(copy.deepcopy(capture["prepared"][0]))
        # This replaces only the preparation boundary in this diagnostic tool.
        # The ordinary sampler, scheduler, CFG and output processing remain in use.
        model._prepare_inference_data = lambda *a, **kw: prepared

        def pre(module, positional, kwargs):
            packed = kwargs["packed_seq"]
            assert packed.action.robot_state_token_indexes.numel() == 0
            inputs.append(cpu(packed))

        def post(module, positional, output):
            outputs.append(cpu({key: output[key] for key in ("preds_vision", "preds_action")}))

        handles = [model.net.register_forward_pre_hook(pre, with_kwargs=True), model.net.register_forward_hook(post)]
        with torch.inference_mode():
            result = model.generate_samples_from_batch(
                copy.deepcopy(capture["batch"]),
                seed=[fixture["args"]["seed"] + sample],
                num_steps=fixture["args"]["steps"],
                guidance=fixture["args"]["guidance"],
                shift=1.0,
            )
        results.append(cpu(result))
        for handle in handles:
            handle.remove()
    if rank == 0:
        torch.save(
            {
                "args": vars(args),
                "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                "outputs": outputs,
                "sample_results": results,
                "sampling_inputs": inputs,
                "routing_audit": routing_audit,
                "diagnostic_only": args.diagnostic_full_gen,
                "tf32": False,
            },
            args.output,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--checkpoint", default="/share/project/eai_pwm/models/Cosmos3-Nano-DCP/model")
    p.add_argument("--config-base", default="/share/project/eai_pwm/models/Cosmos3-Nano")
    p.add_argument("--tokenizer", default="/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--towers", required=True, choices=["two_tower", "three_tower"])
    p.add_argument("--execution", choices=["native", "joint", "nano_compatible"], default="native")
    p.add_argument("--attention-backend", choices=["auto", "nano", "flex"], default="auto")
    p.add_argument("--projection-mode", choices=["auto", "routed"], default="routed")
    p.add_argument(
        "--diagnostic-full-gen",
        action="store_true",
        help="DIAGNOSTIC ONLY: duplicate full GEN evaluation to isolate rounding; never a production configuration",
    )
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--joint-backend", choices=["flex", "math"], default="flex")
    p.add_argument(
        "--sample",
        action="store_true",
        help="Run the complete sampler from explicitly frozen preparation, instead of teacher-forcing every step",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        from cosmos_framework.utils import distributed

        distributed.init()
    rank = torch.distributed.get_rank() if world > 1 else 0
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
    from cosmos_framework.model.generator.mot import action_tower
    from cosmos_framework.model.generator.mot import attention as mot_attention

    if args.precision == "float32":
        mot_attention.attention = math_attention
    if args.joint_backend == "math":
        action_tower._joint_attention = joint_math

    if args.diagnostic_full_gen:
        from tools.action_tower_diagnostics import install_full_gen_diagnostic

        install_full_gen_diagnostic()

    config = Cosmos3OmniConfig.from_pretrained(args.config_base)
    cfg = config.model["config"]
    cfg.update(
        tower_structure=args.towers,
        action_tower_init="copy_gen" if args.towers == "three_tower" else "checkpoint",
        action_tower_execution=args.execution,
        action_tower_attention_backend=args.attention_backend,
        action_tower_projection_mode=args.projection_mode,
        precision=args.precision,
        load_vision_tokenizer=False,
        sound_gen=False,  # Fixture contains no sound; do not load an unused audio tokenizer.
    )
    cfg["vlm_config"]["tokenizer"]["config_variant"] = "hf"
    cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] = args.tokenizer
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    model = Cosmos3OmniModel.from_pretrained_dcp(
        Path(args.checkpoint),
        config=config,
        parallelism_config=ParallelismConfig(data_parallel_shard_degree=world, enable_inference_mode=True),
        compile_config=CompileConfig(enabled=False),
    ).model.eval()
    # Model setup configures compiler defaults; reassert reference precision
    # after setup as well, before any measured forward.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixture = torch.load(args.inputs, map_location="cpu", weights_only=False)
    dtype = getattr(torch, args.precision)
    assert next(model.net.parameters()).dtype == dtype
    print(f"Loaded {args.towers}, dtype={dtype}, world_size={world}, TF32 disabled", flush=True)
    routing_audit = {}
    if args.towers == "three_tower" and not args.diagnostic_full_gen:
        routing_audit = install_routing_audit(model)
    if args.sample:
        sample_frozen(model, fixture, args, rank, routing_audit)
        if world > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    # No resampling. Each model receives the exact same tensor values, including
    # the original random-normal realization, sigma, clean condition and mRoPE.
    outputs, traces, routing, structures = [], [], [], {}
    attention_trace = {}
    from cosmos_framework.model.attention import frontend

    native_backend_calls = []
    original_choose = frontend.choose_backend

    def trace_backend(*positional, **kwargs):
        backend = original_choose(*positional, **kwargs)
        if len(native_backend_calls) < 2:
            native_backend_calls.append(backend)
        return backend

    frontend.choose_backend = trace_backend
    # Capture the first layer's exact post-RoPE operands and pre-O output.
    # Canonical order is [UND | GEN], [tokens, heads, dim] for both backends.
    first_attention = model.net.language_model.model.layers[0].self_attn
    original_dispatch = first_attention.dispatch_attention_fn

    def trace_dispatch(q, k, v, *positional, **kwargs):
        for name, pack in (("q", q), ("k", k), ("v", v)):
            attention_trace[name] = cpu(torch.cat([get_und_seq(pack), get_gen_seq(pack)]))
        result = original_dispatch(q, k, v, *positional, **kwargs)
        attention_trace["out"] = cpu(torch.cat([get_und_seq(result[0]), get_gen_seq(result[0])]))
        first_attention.dispatch_attention_fn = original_dispatch
        return result

    first_attention.dispatch_attention_fn = trace_dispatch
    original_joint = action_tower._joint_attention

    def trace_joint(q, k, v, **kwargs):
        for name, value in (("q", q), ("k", k), ("v", v)):
            attention_trace[name] = cpu(value.squeeze(0).transpose(0, 1))
        result = original_joint(q, k, v, **kwargs)
        attention_trace["out"] = cpu(result.squeeze(0).transpose(0, 1).flatten(1))
        action_tower._joint_attention = original_joint
        return result

    action_tower._joint_attention = trace_joint
    current = [0, 0]
    handles = []

    def layer_hook(index):
        def hook(module, inputs, output):
            pack = inputs[0]
            u, g = get_und_seq(output[0]), get_gen_seq(output[0])
            ai = (
                pack["_action_indices"]
                if "_action_indices" in pack
                else torch.where(torch.isin(pack["_full_indices"], active[0].action.sequence_indexes))[0]
            )
            entry = {"action": cpu(g[ai]), "und_last": cpu(u[-1:]), "video_first": cpu(g[:1])}
            if current == [0, 0]:
                entry.update(und=cpu(u), gen=cpu(g))
            traces[-1][index] = entry

        return hook

    def module_hook(name):
        def hook(module, inputs, output):
            if current == [0, 0]:
                structures[name] = {"in": cpu(inputs[0]), "out": cpu(output)}

        return hook

    if rank == 0:
        for i, layer in enumerate(model.net.language_model.model.layers):
            handles.append(layer.register_forward_hook(layer_hook(i)))
            if i == 0:
                for name, module in layer.named_modules():
                    if name and not list(module.children()):
                        handles.append(module.register_forward_hook(module_hook(name)))

    active = [None]
    for sample, capture in enumerate(fixture["captures"]):
        assert capture["packed_steps"], "Fixture must be created with --freeze-inputs"
        for step, original in enumerate(capture["packed_steps"]):
            current[:] = [sample, step]
            packed = copy.deepcopy(original)
            assert packed.action.robot_state_token_indexes.numel() == 0
            packed.to_cuda()
            active[0] = packed
            traces.append({})
            with torch.inference_mode():
                out = model.net(packed_seq=packed)
            values = cpu({key: out[key] for key in ("preds_vision", "preds_action")})
            for tensors in values.values():
                assert all(torch.isfinite(tensor).all() for tensor in tensors)
            outputs.append(values)
            routing.append(
                {
                    "sample": sample,
                    "step": step,
                    "state_tokens": int(packed.action.robot_state_token_indexes.numel()),
                    "action_sequence_indexes": cpu(packed.action.sequence_indexes),
                    "video_sequence_indexes": cpu(packed.vision.sequence_indexes),
                    "und_sequence_indexes": cpu(packed.text_indexes),
                    "action_timesteps": cpu(packed.action.timesteps),
                    "action_condition_mask": cpu(packed.action.condition_mask),
                    "action_branch": "GEN_ACTION" if args.towers == "three_tower" else "GEN",
                    "video_branch": "GEN_VIDEO" if args.towers == "three_tower" else "GEN",
                }
            )
            print(
                f"rank={rank} replay sample={sample} step={step}, action dtype={out['preds_action'][0].dtype}",
                flush=True,
            )
    for handle in handles:
        handle.remove()
    if rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "args": vars(args),
                "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                "outputs": outputs,
                "traces": traces,
                "routing": routing,
                "structures": structures,
                "attention": attention_trace,
                "native_backend_calls": native_backend_calls,
                "routing_audit": routing_audit,
                "torch": torch.__version__,
                "diagnostic_only": args.diagnostic_full_gen,
                "tf32": False,
            },
            args.output,
        )
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
