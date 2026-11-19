# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimentally save only the exact FA3 forward op during recomputation.

TP and FSDP must use the same selected layer count and prepared inputs in a
comparison. The saved outputs, compiler behavior, actual GPU memory, kernel
counts and complete numerical trajectory require GPU qualification. Disabled
unless this wrapper is explicitly invoked; not a training default.

The profiled TP4 prepared update had 216 FA3 forward kernels totaling 0.466 s:
avoiding the 108 recomputations has an approximate 0.233 s kernel-time ceiling,
not a wall-time guarantee and not enough alone to meet the throughput target.
For the observed BF16 [59648, 8, 128] TP4 query/output, one output and its
FP32 LSE cost about 118 MiB/layer, or 4.16 GiB across 36 layers per active
microstep. FSDP's 32 heads imply about 16.6 GiB. These are tensor-size
estimates, excluding other saved tensors, nonempty accumulators, overlapping
microsteps, allocator and compiler effects; all-rank GPU peaks must be measured.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

FLASH3_FORWARD_SCHEMA = "flash_attn_3::_flash_attn_forward"


def estimate_saved_bytes(tokens, heads, head_dim, value_bytes, layers):
    values = (tokens, heads, head_dim, value_bytes, layers)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("Memory estimate requires positive integer dimensions and layer count")
    output = tokens * heads * head_dim * value_bytes
    logsumexp = tokens * heads * 4
    return {
        "formula": "selected layers * (tokens * query_heads * head_dim * value_bytes + tokens * query_heads * 4)",
        "tokens": tokens,
        "query_heads": heads,
        "head_dim": head_dim,
        "value_bytes": value_bytes,
        "selected_layers": layers,
        "output_bytes_per_layer": output,
        "lse_bytes_per_layer": logsumexp,
        "estimated_bytes_per_active_microstep": layers * (output + logsumexp),
        "excludes": "other saved tensors, nonempty accumulators, allocator/compiler effects and overlapping microsteps",
        "gpu_peak_measured": False,
    }


def selected_layers(names, count):
    names = tuple(names)
    if not 0 < count <= len(names):
        raise ValueError("Selective layer count must fit the model or pipeline stage")
    return frozenset(names[-count:])


def make_policy(observation, *, schema=FLASH3_FORWARD_SCHEMA):
    from torch.utils.checkpoint import CheckpointPolicy

    def policy(ctx, func, *args, **kwargs):
        op_schema = getattr(func, "_schema", None)
        if getattr(op_schema, "name", None) == schema and getattr(op_schema, "overload_name", None) == "":
            phase = "recompute" if ctx.is_recompute else "forward"
            observation[phase] += 1
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return policy


def wrap_block(block, ac, policy):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return checkpoint_wrapper(
        block,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
        preserve_rng_state=ac.preserve_rng_state,
        determinism_check=ac.determinism_check,
    )


def install(layer_count, observation):
    import torch

    from cosmos_framework.model.generator.mot import parallelize_unified_mot as parallel
    from cosmos_framework.training_backends.megatron_tp_pp import model, train

    policy = make_policy(observation)

    def native_apply_ac(native_model, ac):
        if ac.mode != "full":
            raise ValueError("The shared FA3 candidate requires a full-AC baseline")
        layers = native_model.model.layers
        chosen = selected_layers(layers._modules, layer_count)
        for name, block in list(layers.named_children()):
            result = wrap_block(block, ac, policy) if name in chosen else parallel._apply_full_ac(block, ac)
            layers.register_module(name, result)
        visual = parallel._trainable_visual_blocks(native_model)
        if visual is not None:
            for name, block in list(visual.named_children()):
                visual.register_module(name, parallel._apply_full_ac(block, ac))
        observation["native_selected_layers"] += len(chosen)

    def optimize(stage, model_config):
        ac, cfg = model_config.activation_checkpointing, model_config.compile
        if ac.mode != "full":
            raise ValueError("The shared FA3 candidate requires a full-AC baseline")
        if cfg.enabled and (cfg.compiled_region != "language" or cfg.use_cuda_graphs):
            raise ValueError("The candidate requires the existing per-block compilation policy")
        chosen = selected_layers(stage.layers, layer_count)
        options = model.inductor_compile_options(cfg)
        for name, block in list(stage.layers.items()):
            block = wrap_block(block, ac, policy) if name in chosen else parallel._apply_full_ac(block, ac)
            if cfg.enabled:
                block = torch.compile(block, fullgraph=True, dynamic=cfg.compile_dynamic, options=options or None)
            stage.layers[name] = block
        observation["tp_selected_layers"] += len(chosen)

    parallel.apply_ac = native_apply_ac
    train.optimize_stage = optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--selective-layers", type=int, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    parser.add_argument("--estimate-tokens", type=int)
    parser.add_argument("--estimate-heads", type=int)
    parser.add_argument("--estimate-head-dim", type=int)
    parser.add_argument("--estimate-value-bytes", type=int)
    args, remainder = parser.parse_known_args()
    estimate_args = (args.estimate_tokens, args.estimate_heads, args.estimate_head_dim, args.estimate_value_bytes)
    if any(value is not None for value in estimate_args) and not all(value is not None for value in estimate_args):
        raise ValueError("Provide all four memory estimate arguments or none")
    estimate = (
        estimate_saved_bytes(*estimate_args, args.selective_layers)
        if all(value is not None for value in estimate_args)
        else None
    )
    rank = os.environ.get("RANK", "")
    if not rank.isdecimal():
        raise ValueError("Numeric distributed RANK required")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / f"flash3_selective_ac.rank{rank}.json"
    if output.exists():
        raise FileExistsError(output)
    observation = {"forward": 0, "recompute": 0, "native_selected_layers": 0, "tp_selected_layers": 0}
    error = None
    try:
        if args.shared_input_reduction:
            from tools.tp_pp_shared_input_reduction import install as install_shared

            install_shared()
        install(args.selective_layers, observation)
        sys.path.insert(0, str(args.entry.resolve().parent))
        sys.argv = [str(args.entry), *remainder]
        runpy.run_path(str(args.entry), run_name="__main__")
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        with output.open("x") as handle:
            json.dump(
                {
                    "rank": int(rank),
                    "scope": "experimental FA3 output save; no GPU or full-model acceptance implied",
                    "operator_schema": FLASH3_FORWARD_SCHEMA,
                    "selected_layers": args.selective_layers,
                    "memory_estimate": estimate,
                    "policy_trace_counts": observation,
                    "failure": error,
                },
                handle,
                indent=2,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
