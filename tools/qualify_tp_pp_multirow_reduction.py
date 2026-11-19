# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify opt-in row-sum grouping with the production-I/O tiny CUDA oracle.

This invokes the existing four-update model/state comparison; pass --compile to
exercise full activation checkpointing and fullgraph compilation. TP4 uses a
four-KV-head variant of the tiny fixture. This is not 36-layer acceptance.
"""

import argparse
import dataclasses
import functools
import json
import os
import sys
from pathlib import Path

import torch


def upgrade_kv_heads(model, shape):
    """Expand the tiny fixture from two to four KV heads for TP4."""
    shape = dataclasses.replace(shape, kv_heads=4)
    model.language_model.config.num_key_value_heads = 4
    model.language_model.model.small_action_shape = shape
    for layer in model.language_model.model.layers:
        attention = layer.self_attn
        attention.num_key_value_heads = 4
        attention.num_key_value_groups = 1
        for suffix in ("", "_moe_gen"):
            for name in ("k_proj", "v_proj"):
                old = getattr(attention, name + suffix)
                replacement = torch.nn.Linear(old.in_features, 64, bias=old.bias is not None)
                replacement.weight.requires_grad_(old.weight.requires_grad)
                if old.bias is not None:
                    replacement.bias.requires_grad_(old.bias.requires_grad)
                setattr(attention, name + suffix, replacement)
        action = layer.action_block_moe_gen_action
        action.shape = dataclasses.replace(action.shape, kv_heads=4)
        for name in ("k_proj", "v_proj"):
            old = getattr(action, name)
            replacement = torch.nn.Linear(old.in_features, 64, bias=old.bias is not None)
            replacement.weight.requires_grad_(old.weight.requires_grad)
            if old.bias is not None:
                replacement.bias.requires_grad_(old.bias.requires_grad)
            setattr(action, name, replacement)
    return model, shape


def update_collectives(events):
    """Count recorded functional all-reduces nested inside each update range."""
    markers = [event for event in events if event.name == "multirow_forward_backward_update"]
    counts = {id(marker): 0 for marker in markers}
    for event in events:
        if event.name != "_c10d_functional::all_reduce":
            continue
        parent = event.cpu_parent
        while parent is not None:
            if id(parent) in counts:
                counts[id(parent)] += 1
                break
            parent = parent.cpu_parent
    return [counts[id(marker)] for marker in markers]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, choices=(2, 4), default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--profile-collectives", action="store_true")
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    if args.shared_input_reduction:
        from tools.tp_pp_shared_input_reduction import install as install_shared

        install_shared()
    original_make_native = smoke.make_native
    if args.tp == 4:

        def make_native():
            return upgrade_kv_heads(*original_make_native())

        smoke.make_native = make_native
    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=1
    )
    if not args.baseline:
        from tools.run_tp_pp_multirow_reduction import install

        install()
    sys.argv = [sys.argv[0], "--output", str(args.output), *remainder]
    # Set the device before profiler creation; never measure another rank.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.reset_peak_memory_stats()
    counts = None
    if args.profile_collectives:
        original_forward_backward = smoke.ModelParallelDriver.forward_backward

        def observed(driver, *positional, **keywords):
            with torch.profiler.record_function("multirow_forward_backward_update"):
                return original_forward_backward(driver, *positional, **keywords)

        smoke.ModelParallelDriver.forward_backward = observed
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            smoke.main()
        counts = update_collectives(profiler.events())
        if len(counts) != 4 or any(count == 0 for count in counts):
            raise RuntimeError(f"Functional all-reduce events not observed for all four updates: {counts}")
    else:
        smoke.main()
    mode = "baseline" if args.baseline else "candidate"
    sidecar = args.output.with_name(f"{args.output.stem}.multirow_{mode}.rank{os.environ['RANK']}.json")
    with sidecar.open("x") as handle:
        json.dump(
            {
                "passed": True,
                "tp": args.tp,
                "full_ac_compile": "--compile" in remainder,
                "shared_input_reduction": args.shared_input_reduction,
                "mode": mode,
                "scope": "FP32 production-I/O tiny model; no full-model or performance acceptance",
                "full_model_acceptance": False,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "functional_all_reduce_calls_per_update": counts,
                "profiling_scope": "whole model forward/backward, not CUDA kernel count or steady-state timing"
                if counts is not None
                else None,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
