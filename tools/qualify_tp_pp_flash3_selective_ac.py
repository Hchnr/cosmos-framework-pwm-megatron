# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify real H100 FA3 selective recomputation; no full-model claim.

Runs a production video_varlen_attend call with BF16 Q/K/V and real CUDA FA3
under eager and fullgraph Inductor. For each, compare full AC against exact-op
selective AC, including outputs, Q/K/V gradients, profiler CUDA forward-kernel
counts, and peak allocated memory. CUDA profiler overhead invalidates timing.
"""

import argparse
import gc
import json
import os
import traceback
from pathlib import Path


def forward_kernel_count(events):
    return sum(1 for event in events if event.device_type.name == "CUDA" and "FlashAttnFwdSm90" in event.name)


def compare_tensors(reference, candidate, *, atol=1e-3, rtol=5e-3):
    import torch

    if len(reference) != len(candidate):
        return {"passed": False, "error": "tensor count differs"}
    rows = []
    for left, right in zip(reference, candidate, strict=True):
        if left.shape != right.shape or left.dtype != right.dtype:
            return {"passed": False, "error": "shape or dtype differs"}
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            return {"passed": False, "error": "nonfinite tensor"}
        rows.append(
            {
                "exact": torch.equal(left, right),
                "max_absolute_error": (left.float() - right.float()).abs().max().item(),
                "passed": torch.allclose(left, right, atol=atol, rtol=rtol),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "atol": atol, "rtol": rtol, "tensors": rows}


def run():
    # This module reads the environment during import. No CUDA work occurs in
    # either the coordinator or CPU contract tests importing this file.
    os.environ["COSMOS_ATTENTION_DETERMINISTIC"] = "1"
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from cosmos_framework.model.generator.mot.small_action_attention import video_varlen_attend, video_varlen_mask
    from tools.run_tp_pp_flash3_selective_ac import make_policy, wrap_block

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This qualification requires an available Hopper GPU and installed production FA3")
    torch.manual_seed(739)
    torch.cuda.manual_seed_all(739)
    ids_q = torch.tensor([0] * 128 + [1] * 128, device="cuda")
    ids_k = torch.tensor(([0, 1, 0] * 128), device="cuda")
    mask = video_varlen_mask(ids_q, ids_k, 2)
    inputs = (
        torch.randn(256, 4, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(384, 2, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(384, 2, 128, device="cuda", dtype=torch.bfloat16),
    )

    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return video_varlen_attend(q, k, v, mask)

    from types import SimpleNamespace

    ac = SimpleNamespace(preserve_rng_state=True, determinism_check="default")
    records = {}
    outputs = {}
    for compiled in (False, True):
        for selective in (False, True):
            label = f"{'inductor' if compiled else 'eager'}_{'selective' if selective else 'full'}"
            counts = {"forward": 0, "recompute": 0, "native_selected_layers": 0, "tp_selected_layers": 0}
            block = (
                wrap_block(Attention(), ac, make_policy(counts))
                if selective
                else checkpoint_wrapper(Attention(), preserve_rng_state=True, determinism_check="default")
            )
            if compiled:
                block = torch.compile(block, fullgraph=True, dynamic=False)

            def execute(current_block=block):
                q, k, v = (value.detach().clone().requires_grad_() for value in inputs)
                out = current_block(q, k, v)
                out.float().square().mean().backward()
                return [out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()]

            # Fully warm the same block before resetting peaks and profiling.
            warm = execute()
            torch.cuda.synchronize()
            del warm
            gc.collect()
            torch.cuda.synchronize()
            start_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                result = execute()
                torch.cuda.synchronize()
            peaks = torch.cuda.max_memory_allocated()
            events = prof.events()
            records[label] = {
                "flash3_forward_cuda_kernel_count": forward_kernel_count(events),
                "matched_cuda_kernel_names": sorted(
                    {
                        event.name
                        for event in events
                        if event.device_type.name == "CUDA" and "FlashAttnFwdSm90" in event.name
                    }
                ),
                "start_allocated_bytes": start_allocated,
                "peak_allocated_bytes": peaks,
                "peak_increase_bytes": peaks - start_allocated,
                "policy_trace_counts": counts,
            }
            outputs[label] = [value.cpu() for value in result]
            del result, block, prof, events, execute
            torch.cuda.synchronize()
            gc.collect()
    comparisons = {
        mode: compare_tensors(outputs[f"{mode}_full"], outputs[f"{mode}_selective"]) for mode in ("eager", "inductor")
    }
    failures = []
    for mode in ("eager", "inductor"):
        if records[f"{mode}_full"]["flash3_forward_cuda_kernel_count"] != 2:
            failures.append(f"{mode}: full AC did not produce exactly two observed FA3 forward kernels")
        if records[f"{mode}_selective"]["flash3_forward_cuda_kernel_count"] != 1:
            failures.append(f"{mode}: selective AC did not reduce observed FA3 forward kernels to one")
        if not comparisons[mode]["passed"]:
            failures.append(f"{mode}: FA3 outputs or Q/K/V gradients differ beyond BF16 tolerances")
    return {
        "passed": not failures,
        "failures": failures,
        "scope": "one Hopper GPU, production BF16 FA3 attention, eager and fullgraph Inductor; not full model",
        "gpu": torch.cuda.get_device_name(),
        "cases": records,
        "comparisons": comparisons,
        "full_model_acceptance": False,
        "throughput_acceptance": False,
        "timing_valid": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        report = run()
    except BaseException:
        report = {"passed": False, "error": traceback.format_exc(), "full_model_acceptance": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise SystemExit("FA3 numerical or kernel-count qualification failed")


if __name__ == "__main__":
    main()
