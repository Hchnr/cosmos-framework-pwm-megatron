# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture the NCCL/compute overlap of three real-shaped row projections.

This is a forward-only synthetic probe. Its CUDA/NVTX trace and its wall times
are not full-model throughput or numerical acceptance evidence.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from tools.run_tp_pp_async_row_reduction import LaunchRowSum, WaitRowSum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.warmups < 1 or args.steps <= args.warmups:
        raise ValueError("Require at least one warmup and one measured step")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError("This probe uses the actual TP4 row-projection widths")
    torch.manual_seed(9700 + rank)
    # Real physical extents for action, short UND and VIDEO o_proj TP shards.
    specs = ((2944, 1280), (5120, 4096), (59648, 4096))
    inputs = [torch.randn(tokens, 1024, device=device, dtype=torch.bfloat16) for tokens, _ in specs]
    weights = [torch.randn(width, 1024, device=device, dtype=torch.bfloat16) * 0.01 for _, width in specs]
    group = dist.group.WORLD.group_name

    def execute(delayed):
        if delayed:
            pending = [
                LaunchRowSum.apply(F.linear(value, weight), group)
                for value, weight in zip(inputs, weights, strict=True)
            ]
            return [WaitRowSum.apply(value) for value in pending]
        return [
            WaitRowSum.apply(LaunchRowSum.apply(F.linear(value, weight), group))
            for value, weight in zip(inputs, weights, strict=True)
        ]

    # Strictly check identical input and collective order before profiling.
    with torch.no_grad():
        expected = execute(False)
        actual = execute(True)
        exact = [torch.equal(left, right) for left, right in zip(expected, actual, strict=True)]
        if not all(exact):
            raise ValueError(f"Different row results on rank {rank}: {exact}")
        del expected, actual
        torch.cuda.synchronize()
        for step in range(args.steps):
            for name, delayed in (("baseline", False), ("delayed", True)):
                dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"async_row_probe.{name}.step{step}")
                output = execute(delayed)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                del output
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"rank{rank}.json"
    with target.open("x") as handle:
        json.dump(
            {
                "rank": rank,
                "world": world,
                "scope": "synthetic forward-only BF16 TP4 row GEMM/NCCL overlap; not end-to-end throughput",
                "shapes": specs,
                "steps": args.steps,
                "warmups": args.warmups,
                "outputs_exact": exact,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
