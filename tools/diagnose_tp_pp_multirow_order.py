# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Isolate packing and Gloo collective rounding from model math/autograd."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist


def compare(left, right):
    delta = left.to(torch.float64) - right.to(torch.float64)
    return {
        "exact": bool(torch.equal(left, right)),
        "different": int(torch.count_nonzero(left != right)),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm() / left.to(torch.float64).norm().clamp_min(1e-30)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    reports = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(1024 + rank)
        values = [torch.randn(shape).to(dtype) for shape in ((3, 6), (5, 8), (2, 4))]
        local_packet = torch.cat([x.flatten() for x in values])
        lengths = [x.numel() for x in values]
        before = [p.reshape(x.shape) for p, x in zip(local_packet.split(lengths), values, strict=True)]
        assert all(torch.equal(x, y) for x, y in zip(values, before, strict=True))
        separate = []
        for value in values:
            item = value.clone()
            dist.all_reduce(item)
            separate.append(item)
        packed = local_packet.clone()
        dist.all_reduce(packed)
        packed_rows = [p.reshape(x.shape) for p, x in zip(packed.split(lengths), values, strict=True)]
        all_ranks = [torch.empty_like(local_packet) for _ in range(world)]
        dist.all_gather(all_ranks, local_packet)
        canonical = torch.stack([x.float() for x in all_ranks]).sum(dim=0).to(dtype)
        canonical_rows = [p.reshape(x.shape) for p, x in zip(canonical.split(lengths), values, strict=True)]
        reports.append(
            {
                "dtype": str(dtype),
                "pack_unpack_exact": True,
                "separate_vs_packed": [compare(x, y) for x, y in zip(separate, packed_rows, strict=True)],
                "separate_vs_fp32_sum": [compare(x, y) for x, y in zip(separate, canonical_rows, strict=True)],
                "packed_vs_fp32_sum": [compare(x, y) for x, y in zip(packed_rows, canonical_rows, strict=True)],
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(
        json.dumps({"rank": rank, "world": world, "cases": reports}, indent=2) + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
