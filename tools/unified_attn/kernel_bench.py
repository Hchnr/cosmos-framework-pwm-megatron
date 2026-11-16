# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Eight-rank attention integration benchmark using recorded real pack geometry.

Times include packing/copies and Python dispatch. Pure CUDA kernel attribution
comes from the separate full-training Nsight capture. No model/optimizer state.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.attention import two_way_attention
from cosmos_framework.model.generator.mot.joint_attention import joint_attention, prepare_joint_attention

parser = argparse.ArgumentParser()
parser.add_argument("--backend", required=True)
parser.add_argument("--geometry", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--gen-tokens", type=int, default=98304)
parser.add_argument("--frozen-und", action="store_true")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--repeats", type=int, default=30)
parser.add_argument("--reference", type=Path)
parser.add_argument("--write-reference", action="store_true")
args = parser.parse_args()
rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
geometry = json.loads((args.geometry / f"geometry_rank{rank}.json").read_text())
lens, modes = geometry["split_lens"], geometry["attn_modes"]
uids, gids, start = [], [], 0
for size, mode in zip(lens, modes):
    (uids if mode == "causal" else gids).extend(range(start, start + size))
    start += size
pack = sequence_pack_from_packed_sequence(
    torch.zeros(start, 1, device="cuda"),
    modes,
    lens,
    geometry["sample_lens"],
    torch.tensor(uids, device="cuda"),
    torch.tensor(gids, device="cuda"),
    fixed_causal_len=3072,
    fixed_full_len=args.gen_tokens,
)
metadata_times = []
for _ in range(10):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    metadata = None if args.backend == "split_default" else prepare_joint_attention(pack, args.backend)
    torch.cuda.synchronize()
    metadata_times.append((time.perf_counter() - begin) * 1000)
rounds = []
validation = None
for round_id in range(3):
    torch.manual_seed(42 + rank + 1000 * round_id)
    sources = [
        torch.randn(n, h, 128, device="cuda", dtype=torch.bfloat16).requires_grad_(i == 1 or not args.frozen_und)
        for h in (32, 8, 8)
        for i, n in enumerate((3072, args.gen_tokens))
    ]
    dund = torch.randn(len(uids), 32 * 128, device="cuda", dtype=torch.bfloat16)
    dgen = torch.randn(len(gids), 32 * 128, device="cuda", dtype=torch.bfloat16)
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup + args.repeats):
        for source in sources:
            source.grad = None
        dist.barrier()
        torch.cuda.synchronize()
        if index == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        qp, kp, vp = [from_und_gen_splits(*sources[i : i + 2], pack) for i in (0, 2, 4)]
        out = two_way_attention(qp, kp, vp) if metadata is None else joint_attention(qp, kp, vp, metadata)
        torch.cuda.synchronize()
        forward = time.perf_counter() - begin
        loss = (get_gen_seq(out)[: len(gids)] * dgen).sum()
        if not args.frozen_und:
            loss = loss + (get_und_seq(out)[: len(uids)] * dund).sum()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - begin
        if round_id == 0 and index == 0 and args.reference is not None:
            values = {"output": torch.cat((get_und_seq(out)[: len(uids)], get_gen_seq(out)[: len(gids)]))}
            values.update({f"gradient_{i}": t.grad for i, t in enumerate(sources) if t.requires_grad})
            path = args.reference / f"rank{rank}.pt"
            if args.write_reference:
                args.reference.mkdir(parents=True, exist_ok=True)
                assert not path.exists(), path
                torch.save({k: v.detach().cpu() for k, v in values.items()}, path)
                validation = {"reference_written": str(path)}
            else:
                reference = torch.load(path, map_location="cpu", weights_only=True)
                validation = {}
                for key, value in values.items():
                    ref = reference[key].to(value.device).float()
                    diff = value.float() - ref
                    error = (diff.norm() / ref.norm().clamp_min(1e-12)).item()
                    validation[key] = {"relative_l2": error, "max_abs": diff.abs().max().item()}
                    assert torch.isfinite(value).all() and error < 0.03, (args.backend, key, validation[key])
                del reference, ref, diff
            del values
            torch.cuda.empty_cache()
        if index >= args.warmup:
            timings.append({"forward_ms": forward * 1000, "backward_ms": backward * 1000})
        del qp, kp, vp, out, loss
    rounds.append(
        {
            "round": round_id,
            "timings": timings,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    del sources, dund, dgen
args.output.mkdir(parents=True, exist_ok=True)
result = {
    "backend": args.backend,
    "rank": rank,
    "frozen_und": args.frozen_und,
    "gen_tokens": args.gen_tokens,
    "geometry": geometry,
    "metadata_ms": metadata_times,
    "metadata_median_ms": statistics.median(metadata_times),
    "rounds": rounds,
    "validation": validation,
}
(args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
dist.destroy_process_group()
