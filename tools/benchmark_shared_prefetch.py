# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Controlled CPU payload benchmark; no claim about production model throughput.

Run each configuration in a fresh interpreter. PSS sums count shared pages once
within this process tree. cgroup readings include other jobs in the container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from cosmos_framework.data.generator.dataflow import CosmosDataLoader, IdentityProcessor, MapDistributor
from cosmos_framework.data.generator.dataflow.loader import JointCosmosDataLoader
from cosmos_framework.data.generator.dataflow.shared_prefetch import CosmosDataStream, SharedPrefetchDataLoader


class _Payloads(torch.utils.data.Dataset):
    def __init__(self, mib, delay):
        self.elements = mib * 1024 * 1024 // 4
        self.delay = delay

    def __len__(self):
        return 10_000

    def __getitem__(self, index):
        time.sleep(self.delay)
        return {"id": index, "payload": torch.full((self.elements,), float(index))}


def _memory():
    root = psutil.Process()
    processes = [root, *root.children(recursive=True)]
    pss = 0
    for process in processes:
        try:
            pss += process.memory_full_info().pss
        except psutil.NoSuchProcess:
            pass
    result = {"tree_pss_bytes": pss, "processes": len(processes)}
    for name in ("memory.current", "memory.peak", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result[f"container_{name}"] = path.read_text().strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["legacy_per_source", "shared_prefetch"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--payload-mib", type=int, default=16)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--memory-every", type=int, default=4, help="0 samples only after timed batches")
    parser.add_argument("--decode-delay", type=float, default=0.01)
    parser.add_argument("--compute-delay", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.batches or args.memory_every < 0:
        parser.error("require 0 <= warmup < batches and memory-every >= 0")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    streams = {}
    for source in range(args.sources):
        loader = CosmosDataLoader(
            MapDistributor(_Payloads(args.payload_mib, args.decode_delay), shuffle=False, name=f"s{source}"),
            IdentityProcessor(),
            batch_size=1,
            num_workers=1,
            prefetch_factor=args.prefetch,
            persistent_workers=True,
        )
        streams[f"s{source}"] = {
            "dataloader": CosmosDataStream(loader.dataset, 1) if args.backend == "shared_prefetch" else loader,
            "ratio": 1,
        }
    if args.backend == "shared_prefetch":
        joint = SharedPrefetchDataLoader(streams, num_workers=args.workers, prefetch_factor=args.prefetch, seed=42)
    else:
        if args.workers != args.sources:
            raise ValueError("Legacy benchmark has exactly one logical/physical worker per source")
        joint = JointCosmosDataLoader(streams, seed=42, selection_mode="exact_periodic")
    before = _memory()
    checksum = hashlib.sha256()
    waits, memory = [], []
    started = time.perf_counter()
    iterator = iter(joint)
    steady_started = None
    steady_elapsed = None
    try:
        for index in range(args.batches):
            if index == args.warmup:
                steady_started = time.perf_counter()
            fetch = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - fetch)
            checksum.update(f"{batch['dataset_name']}:{batch['id'].item()}".encode())
            # Check payload correctness without another large host copy.
            assert batch["payload"][0, 0].item() == batch["id"].item()
            time.sleep(args.compute_delay)
            del batch
            if index == args.batches - 1:
                steady_elapsed = time.perf_counter() - steady_started
            if (args.memory_every and index % args.memory_every == 0) or index == args.batches - 1:
                memory.append(_memory())
        elapsed = time.perf_counter() - started
        result = {
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "trace_sha256": checksum.hexdigest(),
            "before": before,
            "memory_samples": memory,
            "peak_tree_pss_bytes": max(sample["tree_pss_bytes"] for sample in memory),
            "elapsed_seconds": elapsed,
            "steady_elapsed_seconds": steady_elapsed,
            "steady_batches_per_second": (args.batches - args.warmup) / steady_elapsed,
            "steady_fetch_mean_seconds": float(np.mean(waits[args.warmup :])),
            "steady_fetch_p95_seconds": float(np.percentile(waits[args.warmup :], 95)),
            "prefetch_stats": joint.prefetch_stats if args.backend == "shared_prefetch" else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "memory_samples"}))
    finally:
        if args.backend == "shared_prefetch":
            joint.close()
        elif joint._iterators:
            for iterator in joint._iterators:
                iterator._shutdown_workers()


if __name__ == "__main__":
    main()
