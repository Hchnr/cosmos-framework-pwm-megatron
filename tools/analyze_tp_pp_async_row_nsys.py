# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check actual concurrent NCCL and compute intervals in a synthetic Nsight trace."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

_TID_MASK = (1 << 24) - 1


def intervals(rows, start, end):
    events = []
    count = Counter()
    for left, right, name in rows:
        left, right = max(start, left), min(end, right)
        if left >= right:
            continue
        kind = "nccl" if "ncclDevKernel" in name else "compute"
        count[kind] += 1
        events.extend(((left, kind, 1), (right, kind, -1)))
    events.sort()
    active, previous, elapsed = Counter(), start, Counter()
    for timestamp, kind, change in events:
        label = (
            "overlap"
            if active["nccl"] and active["compute"]
            else "nccl_only"
            if active["nccl"]
            else "compute_only"
            if active["compute"]
            else "no_kernel"
        )
        elapsed[label] += timestamp - previous
        active[kind] += change
        previous = timestamp
    elapsed["no_kernel"] += end - previous
    return {"kernel_counts": dict(count), "seconds": {name: value / 1e9 for name, value in elapsed.items()}}


def analyze(path, *, warmups, steps):
    if warmups < 1 or steps <= warmups:
        raise ValueError("Require warmups and measured steps")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        ranges = connection.execute(
            "SELECT text, globalTid, start, end FROM NVTX_EVENTS "
            "WHERE text LIKE 'async_row_probe.%' AND end IS NOT NULL"
        ).fetchall()
        report = []
        for text, global_tid, start, end in ranges:
            parts = text.split(".")
            if len(parts) != 3 or parts[1] not in ("baseline", "delayed") or not parts[2].startswith("step"):
                raise ValueError(f"Unexpected probe range {text!r}")
            step = int(parts[2][4:])
            if not 0 <= step < steps:
                raise ValueError(f"Unexpected step {step}")
            if step < warmups:
                continue
            global_pid = global_tid & ~_TID_MASK
            rows = connection.execute(
                "SELECT k.start, k.end, s.value, k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id=k.demangledName "
                "WHERE k.globalPid=? AND k.start < ? AND k.end > ?",
                (global_pid, end, start),
            ).fetchall()
            devices = {row[3] for row in rows}
            if len(devices) != 1:
                raise ValueError(f"Expected exactly one CUDA device for {text}: {devices}")
            result = intervals([(a, b, name) for a, b, name, _ in rows], start, end)
            if result["kernel_counts"].get("nccl", 0) < 3 or result["kernel_counts"].get("compute", 0) < 3:
                raise ValueError(f"Missing row GEMMs or NCCL kernels for {text} device {devices}")
            report.append({"mode": parts[1], "step": step, "device": devices.pop(), **result})
    finally:
        connection.close()
    expected = {
        (device, mode, step)
        for device in range(4)
        for mode in ("baseline", "delayed")
        for step in range(warmups, steps)
    }
    found = [(item["device"], item["mode"], item["step"]) for item in report]
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError("Missing, repeated, or unexpected measured probe ranges")
    per_device = []
    for device in range(4):
        averages = {}
        for mode in ("baseline", "delayed"):
            values = [
                item["seconds"].get("overlap", 0.0)
                for item in report
                if item["device"] == device and item["mode"] == mode
            ]
            averages[mode] = sum(values) / len(values)
        per_device.append({"device": device, "mean_overlap_seconds": averages})
    # Evidence on one rank suffices for this synthetic probe; it does not prove
    # that every rank overlaps or that end-to-end training becomes faster.
    demonstrated = any(
        item["mean_overlap_seconds"]["delayed"] >= item["mean_overlap_seconds"]["baseline"] + 1e-6
        for item in per_device
    )
    return {
        "passed": demonstrated,
        "scope": "real-shaped synthetic forward-only BF16 TP4; not end-to-end training performance",
        "threshold_seconds": 1e-6,
        "warmups": warmups,
        "steps": steps,
        "per_device": per_device,
        "ranges": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.database, warmups=args.warmups, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise SystemExit("Trace did not demonstrate measurable NCCL/compute overlap")


if __name__ == "__main__":
    main()
