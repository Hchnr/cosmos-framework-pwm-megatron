# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Retain trace kernel/phase attribution; summed GPU durations are not wall time."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
db = sqlite3.connect(str(args.run / "rank0.sqlite"))
ranges = db.execute("""SELECT coalesce(s.value,n.text),n.start,n.end
    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL""").fetchall()
phases = defaultdict(lambda: {"count": 0, "host_duration_ms_sum": 0.0})
for name, start, end in ranges:
    if name:
        phases[name]["count"] += 1
        phases[name]["host_duration_ms_sum"] += (end - start) / 1e6
kernels = db.execute("""SELECT s.value,k.start,k.end,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName=s.id""").fetchall()
by_name = defaultdict(lambda: {"count": 0, "gpu_duration_ms_sum": 0.0})
for name, start, end, _ in kernels:
    by_name[name]["count"] += 1
    by_name[name]["gpu_duration_ms_sum"] += (end - start) / 1e6
api = []
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
    if table in tables:
        api += db.execute(
            f"SELECT a.correlationId,a.start,s.value FROM {table} a JOIN StringIds s ON a.nameId=s.id"
        ).fetchall()
launches = {correlation: start for correlation, start, name in api if "Launch" in name}
backward_ranges = [(start, end) for name, start, end in ranges if name == "cosmos.backward.autograd"]
attention = []
for name, metrics in by_name.items():
    lower = name.lower()
    if any(key in lower for key in ("flash", "flex_attention", "fmha")):
        counts = {"forward_phase": 0, "backward_phase": 0, "unattributed": 0}
        for kname, _, _, correlation in kernels:
            if kname != name:
                continue
            launch = launches.get(correlation)
            label = (
                "unattributed"
                if launch is None
                else "backward_phase"
                if any(start <= launch <= end for start, end in backward_ranges)
                else "forward_phase"
            )
            counts[label] += 1
        attention.append({"name": name, **metrics, "launch_phase_counts": counts})
copies = []
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    copies = [
        {"copy_kind": kind, "count": count, "bytes": size, "gpu_duration_ms_sum": duration / 1e6}
        for kind, count, size, duration in db.execute(
            "SELECT copyKind,count(*),sum(bytes),sum(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
        )
    ]
result = {
    "run": args.run.name,
    "warning": "GPU duration sums overlap communication/compute and are not step wall times; host NVTX ranges measure enqueue spans. Attention-name matches include helper/fused kernels and require review.",
    "attention_named_kernels": attention,
    "all_kernel_names": by_name,
    "nvtx_host_phases": phases,
    "memory_copies": copies,
    "module_or_jit_api_calls": [
        {"name": name, "start": start}
        for _, start, name in api
        if any(key in name.lower() for key in ("moduleload", "linkcreate", "linkcomplete", "compile"))
    ],
}
(args.run / "trace_breakdown.json").write_text(json.dumps(result, indent=2))
print(
    args.run.name,
    "attention-named kernels:",
    len(attention),
    "module/JIT calls:",
    len(result["module_or_jit_api_calls"]),
)
