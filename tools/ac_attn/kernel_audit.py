# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compare actual NV FA3 and cuBLAS kernels by their host launch phase."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def inspect(run):
    db = sqlite3.connect(str(run / "rank0.sqlite"))
    ranges = db.execute("""SELECT coalesce(s.value,n.text),n.start,n.end
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL""").fetchall()
    backward = [(a, b) for name, a, b in ranges if name == "cosmos.backward.autograd"]
    forward = [
        (a, b) for name, a, b in ranges if name and name.startswith("cosmos.train.") and name.endswith(".forward")
    ]
    assert len(forward) == len(backward) == 4, "Expected four complete microbatches"
    launches = {}
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for correlation, start, name in db.execute(
                f"SELECT a.correlationId,a.start,s.value FROM {table} a JOIN StringIds s ON a.nameId=s.id"
            ):
                if "Launch" in name:
                    launches[correlation] = start
    totals = defaultdict(lambda: defaultdict(lambda: {"count": 0, "gpu_duration_ms_sum": 0.0}))
    gemm_names = defaultdict(lambda: defaultdict(int))
    names = defaultdict(set)
    for name, start, end, correlation in db.execute("""SELECT s.value,k.start,k.end,k.correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id"""):
        if "flash::FlashAttnFwdSm90<" in name:
            category = "fa_forward_core"
        elif "flash::FlashAttnBwdSm90<" in name:
            category = "fa_backward_core"
        elif "flash::" in name:
            category = "fa_helpers"
        elif name.startswith("nvjet_"):
            category = "nvjet_gemm"
        elif "gemm" in name.lower():
            category = "other_gemm"
        elif "nccl" in name.lower():
            category = "communication"
        else:
            category = "other"
        launch = launches.get(correlation)
        phase = "unattributed"
        if launch is not None:
            phase = (
                "backward"
                if any(a <= launch <= b for a, b in backward)
                else "forward"
                if any(a <= launch <= b for a, b in forward)
                else "outside_forward_backward"
            )
        totals[category][phase]["count"] += 1
        totals[category][phase]["gpu_duration_ms_sum"] += (end - start) / 1e6
        names[category].add(name)
        if category in ("nvjet_gemm", "other_gemm"):
            gemm_names[name][phase] += 1
    return {
        "run": str(run),
        "categories": totals,
        "gemm_launch_counts_by_name_and_phase": gemm_names,
        "kernel_names": {key: sorted(value) for key, value in names.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("full", type=Path)
    parser.add_argument("attention_only", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full, saved = inspect(args.full), inspect(args.attention_only)
    for row, replay in ((full, 288), (saved, 0)):
        categories = row["categories"]
        assert categories["fa_forward_core"]["forward"]["count"] == 288
        assert categories["fa_forward_core"]["backward"]["count"] == replay
        assert categories["fa_backward_core"]["backward"]["count"] == 288
        for category in ("fa_forward_core", "fa_backward_core"):
            assert sum(v["count"] for p, v in categories[category].items() if p not in ("forward", "backward")) == 0
        assert categories["fa_backward_core"]["forward"]["count"] == 0
    assert full["gemm_launch_counts_by_name_and_phase"] == saved["gemm_launch_counts_by_name_and_phase"]
    result = {
        "verified": True,
        "scope": "NV FA3, 36 layers, two attention branches, four microbatches, rank 0",
        "warning": "GPU duration sums overlap communication/compute and are not wall time. GEMM backward includes both replay and ordinary gradients; generated graphs distinguish those operations.",
        "full": full,
        "attention_only": saved,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print("Verified: FA replay 288 -> 0; original forward/backward 288 each; GEMM names/phase counts identical")


if __name__ == "__main__":
    main()
