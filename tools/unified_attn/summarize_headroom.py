# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Quantify conditional FA3 removal budgets; never treat a GPU sum as wall time."""

import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/v0.2.7_unified_attn"
result = {"micro": {}, "trace": {}}
for case, directory in [("two", "headroom_two_v1"), ("three", "headroom_three_v3")]:
    files = sorted((PLAN / "artifacts" / directory).glob("rank*.json"))
    assert len(files) == 8, (case, len(files))
    ranks = []
    for path in files:
        d = json.loads(path.read_text())
        assert len(d["rounds"]) == 3
        medians = {
            n: {
                k: statistics.mean(r["medians"][n][k] for r in d["rounds"])
                for k in ("forward_ms", "backward_ms", "fb_ms")
            }
            for n in d["rounds"][0]["medians"]
        }
        baseline = "split" if case == "two" else "split3"
        gains = {n: 100 * (1 - v["fb_ms"] / medians[baseline]["fb_ms"]) for n, v in medians.items()}
        row = {
            "rank": d["rank"],
            "source": str(path.relative_to(ROOT)),
            "medians": medians,
            "fb_reduction_vs_split_pct": gains,
            "max_output_relative_l2": max(max(v["output_relative_l2"]) for v in d["validation"].values()),
            "max_source_gradient_relative_l2": max(
                max(v["source_gradient_relative_l2"])
                for v in d["validation"].values()
                if v["source_gradient_relative_l2"] is not None
            ),
        }
        if case == "two":
            row["prepacked_vs_grouped_fb_reduction_pct"] = 100 * (
                1 - medians["prepacked_grouped"]["fb_ms"] / medians["grouped"]["fb_ms"]
            )
        ranks.append(row)
    result["micro"][case] = ranks
for backend in ("split_fa3_fork", "unified_fa3_grouped"):
    run = PLAN / "artifacts/runs" / f"final_l0_v1_profile_r0_{backend}"
    d = json.loads((run / "trace_breakdown.json").read_text())
    selected = {}
    helper = und = 0
    for name, metrics in d["all_kernel_names"].items():
        if name.startswith("triton_") and ("cat" in name or "index" in name):
            selected[name] = metrics
        if name.startswith("void") and "flash::" in name:
            main = "FlashAttnFwdSm90<" in name or "FlashAttnBwdSm90<" in name
            if not main:
                helper += metrics["gpu_duration_ms_sum"]
            if main and backend == "split_fa3_fork" and metrics["gpu_duration_ms_sum"] < 100:
                und += metrics["gpu_duration_ms_sum"]
    db = sqlite3.connect(run / "rank0.sqlite")
    kernels = db.execute(
        "SELECT k.start,k.end,k.streamId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start"
    ).fetchall()
    prev = {}
    gap = 0
    for start, end, stream, name in kernels:
        if name.startswith("void") and "flash::" in name:
            gap += max(0, start - prev.get(stream, start))
        prev[stream] = max(end, prev.get(stream, 0))
    db.close()
    elapsed = sum(json.loads((run / "summary.json").read_text())["global_step_seconds"]) * 1000
    copies = sum(v["gpu_duration_ms_sum"] for v in selected.values())
    budget = copies + helper + und + gap / 1e6
    result["trace"][backend] = {
        "selected_pack_and_fused_kernels": selected,
        "selected_gpu_sum_ms": copies,
        "all_native_helpers_ms": helper,
        "und_main_ms": und,
        "same_stream_gaps_before_native_ms": gap / 1e6,
        "profile_two_step_wall_ms": elapsed,
        "optimistic_removal_budget_ms": budget,
        "budget_div_wall_pct": 100 * budget / elapsed,
        "caveat": "Conditional remove-only exposure, preserving GEN main-kernel durations and remaining work; fused arithmetic and required helpers cannot actually all vanish. Stream gaps may be dependencies; GPU sums overlap. Not a universal bound on kernel redesign or CPU graph changes.",
    }
u, g = 3072, 98304
q = (u + g) * 32 * 128 * 2
kv = 2 * (2 * u + g) * 8 * 128 * 2
und_duplicate = 2 * u * 8 * 128 * 2
result["logical_packing"] = {
    "q_bytes": q,
    "kv_bytes": kv,
    "duplicate_und_kv_bytes": und_duplicate,
    "flat_cat_read_write_bytes": 2 * (q + kv),
    "flat_cat_at_3p35TBps_ms": 2 * (q + kv) / 3.35e12 * 1000,
    "caveat": "Logical materialization sizes, not measured HBM traffic; peak-bandwidth division is a time lower bound, not a speedup upper bound.",
}
cache_source = Path(
    "/tmp/cosmos-unified-attn/final_l0_v1_profile_r0_unified_fa3_grouped/rank0/5d/c5dw6ir4do3u23edtev7vnnlc2wxbygovd7cw5ich72wpwpudi6a.py"
)
import hashlib

code = cache_source.read_text().splitlines()
result["compiled_buffer_alias_evidence"] = {
    "path": str(cache_source),
    "sha256": hashlib.sha256(cache_source.read_bytes()).hexdigest(),
    "start_line": 3270,
    "end_line": 3365,
    "text": "\n".join(code[3269:3365]),
}
three = result["micro"]["three"]
base = statistics.mean(2 * r["medians"]["split3"]["forward_ms"] + r["medians"]["split3"]["backward_ms"] for r in three)
ideal = statistics.mean(
    2 * r["medians"]["prepacked_grouped2"]["forward_ms"] + r["medians"]["prepacked_grouped2"]["backward_ms"]
    for r in three
)
result["three_ac_cost_model"] = {
    "split_2F_plus_B_ms": base,
    "ideal_2F_plus_B_ms": ideal,
    "local_reduction_pct": 100 * (1 - ideal / base),
    "projected_ms_saved_per_36layer_ga2_step": 72 * (base - ideal),
    "caveat": "Analytical 2F+B from separately measured medians, NOT a measured AC or training step. Assumes 36 active layers, GA2, same synthetic geometry and no overlap.",
    "shared_full_kv_bytes_saved_vs_duplicate_group3": 74368 * 8 * 128 * 2 * 2,
}
result["protocol"] = (
    "8 isolated GPUs; 3 rounds x30 shuffled observations per variant per GPU,10 warmups; same-process rounds, NOT independent job replicas; 3tower synthetic capacities only."
)
(PLAN / "analysis/fa3_headroom.json").write_text(json.dumps(result, indent=2))
for case, ranks in result["micro"].items():
    print(case)
    for row in ranks:
        print(row["rank"], row["fb_reduction_vs_split_pct"])
print({k: {n: v for n, v in d.items() if n != "selected_pack_and_fused_kernels"} for k, d in result["trace"].items()})
