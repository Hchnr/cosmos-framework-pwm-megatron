# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify replay invariants before comparing complete training measurements."""

import argparse
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
reference_inputs = None
reference_rng = None
reference_sources = None
reference_workload = None
results = {}
for run in args.runs:
    m = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert not m["args"]["nsys"], "Profiled times are not benchmark samples"
    assert not m["args"]["audit"], "Audit I/O must not enter performance measurements"
    assert m["args"]["steps"] >= 30 and m["args"]["warmup"] >= 10
    workload = (
        m["args"]["gen_tokens"],
        m["args"]["steps"],
        m["args"]["warmup"],
        m["initial_step"],
        m["fa3_provider"],
        m["args"].get("filter_oversized", False),
    )
    if reference_workload is None:
        reference_workload = workload
    assert workload == reference_workload, (run, "workload mismatch")
    assert json.loads((run / "exit.json").read_text())["returncode"] == 0, run
    sources = {
        k: v
        for k, v in json.loads((run / "source_hashes.json").read_text()).items()
        if k.startswith("cosmos_framework/")
    }
    records = [[json.loads(line) for line in (run / f"rank{rank}.jsonl").read_text().splitlines()] for rank in range(8)]
    all_steps = [[r for r in rank if r["event"] == "step"] for rank in records]
    for rank, steps in enumerate(all_steps):
        boundary = m["initial_step"] + m["args"]["warmup"]
        warm = next(r for r in steps if r["iteration"] == boundary)
        expected_graphs = warm["compiler_counters"]["stats"]
        assert all(r["compiler_counters"]["stats"] == expected_graphs for r in steps if r["iteration"] > boundary), (
            run,
            rank,
            "new compiler graph in measurement window",
        )
    inputs = [[r["input_fingerprints"] for r in rank] for rank in all_steps]
    rng = [[r["rng_fingerprints"] for r in rank] for rank in all_steps]
    if reference_inputs is None:
        reference_inputs, reference_rng, reference_sources = inputs, rng, sources
    assert inputs == reference_inputs, (run, "input replay mismatch")
    assert rng == reference_rng, (run, "RNG replay mismatch")
    assert sources == reference_sources, (run, "production source mismatch")
    results.setdefault(m["args"]["ac"], []).append(summary)
aggregate = {}
for backend, rounds in results.items():
    assert len(rounds) >= 3, (backend, "fewer than three independent runs")
    times = [r["step_median_s"] for r in rounds]
    # Student-t interval over independent run medians, not correlated steps.
    assert len(times) == 3, "This interval uses t(2); adapt the critical value for other round counts"
    half_width = 4.302652729911275 * statistics.stdev(times) / math.sqrt(len(times))
    aggregate[backend] = {
        "runs": [r["run"] for r in rounds],
        "median_step_s": statistics.median(times),
        "round_median_range_s": [min(times), max(times)],
        "mean_of_round_medians_95pct_t_interval_s": [
            statistics.mean(times) - half_width,
            statistics.mean(times) + half_width,
        ],
        "interval_caveat": "Three independent runs, Student-t df=2; approximate normality assumption, limited precision",
        "mean_step_s": statistics.mean(r["step_mean_s"] for r in rounds),
        "mean_round_p90_s": statistics.mean(r["step_p90_s"] for r in rounds),
        "effective_tokens_per_second": statistics.mean(r["effective_tokens_per_second"] for r in rounds),
        "effective_gen_tokens_per_second": statistics.mean(r["effective_gen_tokens_per_second"] for r in rounds),
        "padding_fraction": statistics.mean(r["padding_fraction"] for r in rounds),
        "rank_peak_allocated_gib": [max(r["rank_peak_allocated_gib"][i] for r in rounds) for i in range(8)],
        "rank_peak_reserved_gib": [max(r["rank_peak_reserved_gib"][i] for r in rounds) for i in range(8)],
    }
baseline = aggregate["full"]["median_step_s"]
for backend, result in aggregate.items():
    result["speedup_vs_b0"] = baseline / result["median_step_s"]
args.output.write_text(json.dumps({"replay_verified": True, "results": aggregate}, indent=2))
print(json.dumps(aggregate, indent=2))
