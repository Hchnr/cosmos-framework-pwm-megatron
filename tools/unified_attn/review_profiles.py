# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify common profile inputs and source revision before preparing the review index."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
reference = None
results = {}
for run in args.runs:
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["args"]["nsys"] and manifest["args"]["warmup"] >= 10
    assert json.loads((run / "exit.json").read_text())["returncode"] == 0
    review = json.loads((run / "nsys_review.json").read_text())
    breakdown = json.loads((run / "trace_breakdown.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert review["optimizer_updates"] == 2 and len(review["microbatches"]) == 4
    assert not review["possible_compilation_ranges"]
    assert not breakdown["module_or_jit_api_calls"], (run, "module/JIT API within capture")
    sources = {
        key: value
        for key, value in json.loads((run / "source_hashes.json").read_text()).items()
        if key.startswith("cosmos_framework/")
    }
    inputs, rng, graph_checks = [], [], []
    for rank in range(8):
        records = [json.loads(line) for line in (run / f"rank{rank}.jsonl").read_text().splitlines()]
        steps = [row for row in records if row["event"] == "step"]
        inputs.append([row["input_fingerprints"] for row in steps])
        rng.append([row["rng_fingerprints"] for row in steps])
        boundary = manifest["initial_step"] + manifest["args"]["warmup"]
        last_warm = next(row for row in steps if row["iteration"] == boundary)
        if "compiler_counters" in last_warm:
            assert all(
                row["compiler_counters"]["stats"] == last_warm["compiler_counters"]["stats"]
                for row in steps
                if row["iteration"] > boundary
            ), (run, rank, "new graph in capture")
            graph_checks.append("unchanged")
        else:
            graph_checks.append("not recorded; inspect cache and trace evidence")
    identity = {"inputs": inputs, "rng": rng, "production_sources": sources}
    if reference is None:
        reference = identity
    assert identity == reference, (run, "profile replay/source mismatch")
    backend = manifest["args"]["backend"]
    assert backend not in results
    results[backend] = {
        "run": str(run),
        "report_sha256": review["sha256"],
        "report_bytes": review["bytes"],
        "optimizer_steps": review["actual_optimizer_steps"],
        "kernel_count": review["kernel_count"],
        "rank_peak_allocated_gib": summary["rank_peak_allocated_gib"],
        "rank_peak_reserved_gib": summary["rank_peak_reserved_gib"],
        "compiler_counter_checks": graph_checks,
    }
args.output.write_text(json.dumps({"replay_and_sources_verified": True, "profiles": results}, indent=2))
print("Verified profiles:", len(results))
