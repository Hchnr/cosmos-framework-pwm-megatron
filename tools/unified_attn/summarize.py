# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validate eight-rank completion and summarize steady optimizer-step measurements."""

import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", type=Path)
args = parser.parse_args()
rows = []
for run in args.runs:
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["args"]
    end = manifest["initial_step"] + config["warmup"] + (2 if config["nsys"] else config["steps"])
    ranks = []
    compiler_available = []
    for rank in range(8):
        records = [json.loads(line) for line in (run / f"rank{rank}.jsonl").read_text().splitlines()]
        assert records[-1] == {"event": "end", "rank": rank, "iteration": end}, (run, rank)
        steps = [
            r for r in records if r["event"] == "step" and r["iteration"] > manifest["initial_step"] + config["warmup"]
        ]
        assert len(steps) == (2 if config["nsys"] else config["steps"])
        assert all(r["microbatches"] == 2 for r in steps)
        assert all(all(abs(loss) < float("inf") for loss in r["losses"]) for r in steps)
        all_steps = [r for r in records if r["event"] == "step"]
        if config["warmup"] > 0:
            boundary = manifest["initial_step"] + config["warmup"]
            last_warm = next(r for r in all_steps if r["iteration"] == boundary)
            available = "compiler_counters" in last_warm
            if not config["nsys"]:
                assert available, (run, rank, "formal run lacks compiler counters")
            if available:
                assert all(r["compiler_counters"]["stats"] == last_warm["compiler_counters"]["stats"] for r in steps), (
                    run,
                    rank,
                    "new compiler graph in measured window",
                )
            compiler_available.append(available)
        ranks.append(steps)
    wall = [max(ranks[r][s]["seconds"] for r in range(8)) for s in range(len(ranks[0]))]
    token_counts = [sum(t["und"] + t["gen"] for r in ranks for t in r[s]["real_tokens"]) for s in range(len(wall))]
    gen_counts = [sum(t["gen"] for r in ranks for t in r[s]["real_tokens"]) for s in range(len(wall))]
    physical_tokens = 8 * 2 * (3072 + config["gen_tokens"]) * len(wall)
    row = {
        "run": run.name,
        "backend": config["backend"],
        "profiled": config["nsys"],
        "gen_tokens": config["gen_tokens"],
        "steps": len(wall),
        "compiler_counter_available_ranks": compiler_available,
        "step_mean_s": statistics.mean(wall),
        "step_median_s": statistics.median(wall),
        "step_p90_s": sorted(wall)[min(len(wall) - 1, int(len(wall) * 0.9))],
        "effective_tokens_per_second": sum(token_counts) / sum(wall),
        "effective_gen_tokens_per_second": sum(gen_counts) / sum(wall),
        "padding_fraction": 1 - sum(token_counts) / physical_tokens,
        "step_seconds_all_ranks": [[s["seconds"] for s in rank] for rank in ranks],
        "global_step_seconds": wall,
        "rank_peak_allocated_gib": [max(s["peak_allocated_bytes"] for s in rank) / 2**30 for rank in ranks],
        "rank_peak_reserved_gib": [max(s["peak_reserved_bytes"] for s in rank) / 2**30 for rank in ranks],
    }
    (run / "summary.json").write_text(json.dumps(row, indent=2))
    rows.append(row)
print(json.dumps(rows, indent=2))
