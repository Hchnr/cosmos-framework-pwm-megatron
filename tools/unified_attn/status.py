# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read-only progress for the long, sequential eight-GPU benchmark matrix."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="final_l0_v1")
args = parser.parse_args()
runs = sorted(
    (ROOT / "plans/v0.2.7_unified_attn/artifacts/runs").glob(args.tag + "_bench_*/manifest.json"),
    key=lambda path: path.stat().st_mtime,
)
if not runs:
    print("No benchmark runs yet")
    raise SystemExit
manifest = runs[-1]
run = manifest.parent
config = json.loads(manifest.read_text())
last_steps = []
stable = True
for rank in range(8):
    path = run / f"rank{rank}.jsonl"
    if not path.exists():
        last_steps.append(config["initial_step"])
        continue
    # The final line can be mid-write while training appends it.
    lines = path.read_text().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith("\n")]
    steps = [row for row in rows if row["event"] == "step"]
    last_steps.append(steps[-1]["iteration"] if steps else config["initial_step"])
    boundary = config["initial_step"] + config["args"]["warmup"]
    warm = next((row for row in steps if row["iteration"] == boundary), None)
    if warm:
        stable &= all(
            row["compiler_counters"]["stats"] == warm["compiler_counters"]["stats"]
            for row in steps
            if row["iteration"] > boundary
        )
completed = sum((path.parent / "summary.json").is_file() for path in runs)
result = {
    "completed_runs": completed,
    "current": run.name,
    "rank_step_range": [min(last_steps), max(last_steps)],
    "completed_measurement_steps_all_ranks": max(
        0, min(last_steps) - config["initial_step"] - config["args"]["warmup"]
    ),
    "no_new_graphs_in_recorded_measurements": stable,
}
exit_path = run / "exit.json"
if exit_path.exists():
    result["exit"] = json.loads(exit_path.read_text())
print(json.dumps(result))
