"""Summarize complete distributed microbatch traces without touching training.

Iteration-end log numbers are one greater than the trace iteration. CUDA/host
phase timings include collective waiting; they are not isolated kernel costs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def summarize(launch: Path, ga: int, tail_bytes: int = 0):
    groups = defaultdict(dict)
    world = None
    for path in sorted((launch / "load_balance_trace").glob("rank_*.jsonl")):
        with path.open("rb") as stream:
            offset = max(0, path.stat().st_size - tail_bytes) if tail_bytes else 0
            stream.seek(offset)
            if offset:
                stream.readline()
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if row.get("record_type") != "microbatch":
                    continue
                if world is not None and world != row["world_size"]:
                    raise ValueError("Trace directory contains mixed world sizes")
                world = row["world_size"]
                groups[row["iteration"] + 1][row["rank"], row["batch_index"]] = row
    steps = {}
    with (launch / "node0.log").open(errors="replace") as stream:
        for line in stream:
            if "[RANK " in line and "[RANK 0]" not in line:
                continue
            match = re.search(
                r"Iteration (\d+):.*?(?:Time: ([0-9.]+)s|Iter speed: ([0-9.]+) seconds per iteration)", line
            )
            if match:
                steps[int(match[1])] = float(match[2] or match[3])
    results = []
    for iteration, unique in sorted(groups.items()):
        rows = list(unique.values())
        ranks = defaultdict(set)
        for row in rows:
            ranks[row["rank"]].add(row["microbatch_in_iteration"])
        complete = (
            set(ranks) == set(range(world))
            and all(slots == set(range(ga)) for slots in ranks.values())
            and len(rows) == world * ga
        )
        entry = {
            "iteration": iteration,
            "step_seconds": steps.get(iteration),
            "complete": complete,
            "records": len(rows),
            "expected_records": world * ga,
        }
        if complete:
            worst = sorted(rows, key=lambda r: r["dataloader_host_ms"], reverse=True)[:3]
            entry["largest_data_wait_seconds"] = worst[0]["dataloader_host_ms"] / 1000
            per_rank = defaultdict(list)
            for row in rows:
                per_rank[row["rank"]].append(row)
            rank = max(per_rank, key=lambda rank: sum(r["dataloader_host_ms"] for r in per_rank[rank]))
            entry["largest_rank_total_data_wait"] = {
                "rank": rank,
                "seconds": sum(r["dataloader_host_ms"] for r in per_rank[rank]) / 1000,
                "microbatches": [
                    {
                        "slot": r["microbatch_in_iteration"],
                        "sources": r["data"]["dataset_names"],
                        "seconds": r["dataloader_host_ms"] / 1000,
                    }
                    for r in sorted(per_rank[rank], key=lambda r: r["microbatch_in_iteration"])
                ],
            }
            entry["largest_wait_records"] = [
                {
                    "rank": r["rank"],
                    "batch_index": r["batch_index"],
                    "microbatch_in_iteration": r["microbatch_in_iteration"],
                    "sources": r["data"]["dataset_names"],
                    "data_wait_seconds": r["dataloader_host_ms"] / 1000,
                    "sample_ids": r["active_sample_ids"],
                }
                for r in worst
            ]
        results.append(entry)
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch": str(launch.resolve()),
        "world_size": world,
        "gradient_accumulation": ga,
        "trace_tail_bytes_per_rank": tail_bytes,
        "latest_logged_timed_step": max(steps, default=None),
        "latest_complete_trace_step": max((row["iteration"] for row in results if row["complete"]), default=None),
        "scope": "Only complete all-rank steps receive wait attribution. "
        "Largest individual wait is evidence, not an exact decomposition of step time. "
        "Rank totals sum that rank's own accumulation slots, never maxima from different ranks. "
        "First step can include initialization; CUDA phases include collective waiting.",
        "steps": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tail-bytes-per-rank", type=int, default=0, help="0 reads the full trace; bounded tails suit polling"
    )
    args = parser.parse_args()
    if args.gradient_accumulation < 1:
        parser.error("gradient accumulation must be positive")
    if args.tail_bytes_per_rank < 0:
        parser.error("tail bytes must be nonnegative")
    report = summarize(args.launch_dir, args.gradient_accumulation, args.tail_bytes_per_rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    complete = [row for row in report["steps"] if row["complete"]]
    print(json.dumps({"world_size": report["world_size"], "complete_steps": len(complete)}))


if __name__ == "__main__":
    main()
