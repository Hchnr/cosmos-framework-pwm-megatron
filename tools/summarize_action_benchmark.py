"""Summarize measured update windows from ActionTrainingBenchmark rank JSONL files."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory: Path) -> dict:
    files = sorted(directory.glob("rank_*.jsonl"))
    if len(files) != 8:
        raise ValueError(f"Expected eight rank logs in {directory}, found {len(files)}")
    ranks = [[json.loads(line) for line in path.read_text().splitlines()] for path in files]
    warmup = [[row for row in rows if not row["measured"]] for rows in ranks]
    ranks = [[row for row in rows if row["measured"]] for rows in ranks]
    steps = [row["iteration"] for row in ranks[0]]
    if not steps or any([r["iteration"] for r in rows] != steps for rows in ranks):
        raise ValueError("Missing or mismatched measured iterations")
    slowest_steps = [max(rows[i]["seconds"] for rows in ranks) for i in range(len(steps))]
    window = max(sum(row["seconds"] for row in rows) for rows in ranks)
    samples = sum(row["global_samples"] for row in ranks[0])
    totals = {
        name: sum(row.get(name, 0) for rows in ranks for row in rows)
        for name in ("und_token_length", "vision_token_length", "action_token_length")
    }
    result = dict(
        first_iteration=steps[0],
        last_iteration=steps[-1],
        measured_steps=len(steps),
        slowest_rank_window_seconds=window,
        mean_step_seconds=window / len(steps),
        p50_step_seconds=statistics.median(slowest_steps),
        p90_step_seconds=sorted(slowest_steps)[min(len(steps) - 1, int(0.9 * len(steps)))],
        samples_per_second=samples / window,
        peak_allocated_gib=max(row["peak_allocated_bytes"] for rows in ranks for row in rows) / 2**30,
        peak_reserved_gib=max(row["peak_reserved_bytes"] for rows in ranks for row in rows) / 2**30,
        effective_tokens_per_second={k: v / window for k, v in totals.items()},
        total_samples=samples,
        token_totals=totals,
    )

    if all("compile_unique_graphs" in row for rows in ranks for row in rows):
        result["compile_measured_window"] = {
            path.stem: {
                "unique_graphs_first": rows[0]["compile_unique_graphs"],
                "unique_graphs_last": rows[-1]["compile_unique_graphs"],
                "graph_breaks_first": rows[0]["compile_graph_breaks"],
                "graph_breaks_last": rows[-1]["compile_graph_breaks"],
            }
            for path, rows in zip(files, ranks, strict=True)
        }
        # Include the last warmup sample so compilation on the first measured
        # update cannot be mistaken for a stable window.
        boundaries = [before[-1:] + rows for before, rows in zip(warmup, ranks, strict=True)]
        result["compile_stable_in_measured_window"] = all(
            len({row["compile_unique_graphs"] for row in rows}) == 1
            and len({row["compile_graph_breaks"] for row in rows}) == 1
            for rows in boundaries
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
