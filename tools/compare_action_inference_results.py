"""Compare baseline and final Action benchmarks with matching workloads.

Primary metric is WAM action-ready latency with no predicted-video output.
Full-video measurements are auxiliary and never define the primary speedup.
This checks measurement contracts, not numerical accuracy or closed-loop quality.
"""

import argparse
import json
from pathlib import Path


def compare(baseline: dict, optimized: dict) -> dict:
    for key in ("checkpoint", "input_sha256", "torch", "cuda", "gpu", "seed", "attention_backend", "batch_size"):
        if baseline.get(key) != optimized.get(key):
            raise ValueError(f"Incompatible {key}: {baseline.get(key)!r} != {optimized.get(key)!r}")
    action_only = bool(baseline.get("action_only", False))
    if action_only != bool(optimized.get("action_only", False)):
        raise ValueError("Baseline and final must use the same delivery mode")
    if baseline.get("cuda_visible_devices") != optimized.get("cuda_visible_devices"):
        raise ValueError("Compare on the same physical GPU; cross-GPU results belong in a separate experiment")
    before = {(row["steps"], row["guidance"]): row for row in baseline["runs"]}
    after = {(row["steps"], row["guidance"]): row for row in optimized["runs"]}
    if before.keys() != after.keys():
        raise ValueError("Step/CFG configurations differ")
    rows = []
    for key in before:
        a, b = before[key], after[key]
        if min(len(a["measurements"]), len(b["measurements"])) < 3:
            raise ValueError(f"Need at least three measurements for {key}")
        if any(not m["finite"] or not m["inactive_slots_zero"] for r in (a, b) for m in r["measurements"]):
            raise ValueError(f"Invalid action contract for {key}")
        row = {"steps": key[0], "guidance": key[1], "baseline": {}, "optimized": {}, "speedup": {}, "reduction_pct": {}}
        for metric in ("sample_seconds", "action_ready_seconds", "total_seconds"):
            av, bv = a[metric]["median"], b[metric]["median"]
            row["baseline"][metric] = av
            row["optimized"][metric] = bv
            row["speedup"][metric] = av / bv
            row["reduction_pct"][metric] = 100 * (1 - bv / av)
        row["primary_action_ready_speedup"] = row["speedup"]["action_ready_seconds"]
        if not action_only:
            # Auxiliary delivery-mode comparison only; never the primary WAM speedup.
            row["full_to_action_ready_speedup"] = a["total_seconds"]["median"] / b["action_ready_seconds"]["median"]
        row["baseline_peak_allocated_gib"] = max(m["peak_allocated_gib"] for m in a["measurements"])
        row["optimized_peak_allocated_gib"] = max(m["peak_allocated_gib"] for m in b["measurements"])
        row["optimized_peak_reserved_gib"] = max(m["peak_reserved_gib"] for m in b["measurements"])
        rows.append(row)
    return {
        "workload_matched": True,
        "primary_metric": "action_ready_seconds",
        "primary_comparison": "WAM baseline action-ready vs final action-ready; video decode/save excluded on both sides",
        "action_only_run": action_only,
        "auxiliary_total_mode": "action_only_with_transfer_and_file_save" if action_only else "full_video_output",
        "checkpoint": baseline["checkpoint"],
        "gpu": baseline["gpu"],
        "cuda_visible_devices": baseline.get("cuda_visible_devices"),
        "runs": rows,
        "limitations": [
            "Only fixed-input short-run latency; p95 is descriptive, not a production tail SLA",
            "Full-to-action-ready changes delivery mode; it is not compute-only acceleration",
            "Action-ready excludes initial input preparation and HTTP; see serving results for client wall time",
            "Numerical and closed-loop quality acceptance are separate",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-regression-pct", type=float, default=5.0)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.optimized.read_text()))
    result["sources"] = {"baseline": str(args.baseline), "optimized": str(args.optimized)}
    result["performance_regression_passed"] = all(
        r["reduction_pct"][metric] >= -args.max_regression_pct
        for r in result["runs"]
        for metric in (result["primary_metric"],)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["runs"]:
        print(
            f"steps={row['steps']} guidance={row['guidance']}: "
            f"WAM action-ready {row['primary_action_ready_speedup']:.3f}x "
            f"({row['baseline']['action_ready_seconds']:.4f}s -> {row['optimized']['action_ready_seconds']:.4f}s)"
        )
    if not result["performance_regression_passed"]:
        raise SystemExit("Performance regression exceeds the requested threshold")


if __name__ == "__main__":
    main()
