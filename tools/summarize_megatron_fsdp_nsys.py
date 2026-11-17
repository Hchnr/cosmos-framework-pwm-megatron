"""Produce compact profile comparisons while keeping raw reports untouched."""

import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path

METRICS = [
    "wall_seconds",
    "gpu_busy_union_seconds",
    "nccl_union_seconds",
    "non_nccl_kernel_and_memops_union_seconds",
    "overlap_seconds",
    "nccl_only_seconds",
    "non_nccl_only_seconds",
    "gpu_idle_seconds",
]


def summarize(analysis):
    by_backend = collections.defaultdict(list)
    for profile in analysis["profiles"]:
        by_backend[Path(profile["report"]).parent.name].append(profile)
    results = {}
    for backend, profiles in by_backend.items():
        updates = [u for p in profiles for u in p["updates"]]
        record = {metric: statistics.mean(u[metric] for u in updates) for metric in METRICS}
        record["ranks"] = len(profiles)
        record["observations"] = len(updates)
        record["cpu_phase_wall_seconds"] = {
            k: statistics.mean(u["cpu_phase_wall_seconds"][k] for u in updates)
            for k in ["forward", "backward", "optimizer_step"]
        }
        cats = sorted({k for u in updates for k in u["categories"]})
        record["categories"] = {
            k: {
                field: statistics.mean(u["categories"].get(k, {}).get(field, 0) for u in updates)
                for field in ["count", "sum_gpu_seconds"]
            }
            for k in cats
        }
        api_names = sorted({k for u in updates for k in u["cuda_runtime_api"]})
        record["cuda_runtime_api"] = {
            k: {
                field: statistics.mean(u["cuda_runtime_api"].get(k, {}).get(field, 0) for u in updates)
                for field in ["count", "sum_cpu_seconds"]
            }
            for k in api_names
        }
        # Exact full kernel names, all ranks, restricted to complete update windows.
        kernels = collections.defaultdict(lambda: [0, 0.0])
        diagnostics = []
        for profile in profiles:
            db = sqlite3.connect(str(Path(profile["report"]).with_suffix(".sqlite")))
            for u in profile["updates"]:
                for name, count, duration in db.execute(
                    """
                    SELECT s.value,count(*),sum(min(k.end,?)-max(k.start,?))/1e9
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
                    WHERE k.start<? AND k.end>? GROUP BY s.value
                    """,
                    (u["end_ns"], u["start_ns"], u["end_ns"], u["start_ns"]),
                ):
                    kernels[name][0] += count / len(updates)
                    kernels[name][1] += duration / len(updates)
            messages = db.execute("SELECT severity,text FROM DIAGNOSTIC_EVENT WHERE severity>=2").fetchall()
            diagnostics.append({"report": profile["report"], "warnings_and_errors": messages})
        record["all_kernel_names"] = {
            k: {"count_per_rank_update": v[0], "service_seconds_per_rank_update": v[1]}
            for k, v in sorted(kernels.items(), key=lambda kv: kv[1][1], reverse=True)
        }
        record["diagnostics"] = diagnostics
        results[backend] = record
    if "pytorch" in results and "megatron" in results:
        a, b = results["pytorch"], results["megatron"]
        results["comparison"] = {
            "profiled_wall_reduction_seconds": a["wall_seconds"] - b["wall_seconds"],
            "profiled_throughput_difference_fraction": a["wall_seconds"] / b["wall_seconds"] - 1,
            "partition_reduction_seconds": {
                k: a[k] - b[k]
                for k in ["gpu_idle_seconds", "nccl_only_seconds", "non_nccl_kernel_and_memops_union_seconds"]
            },
            "interpretation": "Observed occupancy partition, not an isolated causal attribution. "
            "Communication-only includes collective wait; CPU phase sums are not GPU phase wall time.",
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(json.loads(args.analysis.read_text())), indent=2) + "\n")


if __name__ == "__main__":
    main()
