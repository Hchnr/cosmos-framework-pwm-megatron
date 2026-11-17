"""Export preserved Nsight reports and compute overlap-aware attribution tables.

Kernel durations are GPU service time, not additive wall-time savings. Interval
unions partition whole-update wall time without double counting stream overlap.
"""

import argparse
import collections
import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def seconds(intervals):
    return sum(b - a for a, b in union(intervals)) / 1e9


def category(name):
    n = name.lower()
    if "nccl" in n:
        for key in ["allgather", "reducescatter", "allreduce", "broadcast"]:
            if key in n.replace("_", ""):
                return "nccl_" + key
        return "nccl_other"
    if any(x in n for x in ["flash", "fmha", "flex_attention", "flexattn"]):
        return "attention"
    if any(x in n for x in ["gemm", "xmma", "matmul", "cutlass", "cublas", "nvjet"]):
        return "gemm"
    if any(x in n for x in ["adam", "multi_tensor_apply"]):
        return "optimizer_or_multi_tensor"
    if any(x in n for x in ["direct_copy", "vectorized_copy", "memcpy", "_to_copy", "cast_kernel"]):
        return "copy_cast"
    return "other_compute"


def analyze(report):
    database = report.with_suffix(".sqlite")
    with report.with_suffix(".export.log").open("a") as log:
        if not database.exists():
            subprocess.run(
                ["nsys", "export", "--type=sqlite", "--output=" + str(database), str(report)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    db = sqlite3.connect(str(database))
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = [
        (s, e, text or names.get(tid, ""), thread)
        for s, e, text, tid, thread in db.execute(
            "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end > start"
        )
    ]
    updates = sorted((a, b, n) for a, b, n, _ in ranges if re.fullmatch(r"attribution.update.\d+", n))
    if len(updates) != 2:
        raise ValueError(f"{report}: expected two whole updates, found {updates}")
    kernels = [
        (a, b, names.get(n, str(n)), stream, correlation)
        for a, b, n, stream, correlation in db.execute(
            "SELECT start,end,demangledName,streamId,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    runtime = {
        corr: (a, b, names.get(n, str(n)), thread)
        for a, b, n, corr, thread in db.execute(
            "SELECT start,end,nameId,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    coarse = [(a, b, n, t) for a, b, n, t in ranges if n in ["forward", "backward", "optimizer_step"]]
    tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    memops = []
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        if table in tables:
            memops.extend(db.execute(f"SELECT start,end FROM {table}"))
    result = {
        "report": str(report.resolve()),
        "bytes": report.stat().st_size,
        "sha256": hashlib.file_digest(report.open("rb"), "sha256").hexdigest(),
        "kernel_count": len(kernels),
        "updates": [],
    }
    for start, end, label in updates:
        rows = [
            (max(a, start), min(b, end), n, stream, corr) for a, b, n, stream, corr in kernels if b > start and a < end
        ]
        comm = [(a, b) for a, b, n, _, _ in rows if category(n).startswith("nccl")]
        compute = [(a, b) for a, b, n, _, _ in rows if not category(n).startswith("nccl")]
        memory = [(max(a, start), min(b, end)) for a, b in memops if b > start and a < end]
        other = compute + memory
        gpu_busy = seconds(comm + other)
        comm_s, other_s = seconds(comm), seconds(other)
        categories = collections.defaultdict(lambda: {"count": 0, "sum_gpu_seconds": 0.0})
        top = collections.defaultdict(lambda: [0, 0.0])
        phase_gpu = collections.defaultdict(float)
        for a, b, n, _, corr in rows:
            c = category(n)
            categories[c]["count"] += 1
            categories[c]["sum_gpu_seconds"] += (b - a) / 1e9
            top[n][0] += 1
            top[n][1] += (b - a) / 1e9
            api = runtime.get(corr)
            if api:
                # Autograd launches on worker threads; use the main-thread
                # coarse time window rather than demanding identical thread ID.
                phase = next((n for s, e, n, t in coarse if s <= api[0] < e), "other")
                phase_gpu[phase] += (b - a) / 1e9
        cpu_phases = {
            n: sum(min(b, end) - max(a, start) for a, b, name, _ in coarse if name == n and b > start and a < end) / 1e9
            for n in ["forward", "backward", "optimizer_step"]
        }
        api_stats = collections.defaultdict(lambda: {"count": 0, "sum_cpu_seconds": 0.0})
        for a, b, n, _ in runtime.values():
            if start <= a < end:
                api_stats[n]["count"] += 1
                api_stats[n]["sum_cpu_seconds"] += (b - a) / 1e9
        result["updates"].append(
            {
                "label": label,
                "start_ns": start,
                "end_ns": end,
                "wall_seconds": (end - start) / 1e9,
                "gpu_busy_union_seconds": gpu_busy,
                "nccl_union_seconds": comm_s,
                "non_nccl_kernel_and_memops_union_seconds": other_s,
                "overlap_seconds": comm_s + other_s - gpu_busy,
                "nccl_only_seconds": gpu_busy - other_s,
                "non_nccl_only_seconds": gpu_busy - comm_s,
                "gpu_idle_seconds": (end - start) / 1e9 - gpu_busy,
                "categories": dict(categories),
                "cpu_phase_wall_seconds": cpu_phases,
                "cuda_runtime_api": dict(api_stats),
                "kernel_service_seconds_by_launch_cpu_phase": dict(phase_gpu),
                "top_kernels_by_service_time": sorted(
                    [{"name": n, "count": v[0], "sum_gpu_seconds": v[1]} for n, v in top.items()],
                    key=lambda r: r["sum_gpu_seconds"],
                    reverse=True,
                )[:40],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = sorted(args.root.glob("*/rank*.nsys-rep"))
    if not reports:
        raise ValueError("No reports found")
    results = []
    for report in reports:
        results.append(analyze(report))
        print(f"Analyzed {report}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "profiles": results,
                "note": "NCCL-only is observed non-overlapped communication occupancy, not proof of critical-path cost; "
                "kernel service sums are not wall time. CPU phase times include asynchronous launch/wait behavior.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
