"""Attribute a multi-process TP/PP Nsight capture without mixing correlation IDs."""

import argparse
import json
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


def union_ns(intervals):
    total, end = 0, None
    for left, right in sorted(intervals):
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def analyze(path, *, phase_prefixes=()):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    # Nsight's own projection report joins on BOTH process and correlation ID.
    # A torchrun-wide SQLite contains repeated correlation IDs across ranks.
    runtime = {
        (tid & 0xFFFFFFFFFF000000, correlation): (start, end, tid)
        for start, end, tid, correlation in db.execute(
            "SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    kernels = [
        {"start": a, "end": b, "pid": pid, "device": device, "name": strings[name], "correlation": correlation}
        for a, b, pid, device, name, correlation in db.execute(
            "SELECT start,end,globalPid,deviceId,demangledName,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    by_process = defaultdict(list)
    by_thread = defaultdict(list)
    for event in kernels:
        by_process[event["pid"]].append(event)
        launch = runtime.get((event["pid"], event["correlation"]))
        if launch:
            by_thread[launch[2]].append((launch[0], event))
    launches = {}
    for tid, events in by_thread.items():
        ordered = sorted(events, key=lambda item: item[0])
        launches[tid] = ([item[0] for item in ordered], [item[1] for item in ordered])
    memory = defaultdict(list)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            for start, end, pid, size in db.execute(f"SELECT start,end,globalPid,bytes FROM {table}"):
                memory[pid].append((start, end, size, table))
    processes = []
    for pid, events in sorted(by_process.items()):
        begin, end = min(e["start"] for e in events), max(e["end"] for e in events)
        active = union_ns((e["start"], e["end"]) for e in events)
        non_nccl = union_ns((e["start"], e["end"]) for e in events if "nccl" not in e["name"].lower())
        memops = [(max(a, begin), min(b, end)) for a, b, _, _ in memory[pid] if a < end and b > begin]
        cuda_active = union_ns([(e["start"], e["end"]) for e in events] + memops)
        processes.append(
            {
                "pid": (pid >> 24) & 0x00FFFFFF,
                "device": sorted({e["device"] for e in events}),
                "kernel_window_seconds": (end - begin) / 1e9,
                "non_nccl_kernel_active_seconds": non_nccl / 1e9,
                "nccl_kernel_only_seconds": (active - non_nccl) / 1e9,
                "no_kernel_seconds": (end - begin - active) / 1e9,
                "memory_ops_without_kernel_seconds": (cuda_active - active) / 1e9,
                "no_cuda_activity_seconds": (end - begin - cuda_active) / 1e9,
                "memcpy_bytes_for_overlapping_ops": sum(
                    size
                    for a, b, size, kind in memory[pid]
                    if kind == "CUPTI_ACTIVITY_KIND_MEMCPY" and a < end and b > begin
                ),
            }
        )
    selected = {
        "cosmos.tp_pp.native_prepare",
        "cosmos.tp_pp.replay_hash",
        "cosmos.tp_pp.replay_load",
        "cosmos.forward.prepare_inputs.vae",
        "cosmos.forward.add_noise",
    }
    phases = defaultdict(lambda: {"cpu": [], "gpu": [], "non_nccl": [], "instances": 0})
    for a, b, text, text_id, tid in db.execute(
        "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end>start"
    ):
        name = text or strings.get(text_id, "")
        if name not in selected and not any(name.startswith(prefix) for prefix in phase_prefixes):
            continue
        pid = tid & 0xFFFFFFFFFF000000
        phase = phases[(pid, name)]
        phase["cpu"].append((a, b))
        phase["instances"] += 1
        starts, events = launches.get(tid, ([], []))
        for event in events[bisect_left(starts, a) : bisect_left(starts, b)]:
            phase["gpu"].append((event["start"], event["end"]))
            if "nccl" not in event["name"].lower():
                phase["non_nccl"].append((event["start"], event["end"]))
    db.close()
    return {
        "scope": "partial capture; process-local kernel windows, not entire optimizer updates",
        "database": str(path),
        "correlation_join_includes_process": True,
        "nccl_kernel_time_includes_waiting_and_is_not_wire_transfer_time": True,
        "no_kernel_time_may_include_CUDA_memops_not_counted_here": True,
        "cuda_activity_includes_kernels_memcpy_memset": True,
        "phase_prefixes": list(phase_prefixes),
        "nested_phases_overlap_and_must_not_be_summed": True,
        "processes": processes,
        "phases": [
            {
                "pid": (pid >> 24) & 0x00FFFFFF,
                "phase": name,
                "instances": values["instances"],
                "cpu_range_union_seconds": union_ns(values["cpu"]) / 1e9,
                "launched_kernel_union_seconds": union_ns(values["gpu"]) / 1e9,
                "launched_non_nccl_kernel_union_seconds": union_ns(values["non_nccl"]) / 1e9,
                "launched_nccl_only_kernel_seconds": (union_ns(values["gpu"]) - union_ns(values["non_nccl"])) / 1e9,
            }
            for (pid, name), values in sorted(phases.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-prefix", action="append", default=[])
    args = parser.parse_args()
    report = analyze(args.database, phase_prefixes=args.phase_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
