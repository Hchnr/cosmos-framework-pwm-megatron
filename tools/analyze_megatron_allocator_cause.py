"""Summarize CUDA virtual-memory work and opt-in allocator observations.

API service times are diagnostic CPU times, not additive wall-time savings.
"""

import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path

from analyze_megatron_semantic_windows import analyze


def inspect(database):
    db = sqlite3.connect(database)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    memory = [
        (a, b, strings[n], result)
        for a, b, n, result in db.execute("SELECT start,end,nameId,returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        if strings[n].startswith(("cuMem", "cudaMalloc", "cudaFree")) and not strings[n].startswith("cudaMemcpy")
    ]
    semantic = analyze(database)
    rows = []
    for update in semantic["updates"]:
        for window in update["windows"]:
            a, b = window["start_ns"], window["end_ns"]
            totals = collections.defaultdict(lambda: dict(calls=0, seconds=0.0, errors=0))
            for x, y, name, result in memory:
                if x < b and y > a:
                    totals[name]["calls"] += int(a <= x < b)
                    totals[name]["seconds"] += (min(y, b) - max(x, a)) / 1e9
                    totals[name]["errors"] += int(result != 0)
            rows.append(dict(update=update["update"], **window, memory_apis=dict(totals)))
    nvtx = [
        (a, b, text or strings.get(tid, ""))
        for a, b, text, tid in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end>start")
    ]
    allocations = []
    for a, b, name in nvtx:
        if name.startswith("attribution.allocate.") and b - a >= 20e6:
            totals = collections.defaultdict(lambda: dict(calls=0, seconds=0.0))
            for x, y, api, _ in memory:
                if a <= x < y <= b:
                    totals[api]["calls"] += 1
                    totals[api]["seconds"] += (y - x) / 1e9
            allocations.append(
                dict(start_ns=a, end_ns=b, label=name, duration_seconds=(b - a) / 1e9, memory_apis=dict(totals))
            )
    db.close()
    return dict(database=str(database.resolve()), windows=rows, slow_allocations=allocations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = [inspect(p) for p in sorted(args.root.glob("*/rank[0-3].sqlite"))]
    summary = {}
    for backend in ("pytorch", "megatron"):
        windows = [w for p in profiles if Path(p["database"]).parent.name == backend for w in p["windows"]]
        # Each update name appears once per rank; average those observations.
        observations = sum(
            len({w["update"] for w in p["windows"]}) for p in profiles if Path(p["database"]).parent.name == backend
        )
        if not observations:
            continue
        totals = collections.defaultdict(lambda: dict(calls_per_update=0.0, cpu_seconds_per_update=0.0))
        for w in windows:
            for api, row in w["memory_apis"].items():
                totals[api]["calls_per_update"] += row["calls"] / observations
                totals[api]["cpu_seconds_per_update"] += row["seconds"] / observations
        allocation_records = []
        step_records = []
        for path in sorted((args.root / backend).glob("allocator_rank_*.json")):
            data = json.loads(path.read_text())
            allocation_records.extend(data["allocations"])
            steps = data["steps"]
            begins = [r for r in steps if r["event"] == "before_dataload"]
            ends = [r for r in steps if r["event"] == "update_end"]
            for end in ends:
                begin = next(r for r in begins if r["iteration"] == end["iteration"] - 1)
                keys = ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free")
                step_records.append(
                    dict(
                        rank=path.stem,
                        iteration=end["iteration"],
                        delta={k: end["counters"][k] - begin["counters"][k] for k in keys},
                    )
                )
        summary[backend] = dict(memory_apis=dict(totals), allocator_update_counters=step_records)
        if allocation_records:
            summary[backend]["allocation_count"] = len(allocation_records)
            summary[backend]["slow_allocations"] = sorted(
                [r for r in allocation_records if r["duration_seconds"] >= 0.02],
                key=lambda r: r["duration_seconds"],
                reverse=True,
            )
        if step_records:
            summary[backend]["mean_allocator_update_counters"] = {
                key: statistics.mean(r["delta"][key] for r in step_records) for key in step_records[0]["delta"]
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(profiles=profiles, summary=summary), indent=2) + "\n")
    print(
        json.dumps(
            {k: {key: value for key, value in row.items() if key != "slow_allocations"} for k, row in summary.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
