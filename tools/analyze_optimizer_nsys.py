"""Measure complete post-autograd updates, including asynchronous GPU completion.

No extra device synchronization is inserted into training. GPU completion is
recovered from CUPTI launch correlations; work queued before the range is
reported separately rather than silently charged to Adam.
"""

import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

from analyze_megatron_fsdp_nsys import seconds


def analyze(report):
    db = sqlite3.connect(report.with_suffix(".sqlite"))
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = [
        (a, b, text or names.get(tid, ""), thread)
        for a, b, text, tid, thread in db.execute(
            "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end>start"
        )
    ]
    runtime = {
        corr: (a, b, thread)
        for a, b, corr, thread in db.execute(
            "SELECT start,end,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }
    events = [
        (a, b, names[n], corr)
        for a, b, n, corr in db.execute("SELECT start,end,demangledName,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    ]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        if table in tables:
            events.extend(
                (a, b, table, corr) for a, b, corr in db.execute(f"SELECT start,end,correlationId FROM {table}")
            )
    result = []
    for a, b, label, thread in sorted(ranges):
        if label != "cosmos.optimizer.full_post_autograd":
            continue
        update = next(n for s, e, n, _ in ranges if s <= a < e and n.startswith("attribution.update."))
        launched = [x for x in events if x[3] in runtime and a <= runtime[x[3]][0] < b]
        prequeued = [x for x in events if x[3] in runtime and runtime[x[3]][0] < a and x[1] > a]
        completion = max([b, *(x[1] for x in launched), *(x[1] for x in prequeued)])
        active = [(max(s, a), min(e, completion)) for s, e, _, _ in events if s < completion and e > a]
        pending = [
            (max(s, a), min(e, completion))
            for s, e, _, c in events
            if s < completion and e > a and c in runtime and runtime[c][0] < a
        ]
        phases = []
        for s, e, name, tid in sorted(ranges):
            if (
                a <= s
                and e <= b
                and (
                    name.startswith("cosmos.optimizer.")
                    or name.startswith("cosmos.callbacks.")
                    or name == "cosmos.backward.model_after_backward"
                    or name == "optimizer_step"
                )
            ):
                own = [x for x in launched if s <= runtime[x[3]][0] < e and runtime[x[3]][2] == tid]
                phases.append(
                    {
                        "name": name,
                        "start_ns": s,
                        "end_ns": e,
                        "cpu_seconds": (e - s) / 1e9,
                        "launched_gpu_busy_seconds": seconds([(x[0], x[1]) for x in own]),
                        "launched_kernel_or_memop_count": len(own),
                    }
                )
        result.append(
            {
                "update": update,
                "start_ns": a,
                "cpu_end_ns": b,
                "gpu_complete_ns": completion,
                "cpu_seconds": (b - a) / 1e9,
                "through_gpu_completion_seconds": (completion - a) / 1e9,
                "gpu_busy_seconds": seconds(active),
                "queued_before_range_gpu_busy_seconds": seconds(pending),
                "phases": phases,
            }
        )
    if len(result) != 2:
        raise ValueError(f"Expected 2 complete extended ranges: {report}, got {len(result)}")
    return {
        "report": str(report.resolve()),
        "sha256": hashlib.file_digest(report.open("rb"), "sha256").hexdigest(),
        "updates": result,
        "unmatched_gpu_event_count": sum(c not in runtime for _, _, _, c in events),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = [analyze(p) for p in sorted(args.root.glob("*/rank*.nsys-rep"))]
    summary = {}
    for backend in ["pytorch", "megatron"]:
        rows = [u for p in profiles if Path(p["report"]).parent.name == backend for u in p["updates"]]
        if not rows:
            continue
        summary[backend] = {
            k: statistics.mean(u[k] for u in rows)
            for k in [
                "cpu_seconds",
                "through_gpu_completion_seconds",
                "gpu_busy_seconds",
                "queued_before_range_gpu_busy_seconds",
            ]
        }
        labels = sorted({p["name"] for u in rows for p in u["phases"]})
        summary[backend]["cpu_phase_seconds"] = {
            name: statistics.mean(sum(p["cpu_seconds"] for p in u["phases"] if p["name"] == name) for u in rows)
            for name in labels
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "profiles": profiles,
                "summary": summary,
                "note": "Phases nest: do not sum parent and child. GPU busy is a union, not critical-path attribution. "
                "Completion includes device work launched inside the CPU range; prequeued work is counted separately. "
                "Whole-update measurement already synchronizes; no extra synchronization was inserted.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
