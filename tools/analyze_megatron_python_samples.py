"""Decode Nsight Python periodic samples; retain stacks behind allocator stalls.

Nsight exports Python samples as NVTX binary payloads in JSON Lines. Counts are
sampling evidence, not additive critical-path durations. Background threads and
loader workers are excluded using CUDA-launch thread IDs of the training rank.
"""

import argparse
import base64
import bisect
import collections
import json
import os
import sqlite3
import struct
import subprocess
from functools import cache
from pathlib import Path

from analyze_megatron_fsdp_nsys import union


@cache
def compiled_instruction(filename, lineno):
    source = Path(filename)
    lines = source.read_text().splitlines() if source.exists() else []
    return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""


def analyze(report, cache):
    database = report.with_suffix(".sqlite")
    db = sqlite3.connect(database)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = [
        (a, b, text or strings.get(i, ""), tid)
        for a, b, text, i, tid in db.execute("SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end>start")
    ]
    updates = sorted((a, b, n) for a, b, n, _ in ranges if n.startswith("attribution.update."))
    assert len(updates) == 2, f"Expected two complete updates: {report}"
    main_tid = next(t for _, _, n, t in ranges if n == "forward")
    backward = [(a, b) for a, b, n, _ in ranges if n == "backward"]
    tids = {main_tid}
    tids.update(
        tid
        for a, tid in db.execute("SELECT start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        if any(x <= a < y for x, y in backward)
    )
    events = list(db.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        events.extend(db.execute(f"SELECT start,end FROM {table}"))
    busy = union(events)
    starts = [a for a, _ in busy]
    cache.mkdir(parents=True, exist_ok=True)
    exported = cache / f"{report.parent.name}_{report.stem}.legacy.jsonl"
    if not exported.exists():
        with exported.with_suffix(".log").open("w") as log:
            subprocess.run(
                ["nsys", "export", "--type=json", "--output=" + str(exported), str(report)],
                env={**os.environ, "NSYS_ENABLE_DEPRECATED_JSON_EXPORT": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    counts = collections.Counter()
    idle_counts = collections.Counter()
    allocation_points = collections.defaultdict(list)
    stacks = collections.Counter()
    with exported.open() as source:
        names = json.loads(next(source))["data"]
        for line in source:
            event = json.loads(line).get("NvtxEvent", {})
            if event.get("Type") != 34 or "Payload" not in event:
                continue
            tid = int(event["GlobalTid"])
            timestamp = int(event["Timestamp"])
            if tid not in tids or not any(a <= timestamp < b for a, b, _ in updates):
                continue
            payload = event["Payload"]["BinaryData"][0]
            if int(payload["SchemaId"]) != 4294967298:
                continue
            blob = base64.b64decode(payload["Data"])
            _, size = struct.unpack_from("<II", blob, 8)
            assert len(blob) >= 16 + 24 * size
            frames = []
            for index in range(size):
                function, filename, lineno = struct.unpack_from("<QQi", blob, 16 + 24 * index)
                frames.append((names[function], names[filename], lineno))
            if not frames:
                continue
            leaf = frames[0]
            key = ":".join(map(str, leaf))
            counts[key] += 1
            previous = bisect.bisect_right(starts, timestamp) - 1
            idle = previous < 0 or timestamp >= busy[previous][1]
            if idle:
                idle_counts[key] += 1
            if leaf[0] == "allocate" and leaf[1].endswith("_fsdp_collectives.py"):
                group = "fsdp_allgather_output_allocate"
            elif leaf[0] == "alloc_storage" and leaf[1].endswith("_fsdp_param.py"):
                group = "fsdp_parameter_storage_allocate"
            elif leaf[0] == "call" and "/torchinductor_" in leaf[1]:
                instruction = compiled_instruction(leaf[1], leaf[2])
                group = "compiled_empty_strided_cuda" if "empty_strided_cuda(" in instruction else "compiled_other"
            else:
                continue
            if group != "compiled_other":
                allocation_points[(tid, group)].append(timestamp)
                stacks[tuple(frames[:7])] += 1
    spans = []
    for (tid, group), points in allocation_points.items():
        run = []
        for timestamp in sorted(points):
            if run and timestamp - run[-1] > 8_000_000:
                if run[-1] - run[0] >= 20_000_000:
                    spans.append(
                        {
                            "tid": tid,
                            "group": group,
                            "first_sample_ns": run[0],
                            "last_sample_ns": run[-1],
                            "sample_count": len(run),
                        }
                    )
                run = []
            run.append(timestamp)
        if run and run[-1] - run[0] >= 20_000_000:
            spans.append(
                {
                    "tid": tid,
                    "group": group,
                    "first_sample_ns": run[0],
                    "last_sample_ns": run[-1],
                    "sample_count": len(run),
                }
            )
    return {
        "report": str(report.resolve()),
        "main_tid": main_tid,
        "training_cuda_threads": sorted(tids),
        "top_leaf_samples": counts.most_common(30),
        "top_leaf_samples_during_gpu_idle": idle_counts.most_common(30),
        "allocation_spans": sorted(spans, key=lambda r: r["first_sample_ns"]),
        "allocator_stacks": [{"samples": count, "frames": frames} for frames, count in stacks.most_common(20)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = []
    for report in sorted(args.root.glob("*/rank[0-3].nsys-rep")):
        print(str(report), flush=True)
        result.append(analyze(report, args.cache))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "profiles": result,
                "note": "250 Hz periodic Python samples. Span endpoints are sampled observations, not exact allocation entry/exit timestamps. Do not sum CPU durations across threads or ranks as wall-time savings.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
