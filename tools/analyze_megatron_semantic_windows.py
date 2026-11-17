"""Partition entire captured updates by matching GPU execution milestones.

CPU backward returns before GPU completion. Boundaries use CUDA launch
correlations and GPU execution, avoiding that queueing artefact. Every nanosecond
of the original whole-update NVTX range belongs to exactly one window.
"""

import argparse
import bisect
import json
import re
import sqlite3
import statistics
from pathlib import Path

from analyze_megatron_fsdp_nsys import seconds


def analyze(database):
    db = sqlite3.connect(database)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = [
        (a, b, text or strings.get(tid, ""))
        for a, b, text, tid in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end>start")
    ]
    runtime = {
        c: (a, b, t)
        for a, b, c, t in db.execute("SELECT start,end,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    }
    events = [
        (a, b, strings[n], c)
        for a, b, n, c in db.execute("SELECT start,end,demangledName,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    ]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        if table in tables:
            events.extend((a, b, table, c) for a, b, c in db.execute(f"SELECT start,end,correlationId FROM {table}"))
    launched = sorted([(runtime[c][0], a, b, n, c) for a, b, n, c in events if c in runtime])
    launch_times = [r[0] for r in launched]

    def from_cpu(a, b):
        return launched[bisect.bisect_left(launch_times, a) : bisect.bisect_left(launch_times, b)]

    def compute_from_cpu(a, b):
        return [r for r in from_cpu(a, b) if "nccl" not in r[3].lower()]

    results = []
    for start, end, label in sorted(ranges):
        if not re.fullmatch(r"attribution.update.\d+", label):
            continue
        forwards = sorted((a, b) for a, b, n in ranges if n == "forward" and start <= a < b <= end)
        backwards = sorted((a, b) for a, b, n in ranges if n == "backward" and start <= a < b <= end)
        assert len(forwards) == len(backwards) == 3
        cursor = start
        windows = []
        preparation = []
        for micro, ((fa, fb), (ba, bb)) in enumerate(zip(forwards, backwards, strict=True), 1):
            f = compute_from_cpu(fa, fb)
            b = compute_from_cpu(ba, bb)
            first_fwd = min(x[1] for x in f)
            last_fwd = max(x[2] for x in f)
            last_bwd = max(x[2] for x in b)
            l0 = next(
                (a, b) for a, b, n in ranges if n == "cosmos.forward.network.decoder.layer_00" and fa <= a < b <= fb
            )
            l35 = next(
                (a, b) for a, b, n in ranges if n == "cosmos.forward.network.decoder.layer_35" and fa <= a < b <= fb
            )
            qkv = [x for x in compute_from_cpu(*l0) if "nvjet" in x[3]]
            first_decoder = min(x[1] for x in qkv)
            last_decoder = max(x[2] for x in compute_from_cpu(*l35))
            vae_cpu = next(
                (a, b) for a, b, n in ranges if n == "cosmos.forward.prepare_inputs.vae" and fa <= a < b <= fb
            )
            vae_end = max(x[2] for x in compute_from_cpu(*vae_cpu))
            assert first_fwd <= vae_end <= first_decoder
            preparation.extend(
                [
                    {
                        "microbatch": micro,
                        "phase": "through_vae_gpu_completion",
                        "wall_seconds": (vae_end - first_fwd) / 1e9,
                    },
                    {
                        "microbatch": micro,
                        "phase": "after_vae_to_first_decoder_gemm",
                        "wall_seconds": (first_decoder - vae_end) / 1e9,
                    },
                ]
            )
            assert cursor <= first_fwd <= first_decoder <= last_decoder <= last_fwd <= last_bwd <= end
            windows.extend(
                [
                    (cursor, first_fwd, "input_and_microbatch_transition", micro),
                    (first_fwd, first_decoder, "forward_prepare_and_first_block_entry", micro),
                    (first_decoder, last_decoder, "forward_decoder", micro),
                    (last_decoder, last_fwd, "forward_heads_and_loss", micro),
                    (last_fwd, last_bwd, "backward_and_recompute", micro),
                ]
            )
            cursor = last_bwd
        windows.append((cursor, end, "post_backward_to_update_end", 3))
        assert sum(b - a for a, b, _, _ in windows) == end - start
        rows = []
        for a, b, n, micro in windows:
            comm = [(max(x, a), min(y, b)) for x, y, name, _ in events if x < b and y > a and "nccl" in name.lower()]
            work = [
                (max(x, a), min(y, b)) for x, y, name, _ in events if x < b and y > a and "nccl" not in name.lower()
            ]
            busy = seconds(comm + work)
            other = seconds(work)
            rows.append(
                {
                    "phase": n,
                    "microbatch": micro,
                    "start_ns": a,
                    "end_ns": b,
                    "wall_seconds": (b - a) / 1e9,
                    "non_nccl_active_seconds": other,
                    "nccl_only_seconds": busy - other,
                    "gpu_idle_seconds": (b - a) / 1e9 - busy,
                }
            )
        results.append(
            {
                "update": label,
                "wall_seconds": (end - start) / 1e9,
                "windows": rows,
                "preparation_breakdown": preparation,
            }
        )
    return {"database": str(database.resolve()), "updates": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = [analyze(p) for p in sorted(args.root.glob("*/rank[0-3].sqlite"))]
    summary = {}
    for backend in ["pytorch", "megatron"]:
        updates = [u for p in profiles if Path(p["database"]).parent.name == backend for u in p["updates"]]
        if not updates:
            continue
        summary[backend] = {}
        for phase in sorted({r["phase"] for u in updates for r in u["windows"]}):
            summary[backend][phase] = {
                key: statistics.mean(sum(r[key] for r in u["windows"] if r["phase"] == phase) for u in updates)
                for key in ["wall_seconds", "non_nccl_active_seconds", "nccl_only_seconds", "gpu_idle_seconds"]
            }
        summary[backend]["preparation_breakdown"] = {
            phase: statistics.mean(
                sum(r["wall_seconds"] for r in u["preparation_breakdown"] if r["phase"] == phase) for u in updates
            )
            for phase in ["through_vae_gpu_completion", "after_vae_to_first_decoder_gemm"]
        }
        summary[backend]["total_seconds"] = statistics.mean(u["wall_seconds"] for u in updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "profiles": profiles,
                "summary": summary,
                "note": "Disjoint observed wall-time accounting, not independent causal effects. First decoder boundary is the first nvjet GEMM launched within layer_00; includes initial normalization in the preceding window.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
