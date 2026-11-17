"""Relate sampled allocator residency to matching GPU execution windows."""

import argparse
import json
import pathlib
import sqlite3
import statistics

from analyze_megatron_fsdp_nsys import seconds

parser = argparse.ArgumentParser()
parser.add_argument("root", type=pathlib.Path, help="Artifact directory containing Python samples and semantic windows")
args = parser.parse_args()
p = args.root
sample = json.loads((p / "python_samples.json").read_text())["profiles"]
sem = json.loads((p / "semantic_windows.json").read_text())["profiles"]
rows = []
for update in ["attribution.update.001204", "attribution.update.001205"]:
    for micro in [1, 2, 3]:
        wall = {}
        spans = {}
        for backend in ["pytorch", "megatron"]:
            durations = []
            points = []
            for r in sem:
                db = pathlib.Path(r["database"])
                if db.parent.name != backend:
                    continue
                u = next(u for u in r["updates"] if u["update"] == update)
                w = next(
                    w
                    for w in u["windows"]
                    if w["phase"] == "forward_prepare_and_first_block_entry" and w["microbatch"] == micro
                )
                v = next(
                    x
                    for x in u["preparation_breakdown"]
                    if x["microbatch"] == micro and x["phase"] == "through_vae_gpu_completion"
                )["wall_seconds"]
                start = w["start_ns"] + round(v * 1e9)
                end = w["end_ns"]
                durations.append((end - start) / 1e9)
                sr = next(s for s in sample if pathlib.Path(s["report"]).with_suffix(".sqlite") == db)
                for s in sr["allocation_spans"]:
                    if s["group"] != "fsdp_allgather_output_allocate":
                        continue
                    a = max(start, s["first_sample_ns"])
                    b = min(end, s["last_sample_ns"])
                    if b > a:
                        points.append(
                            {
                                "rank": int(db.stem[-1]),
                                "seconds": (b - a) / 1e9,
                                "first_sample_ns": a,
                                "last_sample_ns": b,
                            }
                        )
            wall[backend] = statistics.mean(durations)
            spans[backend] = sorted(points, key=lambda x: x["seconds"], reverse=True)
        rows.append(
            {
                "update": update,
                "microbatch": micro,
                "entry_wall_seconds": wall,
                "entry_wall_difference_seconds": wall["pytorch"] - wall["megatron"],
                "longest_sampled_allocation_seconds": spans["pytorch"][0]["seconds"] if spans["pytorch"] else 0,
                "allocation_sample_spans": spans,
            }
        )
result = {
    "entries": rows,
    "mean_entry_wall_difference_per_update": sum(x["entry_wall_difference_seconds"] for x in rows) / 2,
    "sum_longest_sampled_allocation_per_entry_divided_by_two_updates": sum(
        x["longest_sampled_allocation_seconds"] for x in rows
    )
    / 2,
    "note": "Take the longest sampled allocator span across ranks separately for each logical forward entry; do not sum across ranks. Clipped to own VAE GPU completion→first decoder GEMM. Still observational sampled residency, not an isolated net wall-time effect.",
}
(p / "forward_entry_allocation.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    [
        (
            r["update"],
            r["microbatch"],
            round(r["entry_wall_difference_seconds"], 3),
            round(r["longest_sampled_allocation_seconds"], 3),
        )
        for r in rows
    ]
)
print(
    result["mean_entry_wall_difference_per_update"],
    result["sum_longest_sampled_allocation_per_entry_divided_by_two_updates"],
)


backward_rows = []
for r in sem:
    dbpath = pathlib.Path(r["database"])
    sr = next(s for s in sample if pathlib.Path(s["report"]).with_suffix(".sqlite") == dbpath)
    db = sqlite3.connect(dbpath)
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    busy = list(db.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        busy.extend(db.execute(f"SELECT start,end FROM {table}"))
    gc = [
        (a, b)
        for a, b, n, i in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end>start")
        if "attribution.python_gc." in (n or names.get(i, ""))
    ]
    for u in r["updates"]:
        windows = [w for w in u["windows"] if w["phase"] == "backward_and_recompute"]
        alloc = [
            (max(w["start_ns"], x["first_sample_ns"]), min(w["end_ns"], x["last_sample_ns"]))
            for w in windows
            for x in sr["allocation_spans"]
            if x["last_sample_ns"] > w["start_ns"] and x["first_sample_ns"] < w["end_ns"]
        ]
        intersections = [(max(a, x), min(b, y)) for a, b in alloc for x, y in busy + gc if x < b and y > a]
        covered_idle = seconds(alloc) - seconds(intersections)
        idle = sum(w["gpu_idle_seconds"] for w in windows)
        assert -1e-8 <= covered_idle <= idle + 1e-8
        backward_rows.append(
            {
                "backend": dbpath.parent.name,
                "rank": int(dbpath.stem[-1]),
                "update": u["update"],
                "backward_gpu_idle_seconds": idle,
                "idle_overlapping_long_allocator_sample_spans_excluding_gc_seconds": covered_idle,
            }
        )
backward_summary = {
    backend: {
        key: statistics.mean(r[key] for r in backward_rows if r["backend"] == backend)
        for key in ["backward_gpu_idle_seconds", "idle_overlapping_long_allocator_sample_spans_excluding_gc_seconds"]
    }
    for backend in ["pytorch", "megatron"]
}
(p / "backward_allocation_idle_overlap.json").write_text(
    json.dumps(
        {
            "observations": backward_rows,
            "summary": backward_summary,
            "note": "Overlap of exact GPU-idle intervals with approximate CPU allocator sample spans >=20ms. GC NVTX intervals excluded. Sample endpoints have sampling uncertainty; this is not an isolated causal effect.",
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(backward_summary, indent=2))
