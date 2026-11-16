#!/usr/bin/env python3
"""Classify Nsight Systems kernels on each GPU's main compute stream."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path


def category(name: str) -> str:
    lower = name.lower()
    if "flash::" in lower or "flashattn" in lower or "flash_attn" in lower or "flex_attention" in lower:
        return "attention"
    if "nvjet" in lower or "gemm" in lower or "cublas" in lower or "matmul" in lower or "cutlass::kernel2" in lower:
        return "gemm"
    if "nccl" in lower:
        return "nccl"
    return "other"


def analyze(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT k.deviceId, k.streamId, s.value, COUNT(*), SUM(k.end-k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
        "GROUP BY k.deviceId, k.streamId, s.value"
    ).fetchall()
    streams: dict[int, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    kernels: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for device, stream, name, count, duration_ns in rows:
        cat = category(name)
        streams[device][stream][cat] += duration_ns
        kernels[(device, stream)].append({"name": name, "category": cat, "count": count, "seconds": duration_ns / 1e9})
    result = {"sqlite": str(path), "gpus": []}
    for device in sorted(streams):
        primary = max(
            streams[device],
            key=lambda stream: sum(t for cat, t in streams[device][stream].items() if cat != "nccl"),
        )
        cat_ns = dict(streams[device][primary])
        total = sum(cat_ns.values())
        if not total:
            raise ValueError(f"GPU {device} has no primary stream kernels")
        result["gpus"].append(
            {
                "device": device,
                "primary_stream": primary,
                "kernel_count": sum(item["count"] for item in kernels[(device, primary)]),
                "primary_stream_seconds": total / 1e9,
                "categories": {
                    cat: {"seconds": duration / 1e9, "pct": duration / total * 100}
                    for cat, duration in sorted(cat_ns.items())
                },
                "other_streams": {
                    str(stream): {cat: duration / 1e9 for cat, duration in cats.items()}
                    for stream, cats in streams[device].items()
                    if stream != primary
                },
                "top_kernels": sorted(kernels[(device, primary)], key=lambda item: item["seconds"], reverse=True)[:30],
            }
        )
    if not result["gpus"]:
        raise ValueError(f"No CUDA kernel records in {path}")
    con.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help=".nsys-rep file or exported .sqlite")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    if report.suffix == ".nsys-rep":
        sqlite = report.with_suffix(".sqlite")
        if not sqlite.exists():
            subprocess.run(["nsys", "export", "--type=sqlite", f"--output={sqlite}", str(report)], check=True)
    elif report.suffix == ".sqlite":
        sqlite = report
    else:
        parser.error("report must end in .nsys-rep or .sqlite")
    result = analyze(sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for gpu in result["gpus"]:
        cats = gpu["categories"]
        print(
            f"GPU {gpu['device']} stream {gpu['primary_stream']}: "
            f"attn={cats.get('attention', {}).get('pct', 0):.1f}% "
            f"gemm={cats.get('gemm', {}).get('pct', 0):.1f}%"
        )


if __name__ == "__main__":
    main()
