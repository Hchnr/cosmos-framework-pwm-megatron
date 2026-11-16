#!/usr/bin/env python3
"""Audit the existing T2V traces by FlashAttention stage and launch grid (CPU only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from tools.seqlen_efficiency_nsys import category


def stage(name: str) -> str:
    if name.startswith("triton_") and "_flash_attn_" in name:
        return "compiled_aux"
    for symbol, label in (
        ("FlashAttnFwdSm90", "forward"),
        ("FlashAttnBwdSm90", "backward_main"),
        ("FlashAttnBwdPreprocess", "backward_aux"),
        ("FlashAttnBwdPostprocessConvertdQ", "backward_aux"),
        ("prepare_varlen_num_blocks_kernel", "metadata"),
    ):
        if symbol in name:
            return label
    raise ValueError(f"Unrecognized attention kernel: {name}")


def analyze(run: Path, n: int, microbatches: int) -> tuple[dict, list[dict]]:
    baseline = json.loads((run / "nsys/kernels.json").read_text())
    database = run / "nsys" / f"t2v_n{n:03d}.sqlite"
    con = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    details = []
    per_gpu = []
    try:
        for gpu in baseline["gpus"]:
            rows = con.execute(
                "SELECT s.value, k.gridX, k.gridY, k.gridZ, COUNT(*), SUM(k.end-k.start) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                "WHERE k.deviceId=? AND k.streamId=? GROUP BY s.value,k.gridX,k.gridY,k.gridZ",
                (gpu["device"], gpu["primary_stream"]),
            ).fetchall()
            times = defaultdict(float)
            gen_backward = []
            gen_preprocess = []
            for name, x, y, z, count, ns in rows:
                if category(name) != "attention":
                    continue
                label = stage(name)
                seconds = ns / 1e9 / microbatches
                times[label] += seconds
                detail = {
                    "n": n,
                    "gpu": gpu["device"],
                    "stream": gpu["primary_stream"],
                    "stage": label,
                    "kernel": name,
                    "grid": [x, y, z],
                    "calls_in_capture": count,
                    "seconds_per_microbatch": seconds,
                }
                details.append(detail)
                if label == "backward_main" and "flash::SingleTileScheduler<" in name:
                    gen_backward.append(detail)
                if "FlashAttnBwdPreprocess<cute::tuple<cute::C<(int)80>" in name:
                    gen_preprocess.append(detail)
            assert len(gen_backward) == len(gen_preprocess) == 1, (n, gpu["device"])
            backward, preprocess = gen_backward[0], gen_preprocess[0]
            assert backward["grid"] == [512, 32, n + 1], backward
            assert preprocess["grid"] == [745, 32, n + 1], preprocess
            assert backward["calls_in_capture"] == preprocess["calls_in_capture"] == 36 * microbatches
            times["attention"] = sum(times.values())
            assert math.isclose(
                times["attention"], gpu["categories"]["attention"]["seconds"] / microbatches, abs_tol=1e-9
            ), (n, gpu["device"])
            times["gen_backward_main"] = backward["seconds_per_microbatch"]
            per_gpu.append(times)
    finally:
        con.close()
    assert len(per_gpu) == 4, n
    summary = {"n": n}
    for key in (
        "forward",
        "backward_main",
        "backward_aux",
        "metadata",
        "compiled_aux",
        "attention",
        "gen_backward_main",
    ):
        summary[f"{key}_seconds_per_microbatch"] = statistics.mean(gpu[key] for gpu in per_gpu)
    summary.update(
        gen_backward_calls_per_gpu_capture=36 * microbatches,
        gen_backward_grid_x=512,
        gen_backward_grid_y=32,
        gen_backward_grid_z=n + 1,
        gen_backward_blocks_per_call=512 * 32 * (n + 1),
        gen_preprocess_blocks_per_call=745 * 32 * (n + 1),
    )
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/v0.2.6_seqlen_efficiency"))
    parser.add_argument("--microbatches", type=int, default=4, help="Captured optimizer steps × gradient accumulation")
    args = parser.parse_args()
    if args.microbatches <= 0:
        parser.error("--microbatches must be positive")
    summaries, details = [], []
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        summary, records = analyze(args.root / "runs" / f"t2v_n{n:03d}", n, args.microbatches)
        summaries.append(summary)
        details.extend(records)
        print(
            f"N={n:3d}: fwd={summary['forward_seconds_per_microbatch']:.3f}, "
            f"bwd={summary['backward_main_seconds_per_microbatch']:.3f}, "
            f"aux={summary['backward_aux_seconds_per_microbatch']:.3f} s/microbatch; "
            f"GEN bwd blocks/call={summary['gen_backward_blocks_per_call']:,}"
        )
    with (args.root / "kernel_detail.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.root / "kernel_detail.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    main()
