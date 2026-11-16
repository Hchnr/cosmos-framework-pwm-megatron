# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sequential fresh-process Action loader ablation (no model or online logging).

Requires the same model-path environment as validate_action128_prefetch.
Results include commands, revisions, exit status and separate per-run logs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cases", nargs="+", help="Optional subset; also supports shared_w23_k8")
    parser.add_argument("--episode-columns", nargs="+", choices=["all", "projected"], default=["all", "projected"])
    parser.add_argument("--pin-memory-mode", choices=["background", "inline"], default="background")
    args = parser.parse_args()
    # Match the production launcher's process-local policy in every cell.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
    args.output.mkdir(parents=True, exist_ok=True)
    configs = [
        ("legacy_w23_f1", "legacy_per_source", 23, 1),
        ("legacy_w23_f2", "legacy_per_source", 23, 2),
        ("shared_w23_k4", "shared_prefetch", 23, 4),
        ("shared_w2_k4", "shared_prefetch", 2, 4),
        ("shared_w2_k1", "shared_prefetch", 2, 1),
    ]
    if args.cases:
        available = {
            config[0]: config
            for config in [
                *configs,
                *((f"shared_w23_k{k}", "shared_prefetch", 23, k) for k in (8, 16, 32, 48)),
                *((f"shared_w{w}_k{k}", "shared_prefetch", w, k) for w in (4, 8) for k in (16, 32)),
            ]
        }
        unknown = set(args.cases) - available.keys()
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        configs = [available[name] for name in args.cases]
    cases = [(columns, config) for columns in args.episode_columns for config in configs]
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "logical_workers_per_source": 1,
        "transparent_hugepages_disabled": True,
        "pin_memory_mode": args.pin_memory_mode,
        "runs": [],
    }
    for repeat in range(args.repeats):
        # Reverse order on repeat to reduce systematic cache/order bias. Never
        # run competing configurations concurrently on this shared host.
        for columns, (name, backend, workers, prefetch) in cases if repeat % 2 == 0 else reversed(cases):
            run_name = f"{columns}_{name}_r{repeat}"
            output = args.output / f"{run_name}.json"
            command = [
                sys.executable,
                "-m",
                "tools.validate_action128_prefetch",
                "--recipe",
                "examples/toml/sft_config/action128_wam_nano_160gpu.toml",
                "--backend",
                backend,
                "--workers",
                str(workers),
                "--legacy-workers",
                str(workers),
                "--prefetch",
                str(prefetch),
                "--logical-workers",
                "1",
                "--episode-columns",
                columns,
                "--pin-memory-mode",
                args.pin_memory_mode,
                "--benchmark",
                "--batches",
                str(args.batches),
                "--warmup",
                str(args.warmup),
                "--output",
                str(output),
            ]
            print(f"START {run_name}", flush=True)
            start = time.monotonic()
            with (args.output / f"{run_name}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    exit_code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    exit_code = "timeout"
            manifest["runs"].append(
                {
                    "name": run_name,
                    "command": command,
                    "exit_code": exit_code,
                    "wall_seconds": time.monotonic() - start,
                }
            )
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"DONE {run_name}: {exit_code}", flush=True)
            if exit_code != 0:
                raise RuntimeError(f"Ablation failed: {run_name}; inspect its log before continuing")


if __name__ == "__main__":
    main()
