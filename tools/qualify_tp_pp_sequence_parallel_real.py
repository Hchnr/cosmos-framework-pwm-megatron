# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serial TP2/TP4 real-layer SP qualification; no production or speed claim.

NCCL invocation must be placed behind the cooperative GPU queue. Output must
be a fresh directory, so stale success files cannot satisfy this gate.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests/megatron/tp_pp_sequence_parallel_real_worker.py"


def qualify(backend, output, prerequisite=None):
    if backend == "nccl":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 4 or len(set(visible)) != 4 or any(not value for value in visible):
            raise ValueError("Four distinct GPU identities from the outer queue are required")
        if prerequisite is None or json.loads(prerequisite.read_text()).get("passed") is not True:
            raise ValueError("The prior synthetic NCCL qualification must pass first")
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LD_LIBRARY_PATH="")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), environment.get("PYTHONPATH"))))
    if backend == "gloo":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for world in (2, 4):
        directory = output / f"tp{world}"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
               str(WORKER), "--backend", backend, "--output", str(directory)]
        with (output / f"tp{world}.log").open("x") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time(), "started": time.time()}
            (output / f"tp{world}_process.json").write_text(json.dumps(identity, indent=2) + "\n")
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                raise
            if code:
                raise RuntimeError(f"TP{world} qualification failed with exit code {code}")
        for rank in range(world):
            report = json.loads((directory / f"rank{rank}.json").read_text())
            if not (report["passed"] and report["rank"] == rank and report["world"] == world
                    and report["backend"] == backend and report["full_ac"] and len(report["steps"]) == 4
                    and report["negative_controls"]["dummy_gradient_factor"] == world
                    and report["negative_controls"]["replicated_output_gradient_factor"] == 1 / world):
                raise ValueError(f"Incomplete or unexpected TP{world} rank {rank} report")
            if backend == "nccl" and not report["cuda_peak_allocated_bytes"]:
                raise ValueError("CUDA peak memory missing")
            results.append(report)
    summary = {"passed": True, "backend": backend, "world_sizes": [2, 4], "rank_reports": len(results),
               "scope": "two real decoder layers and real I/O; synthetic inputs/loss; no production or throughput qualification"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prerequisite-summary", type=Path)
    args = parser.parse_args()
    qualify(args.backend, args.output.resolve(), args.prerequisite_summary)


if __name__ == "__main__":
    main()
