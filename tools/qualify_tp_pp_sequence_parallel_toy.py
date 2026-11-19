# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serial CUDA/NCCL FP32 qualification for the isolated three-stream SP toy.

Run only as a child of the shared-lock GPU queue, never directly alongside
other GPU workloads. This is a synthetic two-layer mathematical control, not
production training, full-model numerical acceptance or a throughput result.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def gate(directory, kind, world):
    checks = []
    for rank in range(world):
        report = json.loads((directory / f"rank{rank}.json").read_text())
        if (
            report.get("passed") is not True
            or report.get("rank") != rank
            or report.get("world") != world
            or report.get("backend") != "nccl"
            or report.get("production_acceptance") is not False
        ):
            raise ValueError(f"Invalid {kind} rank {rank} report")
        if kind == "adjoint":
            controls = report["loss_scale_negative_control"]
            if controls.get("gather_loss_divisor_1") != world or controls.get(f"gather_loss_divisor_{world}") != 1.0:
                raise ValueError("Replicated-loss adjoint control failed")
        else:
            steps = report.get("steps", [])
            if (
                report.get("dtype") != "float32"
                or len(steps) != 3
                or [step.get("update") for step in steps] != [0, 1, 2]
                or steps[1].get("missing_action") is not True
                or steps[2].get("no_generation_targets") is not True
                or any(step["gradient"]["compared_tensors"] < 56 for step in steps)
                or any(step["optimizer"]["compared_tensors"] != 126 for step in steps)
            ):
                raise ValueError("Toy optimizer, gradients, or missing-input coverage incomplete")
        checks.append({"rank": rank, "passed": True})
    return {"passed": True, "ranks": checks}


def stop_owned(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_case(name, world, output, visible, timeout):
    python = ROOT / ".venv/bin/python"
    worker = (
        ROOT
        / "tests/megatron"
        / ("tp_pp_sequence_parallel_adjoint.py" if name == "adjoint" else "tp_pp_sequence_parallel_toy.py")
    )
    case_dir = output / f"{name}{world}"
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        str(worker),
        "--backend",
        "nccl",
        "--output",
        str(case_dir),
    ]
    if name == "toy":
        command += ["--dtype", "float32"]
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(visible[:world]),
        PYTHONPATH=str(ROOT),
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = {"command": command, "started": time.time()}
    with (output / f"{name}{world}.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        result.update(pid=process.pid, create_time=psutil.Process(process.pid).create_time())
        try:
            result["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_owned(process)
            raise TimeoutError(f"{name}{world} exceeded {timeout} seconds") from None
    result["ended"] = time.time()
    if result["returncode"]:
        raise RuntimeError(f"{name}{world} failed with exit code {result['returncode']}")
    result["gate"] = gate(case_dir, name, world)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 4 or len(set(visible)) != 4 or any(not value for value in visible):
        raise ValueError("Four distinct GPU identities from the outer queue are required")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "summary.json"
    report = {
        "passed": False,
        "scope": "synthetic two-layer three-stream FP32 CUDA/NCCL toy",
        "production_acceptance": False,
        "performance_acceptance": False,
        "gpu_exclusivity": "cooperative outer lock and polling; external transient activity cannot be excluded",
        "cases": {},
    }
    try:
        for world in (2, 4):
            for kind in ("adjoint", "toy"):
                name = f"{kind}{world}"
                report["cases"][name] = run_case(kind, world, args.output, visible, args.timeout)
                target.write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = True
    except BaseException as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
