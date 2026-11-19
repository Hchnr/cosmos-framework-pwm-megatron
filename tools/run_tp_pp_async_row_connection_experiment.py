# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare CUDA connection limits with identical TP4/FSDP4 prepared inputs.

Run only inside the shared GPU lock. Numerical tiny checks and synthetic Nsight
overlap observations precede four-rank timing. The benchmark excludes VAE and
media loading; it does not establish full-model numerical acceptance.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.run_tp_pp_async_row_gpu_sequence import NSYS, PYTHON, runtime, save, verify_snapshot, verify_tiny


def require_terminal_queue(result):
    if result.get("returncode") is None:
        raise ValueError("Earlier qualification must have a terminal child return code")
    return result["returncode"]


def require_conn8_overlap(result):
    if not result.get("cases", {}).get("conn8_profile", {}).get("overlap", {}).get("passed"):
        raise ValueError("Connection 8 did not demonstrate increased overlap in the synthetic trace")


def benchmark_launch(guards, mode, folder):
    if mode not in ("tp_shared", "tp_async", "fsdp"):
        raise ValueError(f"Unexpected benchmark mode: {mode}")
    backend = "fsdp" if mode == "fsdp" else "tp"
    bench = "tools/run_tp4_fsdp4_microbatch_benchmark.py"
    distributed = [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"]
    arguments = guards.benchmark_args(backend, folder, 7)
    if mode == "tp_shared":
        return backend, distributed + ["tools/run_tp_pp_shared_input_reduction.py", "--run-path", bench, *arguments]
    if mode == "tp_async":
        return backend, distributed + [
            "tools/run_tp_pp_async_row_reduction.py",
            "--entry",
            bench,
            "--shared-input-reduction",
            *arguments,
        ]
    return backend, distributed + [bench, *arguments]


def profile(guards, *, snapshot, output, env, visible, timeout, connection):
    capture = output / f"conn{connection}_probe"
    report_dir = output / f"conn{connection}_probe_reports"
    command = [
        str(NSYS),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--kill=none",
        "--wait=all",
        "-o",
        str(capture),
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(snapshot / "tests/megatron/tp_pp_async_row_gpu_probe.py"),
        "--output",
        str(report_dir),
    ]
    case = guards.run_case(f"conn{connection}_probe", command, output, snapshot, env, timeout, visible)
    if case["returncode"]:
        raise RuntimeError(f"Connection {connection} synthetic trace failed")
    for rank in range(4):
        row = json.loads((report_dir / f"rank{rank}.json").read_text())
        if row["rank"] != rank or row["world"] != 4 or not all(row["outputs_exact"]):
            raise ValueError(f"Connection {connection} synthetic rank {rank} mismatch")
    database = capture.with_suffix(".sqlite")
    subprocess.run(
        [str(NSYS), "export", "--type", "sqlite", "--output", str(database), str(capture.with_suffix(".nsys-rep"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    spec = importlib.util.spec_from_file_location(
        "_async_row_conn_analysis", snapshot / "tools/analyze_tp_pp_async_row_nsys.py"
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    case["overlap"] = analysis.analyze(database, warmups=2, steps=5)
    save(output / f"conn{connection}_probe_analysis.json", case["overlap"])
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=10800)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("Positive timeout required")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "summary.json"
    result = {
        "completed": False,
        "failure": None,
        "full_model_numerical_acceptance": False,
        "performance_acceptance": False,
        "scope": "matching prepared inputs; media, VAE/noising and checkpoint writes excluded",
        "gpu_exclusivity": "provisional: cooperative lock and polling cannot exclude transient foreign use",
        "cases": {},
    }
    save(target, result)
    try:
        prerequisite = json.loads(args.prerequisite.read_text())
        # A connection-1 trace can legitimately show no overlap. Rerun all
        # numerical gates independently rather than requiring that trace to pass.
        result["previous_qualification_returncode"] = require_terminal_queue(prerequisite)
        guards = runtime()
        baseline = json.loads(args.baseline_manifest.read_text())
        if guards.fingerprints(args.snapshot) != baseline.get("files"):
            raise ValueError("Source or prepared input changed since the queue was created")
        initial = verify_snapshot(args.snapshot, json.loads(args.snapshot_manifest.read_text()))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 4 or len(set(visible)) != 4 or not all(value.startswith("GPU-") for value in visible):
            raise ValueError("Outer queue must pin four distinct GPU UUIDs")
        template_env = dict(guards.load_sequence().ENV, PYTHONPATH=str(args.snapshot), PYTHONDONTWRITEBYTECODE="1")
        worker = args.snapshot / "tests/megatron/tp_pp_async_row_layer_worker.py"
        for connection in (1, 8):
            env = dict(template_env, CUDA_DEVICE_MAX_CONNECTIONS=str(connection))
            for world in (2, 4):
                for dtype, compiled in (("fp32", False), ("fp32", True), ("bf16", True)):
                    name = f"conn{connection}_tiny_tp{world}_{dtype}_{'inductor_ac' if compiled else 'eager'}"
                    folder = args.output / name
                    command = [
                        str(PYTHON),
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc-per-node={world}",
                        str(worker),
                        "--worktree",
                        str(args.snapshot),
                        "--output",
                        str(folder),
                        "--backend",
                        "nccl",
                        "--dtype",
                        dtype,
                    ]
                    if compiled:
                        command += ["--compile", "--full-ac", "--compile-backend", "inductor"]
                    case = guards.run_case(
                        name, command, args.output, args.snapshot, env, args.timeout, visible[:world]
                    )
                    result["cases"][name] = case
                    save(target, result)
                    if case["returncode"]:
                        raise RuntimeError(f"{name} exited {case['returncode']}")
                    case["ranks"] = verify_tiny(folder, world, compiled=compiled, dtype=dtype)
            result["cases"][f"conn{connection}_profile"] = profile(
                guards,
                snapshot=args.snapshot,
                output=args.output,
                env=env,
                visible=visible,
                timeout=args.timeout,
                connection=connection,
            )
            save(target, result)
        require_conn8_overlap(result)
        if guards.fingerprints(args.snapshot) != baseline["files"]:
            raise ValueError("Source or input changed before performance measurements")
        for connection in (1, 8):
            env = dict(template_env, CUDA_DEVICE_MAX_CONNECTIONS=str(connection))
            for mode in ("tp_shared", "tp_async", "fsdp"):
                name = f"conn{connection}_{mode}"
                folder = args.output / name
                backend, command = benchmark_launch(guards, mode, folder)
                case = guards.run_case(name, command, args.output, args.snapshot, env, args.timeout, visible)
                result["cases"][name] = case
                save(target, result)
                if case["returncode"]:
                    raise RuntimeError(f"{name} benchmark failed")
                case["peak_cuda_allocated_bytes_per_rank"] = guards.peak_training_memory(folder, backend)
                if guards.fingerprints(args.snapshot) != baseline["files"]:
                    raise RuntimeError(f"Source or input changed during {name}")
        sys.path.insert(0, str(args.snapshot))
        from tools.compare_tp4_fsdp4_microbatch_benchmark import compare

        result["comparisons"] = {}
        for connection in (1, 8):
            result["comparisons"][f"conn{connection}_shared"] = compare(
                args.output / f"conn{connection}_tp_shared", args.output / f"conn{connection}_fsdp"
            )
            result["comparisons"][f"conn{connection}_async"] = compare(
                args.output / f"conn{connection}_tp_async", args.output / f"conn{connection}_fsdp"
            )
        result["snapshot_sha256_after"] = verify_snapshot(args.snapshot, json.loads(args.snapshot_manifest.read_text()))
        if result["snapshot_sha256_after"] != initial:
            raise RuntimeError("Snapshot changed during qualification")
        result["completed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(target, result)


if __name__ == "__main__":
    main()
