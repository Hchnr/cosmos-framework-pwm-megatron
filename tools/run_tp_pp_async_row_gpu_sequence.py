# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify delayed row waits serially under an already-held GPU lock.

Runs isolated tiny MCore numerical checks before a synthetic real-shape Nsight
probe. The outer queue must establish GPU exclusivity. None of these checks
establishes full-model correctness or aggregate training throughput.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/opt/cosmos-tp-pp-work-20261119")
PYTHON = ROOT / ".venv/bin/python"
NSYS = Path("/usr/local/cuda/bin/nsys")
OVERLAYS = (
    "tools/run_tp_pp_async_row_reduction.py",
    "tools/analyze_tp_pp_async_row_nsys.py",
    "tests/megatron/tp_pp_async_row_layer_worker.py",
    "tests/megatron/tp_pp_async_row_gpu_probe.py",
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprints(snapshot):
    paths = sorted(path for sub in ("cosmos_framework", "tools", "tests") for path in (snapshot / sub).rglob("*.py"))
    return {str(path.relative_to(snapshot)): digest(path) for path in paths}


def build_snapshot(snapshot):
    if snapshot.exists():
        raise FileExistsError(snapshot)
    shutil.copytree(BASE, snapshot, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
    for relative in OVERLAYS:
        shutil.copy2(ROOT / relative, snapshot / relative)
    for sub in ("cosmos_framework", "tools", "tests"):
        for source in (BASE / sub).rglob("*.py"):
            relative = source.relative_to(BASE)
            if str(relative) not in OVERLAYS and digest(source) != digest(snapshot / relative):
                raise ValueError(f"Base changed while snapshotting: {relative}")
    for relative in OVERLAYS:
        if digest(ROOT / relative) != digest(snapshot / relative):
            raise ValueError(f"Candidate changed while snapshotting: {relative}")


def verify_snapshot(snapshot, manifest):
    if manifest.get("snapshot") != str(snapshot.resolve()):
        raise ValueError("Snapshot path does not match the queued manifest")
    initial = fingerprints(snapshot)
    if initial != manifest.get("files"):
        raise ValueError("Snapshot source files changed while waiting")
    return initial


def runtime():
    path = ROOT / "tools/run_tp_pp_flash3_benchmark_sequence.py"
    spec = importlib.util.spec_from_file_location("_async_row_gpu_guards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify_tiny(directory, world, *, compiled, dtype):
    evidence = []
    for rank in range(world):
        item = json.loads((directory / f"rank{rank}.json").read_text())
        cases = item["cases"]
        if item["rank"] != rank or len(cases) != 4 or any(not case["passed"] for case in cases):
            raise ValueError(f"Rank {rank} tiny numerical gate failed")
        if any(case["dtype"] != dtype or case["full_activation_checkpointing"] != compiled for case in cases):
            raise ValueError(f"Rank {rank} tiny test configuration mismatch")
        if compiled and any(case["compile_backend"] != "inductor" for case in cases):
            raise ValueError(f"Rank {rank} did not run the requested Inductor backend")
        if not compiled and any(
            case["baseline_forward_sums"] != 6 or case["candidate_forward_sums"] != 6 for case in cases
        ):
            raise ValueError(f"Rank {rank} changed the number of row collectives")
        evidence.append(
            {
                "rank": rank,
                "max_relative_l2": max(case["max_relative_l2"] for case in cases),
                "peak_cuda_bytes": item["peak_cuda_bytes"],
            }
        )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("Positive per-case timeout required")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "summary.json"
    result = {
        "completed": False,
        "passed": False,
        "failure": None,
        "scope": "tiny MCore numerical checks and synthetic forward overlap; not full-model acceptance",
        "gpu_exclusivity": "provisional: outer cooperative lock and idle polling cannot exclude transient foreign use",
        "cases": {},
    }
    save(target, result)
    try:
        guards = runtime()
        manifest = json.loads(args.snapshot_manifest.read_text())
        initial = verify_snapshot(args.snapshot, manifest)
        result["snapshot_sha256_before"] = initial
        sequence = guards.load_sequence()
        env = dict(sequence.ENV, PYTHONPATH=str(args.snapshot))
        rows = guards.gpu_memory()
        if len(rows) != 4:
            raise ValueError("Four GPU devices required")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 4 or not all(part.startswith("GPU-") for part in visible) or len(set(visible)) != 4:
            raise ValueError("Outer queue must pin four distinct GPU UUIDs")
        worker = args.snapshot / "tests/megatron/tp_pp_async_row_layer_worker.py"
        for world in (2, 4):
            for dtype, compiled in (("fp32", False), ("fp32", True), ("bf16", True)):
                name = f"tiny_tp{world}_{dtype}_{'inductor_ac' if compiled else 'eager'}"
                output = args.output / name
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
                    str(output),
                    "--backend",
                    "nccl",
                    "--dtype",
                    dtype,
                ]
                if compiled:
                    command += ["--compile", "--full-ac", "--compile-backend", "inductor"]
                case = guards.run_case(name, command, args.output, args.snapshot, env, args.timeout, visible[:world])
                result["cases"][name] = case
                save(target, result)
                if case["returncode"]:
                    raise RuntimeError(f"{name} exited {case['returncode']}")
                case["ranks"] = verify_tiny(output, world, compiled=compiled, dtype=dtype)
        capture = args.output / "row_probe"
        probe_output = args.output / "row_probe_reports"
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
            str(args.snapshot / "tests/megatron/tp_pp_async_row_gpu_probe.py"),
            "--output",
            str(probe_output),
        ]
        case = guards.run_case("profile_probe", command, args.output, args.snapshot, env, args.timeout, visible)
        result["cases"]["profile_probe"] = case
        save(target, result)
        if case["returncode"]:
            raise RuntimeError("Synthetic Nsight probe failed")
        for rank in range(4):
            row = json.loads((probe_output / f"rank{rank}.json").read_text())
            if row["rank"] != rank or row["world"] != 4 or not all(row["outputs_exact"]):
                raise ValueError(f"Rank {rank} synthetic probe did not match")
        database = capture.with_suffix(".sqlite")
        subprocess.run(
            [str(NSYS), "export", "--type", "sqlite", "--output", str(database), str(capture.with_suffix(".nsys-rep"))],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        spec = importlib.util.spec_from_file_location(
            "_async_row_analysis", args.snapshot / "tools/analyze_tp_pp_async_row_nsys.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        analysis = module.analyze(database, warmups=2, steps=5)
        save(args.output / "row_probe_analysis.json", analysis)
        if not analysis["passed"]:
            raise ValueError("Nsight did not establish increased NCCL/compute overlap")
        result["snapshot_sha256_after"] = fingerprints(args.snapshot)
        if result["snapshot_sha256_after"] != initial:
            raise RuntimeError("Snapshot changed during qualification")
        result["passed"] = True
        result["completed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(target, result)


if __name__ == "__main__":
    main()
