# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run qualified FA3 experiments serially inside an already-held GPU lock.

This runner is intended as the child of queue_tp_pp_sharded_vae_gpu.py, which
waits for the predecessor, verifies PID/create_time and holds the shared lock.
The single-GPU real-FA3 gate must pass before any 4-GPU run. Timed and Nsight
captures are separate; no full-model numerical or strict-exclusivity claim is
made. GPU and cgroup guards may terminate only this runner's own process group.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/opt/cosmos-tp-pp-work-20261119")
OUTPUTS = ROOT / "outputs/megatron_tp4_fsdp4_microbatch_20261119"
SPOOL = (
    ROOT
    / "outputs/megatron_tp_pp_default_gc16_full576_strict_restart_20261118/training/cosmos3_action/action128/uninterrupted/prepared_spool/iter000001200.t4z9_emx"
)
TP2 = Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp2_pp2_recovered_20261119")
TP4 = Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp4_pp1_20261119")
TOML = ROOT / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
PYTHON = ROOT / ".venv/bin/python"
NSYS = Path("/usr/local/cuda/bin/nsys")
OVERLAYS = (
    "run_tp_pp_flash3_selective_ac.py",
    "qualify_tp_pp_flash3_selective_ac.py",
    "profile_tp_fsdp_flash3_ac.py",
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_snapshot(snapshot):
    if snapshot.exists():
        raise FileExistsError(snapshot)
    shutil.copytree(BASE, snapshot, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
    for name in OVERLAYS:
        shutil.copy2(ROOT / "tools" / name, snapshot / "tools" / name)
    for base in ("tools", "tests", "cosmos_framework"):
        for source in (BASE / base).rglob("*.py"):
            if source.parent == BASE / "tools" and source.name in OVERLAYS:
                continue
            if digest(source) != digest(snapshot / source.relative_to(BASE)):
                raise ValueError(f"Snapshot differs from isolated base: {source}")


def baseline_fingerprints():
    files = sorted(path for base in ("tools", "tests", "cosmos_framework") for path in (BASE / base).rglob("*.py"))
    return {str(path.relative_to(BASE)): digest(path) for path in files}


def fingerprints(snapshot):
    files = sorted(path for base in ("tools", "tests", "cosmos_framework") for path in (snapshot / base).rglob("*.py"))
    assets = [TOML, TP2 / "manifest.json", TP4 / "manifest.json", Path(__file__)]
    assets += [SPOOL / f"update000001200.micro{index:03d}.rank000.pt" for index in range(3)]
    return {str(path): digest(path) for path in [*files, *assets]}


def load_sequence():
    path = ROOT / "artifacts/megatron_tp_pp/run_fp32_and_microbatch_sequence_20261119.py"
    spec = importlib.util.spec_from_file_location("_tp_pp_sequence_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def memory_state():
    base = Path("/sys/fs/cgroup")
    current = int((base / "memory.current").read_text())
    maximum = (base / "memory.max").read_text().strip()
    stats = {
        key: int(value) for key, value in (line.split() for line in (base / "memory.stat").read_text().splitlines())
    }
    events = {
        key: int(value) for key, value in (line.split() for line in (base / "memory.events").read_text().splitlines())
    }
    host = psutil.virtual_memory().available
    cgroup = (
        host
        if maximum == "max"
        else max(0, int(maximum) - current) + stats.get("inactive_file", 0) + stats.get("slab_reclaimable", 0)
    )
    return {"headroom": min(host, cgroup), "events": events, "current": current, "limit": maximum}


def gpu_memory():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    ).stdout
    rows = [tuple(int(part.strip()) for part in line.split(",")) for line in out.splitlines() if line.strip()]
    if sorted(index for index, _, _ in rows) != [0, 1, 2, 3]:
        raise ValueError("Expected all four visible GPU inventory rows")
    return rows


def terminate_owned(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_case(name, command, output, snapshot, env, timeout, visible):
    environment = env.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(output / f"compile_cache_{name}")
    before = memory_state()
    if before["headroom"] < 24 * 1024**3:
        raise RuntimeError("Insufficient conservative cgroup/host memory headroom")
    result = {"command": [str(item) for item in command], "memory_before": before, "started": time.time()}
    with (output / f"{name}.log").open("x") as handle:
        process = subprocess.Popen(
            command, cwd=snapshot, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        result["pid"] = process.pid
        result["create_time"] = psutil.Process(process.pid).create_time()
        deadline = time.monotonic() + timeout
        maximum_used = {str(index): 0 for index in range(4)}
        low_free_polls = 0
        try:
            while process.poll() is None:
                current = memory_state()
                if current["headroom"] < 24 * 1024**3:
                    raise RuntimeError("Conservative cgroup/host headroom below 24 GiB")
                if any(current["events"].get(key, 0) > before["events"].get(key, 0) for key in ("oom", "oom_kill")):
                    raise RuntimeError("Cgroup OOM event counter increased")
                rows = gpu_memory()
                for index, _, used in rows:
                    maximum_used[str(index)] = max(maximum_used[str(index)], used)
                low_free_polls = low_free_polls + 1 if any(total - used < 256 for _, total, used in rows) else 0
                if low_free_polls >= 2:
                    raise RuntimeError("GPU free memory below 256 MiB for two consecutive polls")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} exceeded {timeout} seconds")
                time.sleep(2)
        except BaseException:
            terminate_owned(process)
            raise
        finally:
            result["ended"] = time.time()
            result["peak_nvml_used_mib"] = maximum_used
            result["memory_after"] = memory_state()
        result["returncode"] = process.wait()
    return result


def benchmark_args(backend, output, updates):
    return [
        "--backend",
        backend,
        "--bundle",
        str(TP4 if backend == "tp" else TP2),
        "--prepared-spool",
        str(SPOOL),
        "--output",
        str(output),
        "--toml",
        str(TOML),
        "--updates",
        str(updates),
        "--warmup-updates",
        "2",
    ]


def count_flash_kernels(database):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT k.deviceId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName "
            "WHERE s.value LIKE '%FlashAttnFwdSm90%' GROUP BY k.deviceId ORDER BY k.deviceId"
        ).fetchall()
    finally:
        connection.close()
    return {str(device): count for device, count in rows}


def require_policy_reports(directory, backend, layers, *, profiled=False):
    results = []
    for rank in range(4):
        name = f"profile.rank{rank}.json" if profiled else f"flash3_selective_ac.rank{rank}.json"
        report = json.loads((directory / name).read_text())
        counts = report["policy_trace_counts"]
        key = "tp_selected_layers" if backend == "tp" else "native_selected_layers"
        if (
            report["rank"] != rank
            or report["selected_layers"] != layers
            or report["failure"] is not None
            or counts[key] != layers
            or counts["forward"] < 1
        ):
            raise ValueError(f"Missing selected FA3 policy evidence: {backend} rank {rank}")
        results.append({"rank": rank, "selected_layers": counts[key], "trace_counts": counts})
    return results


def peak_training_memory(root, backend):
    folder = root / "training"
    if backend == "tp":
        folder /= "cosmos3_action/action128/tp4_microbatch"
    result = []
    for rank in range(4):
        rows = [json.loads(line) for line in (folder / f"updates.rank{rank}.jsonl").read_text().splitlines()]
        if len(rows) != 7 or any("peak_cuda_allocated_bytes" not in row for row in rows):
            raise ValueError(f"Incomplete peak memory data for rank {rank}")
        result.append(max(row["peak_cuda_allocated_bytes"] for row in rows))
    return result


def require_single_gate(report):
    if report.get("passed") is not True:
        raise ValueError("Single-GPU production FA3 qualification failed")
    for mode in ("eager", "inductor"):
        if (
            report.get("cases", {}).get(f"{mode}_full", {}).get("flash3_forward_cuda_kernel_count") != 2
            or report.get("cases", {}).get(f"{mode}_selective", {}).get("flash3_forward_cuda_kernel_count") != 1
            or report.get("comparisons", {}).get(mode, {}).get("passed") is not True
        ):
            raise ValueError(f"Missing actual {mode} FA3 forward or gradient evidence")


def save(path, result):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--selective-layers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10800)
    args = parser.parse_args()
    if not 0 < args.selective_layers <= 36 or args.timeout <= 0:
        parser.error("Require 1–36 layers and a positive case timeout")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 4 or any(not value for value in visible) or len(set(visible)) != 4:
        raise ValueError("The outer queue must supply four distinct CUDA-visible devices")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "summary.json"
    result = {
        "passed": False,
        "numerical_acceptance": False,
        "performance_acceptance": False,
        "gpu_exclusivity": "provisional: external cooperative lock and polling, host PID ownership unavailable",
        "cases": {},
    }
    save(target, result)
    try:
        baseline = json.loads(args.baseline_manifest.read_text())
        if baseline.get("files") != baseline_fingerprints():
            raise RuntimeError("Isolated baseline source changed after queueing")
        result["baseline_manifest_sha256"] = digest(args.baseline_manifest)
        copy_snapshot(args.snapshot)
        hashes = fingerprints(args.snapshot)
        result["source_and_input_sha256"] = hashes
        env = load_sequence().ENV.copy()
        env.update(PYTHONPATH=str(args.snapshot), PYTHONDONTWRITEBYTECODE="1")
        single_report = args.output / "single_fa3.json"
        single = run_case(
            "single_fa3",
            [str(PYTHON), "tools/qualify_tp_pp_flash3_selective_ac.py", "--output", str(single_report)],
            args.output,
            args.snapshot,
            env,
            args.timeout,
            visible[:1],
        )
        result["cases"]["single_fa3"] = single
        save(target, result)
        if single["returncode"] != 0:
            raise RuntimeError("Single-GPU FA3 qualification failed")
        require_single_gate(json.loads(single_report.read_text()))
        bench = "tools/run_tp4_fsdp4_microbatch_benchmark.py"
        distributed = [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"]
        for backend in ("tp", "fsdp"):
            for selected in (False, True):
                label = f"{backend}_{'selective' if selected else 'full'}"
                train_output = args.output / label
                parameters = benchmark_args(backend, train_output, 7)
                if selected:
                    command = distributed + [
                        "tools/run_tp_pp_flash3_selective_ac.py",
                        "--entry",
                        bench,
                        "--report-dir",
                        str(args.output / f"{label}_reports"),
                        "--selective-layers",
                        str(args.selective_layers),
                    ]
                    if backend == "tp":
                        command.append("--shared-input-reduction")
                    command += parameters
                elif backend == "tp":
                    command = distributed + [
                        "tools/run_tp_pp_shared_input_reduction.py",
                        "--run-path",
                        bench,
                        *parameters,
                    ]
                else:
                    command = distributed + [bench, *parameters]
                case = run_case(label, command, args.output, args.snapshot, env, args.timeout, visible)
                result["cases"][label] = case
                save(target, result)
                if case["returncode"]:
                    raise RuntimeError(f"{label} failed; no matched performance comparison")
                case["peak_cuda_allocated_bytes_per_rank"] = peak_training_memory(train_output, backend)
                if selected:
                    case["policy_reports"] = require_policy_reports(
                        args.output / f"{label}_reports", backend, args.selective_layers
                    )
                if fingerprints(args.snapshot) != hashes:
                    raise RuntimeError(f"Source or input changed during {label}")
        sys.path.insert(0, str(args.snapshot))
        from tools.compare_tp4_fsdp4_microbatch_benchmark import compare

        result["full_comparison"] = compare(args.output / "tp_full", args.output / "fsdp_full")
        result["selective_comparison"] = compare(args.output / "tp_selective", args.output / "fsdp_selective")
        save(target, result)
        for backend in ("tp", "fsdp"):
            for selected in (False, True):
                label = f"profile_{backend}_{'selective' if selected else 'full'}"
                capture = args.output / label
                reports = args.output / f"{label}_reports"
                command = [
                    str(NSYS),
                    "profile",
                    "--trace=cuda,nvtx",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--kill=none",
                    "--wait=all",
                    "-o",
                    str(capture),
                    *distributed,
                    "tools/profile_tp_fsdp_flash3_ac.py",
                    "--capture-iteration",
                    "1202",
                    "--selective-layers",
                    str(args.selective_layers if selected else 0),
                    "--report-dir",
                    str(reports),
                ]
                if backend == "tp":
                    command.append("--shared-input-reduction")
                command += benchmark_args(backend, args.output / f"{label}_training", 3)
                case = run_case(label, command, args.output, args.snapshot, env, args.timeout, visible)
                result["cases"][label] = case
                save(target, result)
                if case["returncode"]:
                    raise RuntimeError(f"{label} Nsight capture failed")
                for rank in range(4):
                    report = json.loads((reports / f"profile.rank{rank}.json").read_text())
                    if not (report["capture_started"] and report["capture_finished"] and report["failure"] is None):
                        raise ValueError(f"{label} missing complete capture markers for rank {rank}")
                if selected:
                    case["policy_reports"] = require_policy_reports(
                        reports, backend, args.selective_layers, profiled=True
                    )
                database = capture.with_suffix(".sqlite")
                export = subprocess.run(
                    [
                        str(NSYS),
                        "export",
                        "--type",
                        "sqlite",
                        "--output",
                        str(database),
                        str(capture.with_suffix(".nsys-rep")),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                case["export_returncode"] = export.returncode
                case["flash3_forward_kernels_per_gpu"] = count_flash_kernels(database)
                if fingerprints(args.snapshot) != hashes:
                    raise RuntimeError(f"Source or input changed during {label}")
                save(target, result)
        expected_full, expected_selected = 216, 216 - 3 * args.selective_layers
        for backend in ("tp", "fsdp"):
            for mode, expected in (("full", expected_full), ("selective", expected_selected)):
                actual = result["cases"][f"profile_{backend}_{mode}"]["flash3_forward_kernels_per_gpu"]
                if actual != {str(rank): expected for rank in range(4)}:
                    raise ValueError(f"Unexpected {backend} {mode} FA3 kernel counts: {actual}")
        result["source_and_input_sha256_unchanged"] = True
        result["passed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(target, result)


if __name__ == "__main__":
    main()
