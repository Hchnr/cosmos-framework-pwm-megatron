# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Queue untraced original-capacity FP32 native, TP4 and TP2 diagnostics.

The streamed-vs-native cropped reference, tiny nested checkpoint oracle, and
CUDA backward-peak qualification must all have passed. A failed numerical TP4
observation does not prevent TP2 if all four rank reports are complete; an
infrastructure failure or any resource/source guard failure ends the sequence.
This uses two fixed real microbatches per update, not 576-slot acceptance.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import psutil

from tools import run_tp_pp_two_qualifications_coordinator as guard

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/opt/cosmos-tp-pp-work-20261119")
OUT = ROOT / "outputs/megatron_tp4_fsdp4_microbatch_20261119"
PYTHON = ROOT / ".venv/bin/python"
TP2 = Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp2_pp2_recovered_20261119")
TP4 = Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp4_pp1_20261119")
TOML = ROOT / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
ORDER = OUT / "native_parameter_order_from_meta.json"
INPUTS = OUT / "recovered_math_inputs"
LOCK = Path("/tmp/cosmos-tp-pp-norm-post-fused.lock")
MIN_HEADROOM = 100 * 1024**3
MAX_TREE_RSS = 700 * 1024**3
CASE_TIMEOUT = 12 * 3600
DIAGNOSTIC_SEED = 42


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprints(snapshot):
    files = sorted(path for base in ("tools", "tests", "cosmos_framework") for path in (snapshot / base).rglob("*.py"))
    assets = [
        TOML,
        ORDER,
        TP2 / "manifest.json",
        TP4 / "manifest.json",
        OUT / "recover_math_inputs_command.json",
        ROOT / "artifacts/megatron_tp_pp/run_fp32_and_microbatch_sequence_20261119.py",
        ROOT / "tools/run_tp_pp_two_qualifications_coordinator.py",
        ROOT / "tools/run_tp_pp_original_capacity_reference_coordinator.py",
        ROOT / "tools/run_tp_pp_untraced_capacity_math.py",
        ROOT / "tools/run_tp_pp_checkpointed_chunked_mlp_math.py",
        ROOT / "tools/build_tp_pp_capacity_telemetry.py",
        ROOT / "tools/tp_pp_capacity_gate.py",
    ]
    assets += [INPUTS / f"rank{rank}.micro0{suffix}.pt" for rank in (0, 1) for suffix in ("", ".loss")]
    return {str(path): digest(path) for path in [*files, *assets]}


def require_gates(streamed, qualifications):
    if not (
        streamed.get("streamed_native", {}).get("gate", {}).get("passed") is True
        and streamed.get("streamed_compare", {}).get("gate", {}).get("passed") is True
        and streamed.get("streamed_native", {}).get("returncode") == 0
        and streamed.get("streamed_compare", {}).get("returncode") == 0
    ):
        raise ValueError("Streamed three-update native-reference comparison has not passed")
    if qualifications.get("completed") is not True or qualifications.get("failure") is not None:
        raise ValueError("The tiny and memory qualification coordinator did not complete safely")
    for name in ("checkpointed_chunk", "nested_memory"):
        case = qualifications.get("cases", {}).get(name, {})
        if (
            case.get("passed") is not True
            or case.get("returncode") != 0
            or case.get("gate", {}).get("passed") is not True
        ):
            raise ValueError(f"Required GPU qualification did not pass: {name}")


def snapshot_copy(snapshot):
    if snapshot.exists():
        raise FileExistsError(snapshot)
    shutil.copytree(WORK, snapshot, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
    overlays = ("run_tp_pp_untraced_capacity_math.py", "run_tp_pp_checkpointed_chunked_mlp_math.py")
    for name in overlays:
        shutil.copy2(ROOT / "tools" / name, snapshot / "tools" / name)
    for base in ("tools", "tests", "cosmos_framework"):
        for source in (WORK / base).rglob("*.py"):
            if source.parent == WORK / "tools" and source.name in overlays:
                continue
            if digest(source) != digest(snapshot / source.relative_to(WORK)):
                raise ValueError(f"Snapshot source differs from isolated base: {source}")
    for name in overlays:
        if digest(ROOT / "tools" / name) != digest(snapshot / "tools" / name):
            raise ValueError(f"Snapshot overlay differs from reviewed source: {name}")

    # Instrument only the isolated math diagnostic; do not change training code.
    from tools.build_tp_pp_capacity_telemetry import apply as add_comparison_telemetry

    add_comparison_telemetry(snapshot)


def common(reference):
    return [
        "--seed",
        str(DIAGNOSTIC_SEED),
        "--objective",
        "native_rf",
        "--retain-full-text",
        "--original-capacity",
        "--attention-backend",
        "flex",
        "--deterministic",
        "--toml",
        str(TOML),
        "--prepared-dir",
        str(INPUTS),
        "--reference-dir",
        str(reference),
    ]


def command(snapshot, output, reference, mode, tp=None):
    runner = str(snapshot / "tools/run_tp_pp_untraced_capacity_math.py")
    helper = str(snapshot / "tests/megatron/tp_pp_full_source_math_cuda_compare_streamed_reference_v2.py")
    prefix = [str(PYTHON)]
    if mode == "tp":
        prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"]
    prefix += [runner, "--chunk-tokens", "4096", "--run-path", helper]
    if mode == "native":
        return prefix + [
            "--mode",
            "native",
            "--gpu-leaf-reference",
            "--preserve-saved-layout",
            "--native-device",
            "cuda",
            "--native-optimizer",
            "native_cuda",
            "--bundle",
            str(TP2),
            *common(reference),
            "--output",
            str(output),
        ]
    if tp not in (2, 4):
        raise ValueError("Only TP2 or TP4 is supported")
    return prefix + [
        "--native-norm-order",
        str(ORDER),
        "--mode",
        "tp",
        "--tp",
        str(tp),
        "--bundle",
        str(TP4 if tp == 4 else TP2),
        "--native-adam-arithmetic",
        "--activation-cpu-offload",
        "--observe-all-updates",
        *common(reference),
        "--output",
        str(output),
    ]


def reference_gate(reference, result):
    manifest = json.loads((reference / "manifest.json").read_text())
    files = list(reference.glob("update*.parameter*.pt"))
    if not (
        result["returncode"] == 0
        and manifest["passed"] is True
        and type(manifest.get("seed")) is int
        and manifest["seed"] == DIAGNOSTIC_SEED
        and manifest["original_capacity"] is True
        and manifest["gradient_streaming_candidate"] is True
        and manifest["clipping_implementation"] == "streamed_cuda_norm_and_clip_candidate"
        and manifest["gpu_leaf_reference"] is True
        and manifest["preserve_saved_tensor_dense_layout"] is True
        and manifest["native_optimizer"] == "native_cuda"
        and manifest["deterministic_algorithms"] is True
        and manifest["attention_backend"] == "flex"
        and manifest["objective"] == "native_rf"
        and len(manifest["names"]) == 811
        and len(files) == 811 * 3
        and len(manifest["input_hashes"]) == 2
        and (reference / "outputs.pt").is_file()
    ):
        raise ValueError("Incomplete or incorrect three-update original-capacity native reference")
    updates = []
    for line in Path(result["log"]).read_text().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and "native_completed_update" in row:
            updates.append(row["native_completed_update"])
    if updates != [0, 1, 2]:
        raise ValueError(f"Unexpected native completed updates: {updates}")
    return {"passed": True, "updates": updates, "parameters": 811, "parameter_files": len(files)}


def tp_gate(output, tp, result, reference):
    from tools.tp_pp_capacity_gate import tp_gate as validate_tp_gate

    return validate_tp_gate(output, tp, result, reference, TP4 if tp == 4 else TP2)


def run_case(label, args, snapshot, directory, gpu_count):
    started_unix = time.time()
    state = guard.wait_idle()
    env = os.environ.copy()
    sequence_path = ROOT / "artifacts/megatron_tp_pp/run_fp32_and_microbatch_sequence_20261119.py"
    spec = importlib.util.spec_from_file_location("_capacity_sequence_environment", sequence_path)
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
    env.update(sequence.ENV)
    env.update(
        PYTHONPATH=str(snapshot),
        CUDA_VISIBLE_DEVICES=",".join(item["uuid"] for item in state["inventory"][:gpu_count]),
        PYTHONDONTWRITEBYTECODE="1",
        COSMOS_TP_PP_NATIVE_FP32_NORM="0",
    )
    before = guard.memory_state()
    if before["conservative_reclaimable_headroom"] < MIN_HEADROOM:
        raise RuntimeError("Insufficient host/cgroup headroom for reference")
    log_path = directory / f"{label}.log"
    with log_path.open("x") as log:
        child = subprocess.Popen(
            args, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        created = psutil.Process(child.pid).create_time()
        deadline, peak = time.monotonic() + CASE_TIMEOUT, 0
        reason = None
        try:
            while child.poll() is None:
                memory = guard.memory_state()
                if memory["conservative_reclaimable_headroom"] < MIN_HEADROOM:
                    reason = "conservative memory headroom guard"
                    break
                if any(memory["events"].get(key, 0) > before["events"].get(key, 0) for key in ("oom", "oom_kill")):
                    reason = "cgroup OOM event"
                    break
                try:
                    parent = psutil.Process(child.pid)
                    peak = max(
                        peak,
                        sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running()),
                    )
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                if peak > MAX_TREE_RSS or time.monotonic() > deadline:
                    reason = "process RSS or case timeout guard"
                    break
                time.sleep(5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        code = child.wait()
    after = guard.wait_idle()
    memory_after = guard.memory_state()
    result = {
        "command": args,
        "pid": child.pid,
        "create_time": created,
        "returncode": code,
        "stop_reason": reason,
        "log": str(log_path),
        "gpu_before": state,
        "gpu_after": after,
        "memory_before": before,
        "memory_after": memory_after,
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "peak_process_tree_rss": peak,
    }
    if reason:
        raise RuntimeError(f"{label} stopped: {reason}")
    if any(memory_after["events"].get(key, 0) > before["events"].get(key, 0) for key in ("oom", "oom_kill")):
        raise RuntimeError(f"{label}: cgroup OOM event counter increased")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualifier-pid", type=int, required=True)
    parser.add_argument("--qualifier-output", type=Path, required=True)
    parser.add_argument("--streamed-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    for field in ("qualifier_output", "streamed_output", "snapshot", "output", "reference"):
        setattr(args, field, getattr(args, field).resolve())
    predecessor = guard.identity(args.qualifier_pid, "run_tp_pp_two_qualifications_coordinator.py")
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "coordinator_result.json"
    result = {
        "completed": False,
        "passed": False,
        "failure": None,
        "predecessor": predecessor,
        "scope": "two original-capacity real microbatches per update; not 576-slot production acceptance",
        "gpu_isolation_limitation": "cooperative lock and idle checks; container and host PID namespaces differ",
        "snapshot": str(args.snapshot),
        "reference": str(args.reference),
        "cases": {},
    }
    guard.save(result_path, result)
    observed = {}
    try:
        while guard.alive(predecessor) or any(guard.alive(item) for item in observed.values()):
            guard.record_children(predecessor, observed)
            for child in list(observed.values()):
                guard.record_children(child, observed)
            time.sleep(3)
        with LOCK.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["lock_acquired_unix"] = time.time()
            result["predecessor_children_seen"] = list(observed.values())
            require_gates(json.loads(args.streamed_output.read_text()), json.loads(args.qualifier_output.read_text()))
            if args.reference.exists():
                raise FileExistsError(args.reference)
            snapshot_copy(args.snapshot)
            before = fingerprints(args.snapshot)
            result["source_and_input_sha256_before"] = before
            guard.save(result_path, result)
            for label, mode, tp in (("native", "native", None), ("tp4", "tp", 4), ("tp2", "tp", 2)):
                output = args.output / f"{label}.json"
                case = run_case(
                    label,
                    command(args.snapshot, output, args.reference, mode, tp),
                    args.snapshot,
                    args.output,
                    1 if mode == "native" else 4,
                )
                result["cases"][label] = case
                after = fingerprints(args.snapshot)
                if after != before:
                    raise RuntimeError("Snapshot, canonical metadata or real input changed during diagnostics")
                case["source_and_input_sha256_unchanged"] = True
                case["gate"] = (
                    reference_gate(args.reference, case)
                    if mode == "native"
                    else tp_gate(output, tp, case, args.reference)
                )
                guard.save(result_path, result)
            result["completed"] = True
            result["source_and_input_sha256_after"] = after
            result["passed"] = all(result["cases"][label]["gate"]["passed"] for label in ("native", "tp4", "tp2"))
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        guard.save(result_path, result)


if __name__ == "__main__":
    main()
