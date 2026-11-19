# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Conditionally queue the fixed-native-state TP4 K/V diagnostic.

Only a completed, numerically failing three-update K/V intervention permits
launch. A passing intervention, incomplete report or runtime failure skips
this optional diagnosis. This does not claim continuous-trajectory acceptance.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tools import queue_tp_pp_sharded_vae_gpu as queue
from tools import run_tp_pp_two_qualifications_coordinator as safety

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/megatron_tp4_fsdp4_microbatch_20261119"
INTERVENTION = BASE / "full_vjp_real36_20261119T100656_574178"
PLAN = BASE / "replay_native_state_tp4_diagnostic_plan_20261119.json"
LOCK = Path("/tmp/cosmos-tp-pp-norm-post-fused.lock")
SEQUENCE = ROOT / "artifacts/megatron_tp_pp/run_fp32_and_microbatch_sequence_20261119.py"
SNAPSHOT = Path("/opt/cosmos-tp-pp-kv-qualification-20261119")
EXPECTED_FAILURE = "SystemExit: Full-source three-update observation completed with FAILED fixed gates"


def intervention_decision(directory):
    """Require complete failed math evidence; skip success, errors and gaps."""
    try:
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("passed") is True:
            return False, "Continuous K/V diagnostic passed; replay is unnecessary"
        if summary.get("error") or summary.get("tp4", {}).get("returncode") != 1:
            return False, "K/V intervention did not finish with an expected numerical failure"
        if not summary.get("source_sha256_after") or summary["source_sha256_before"] != summary["source_sha256_after"]:
            return False, "K/V intervention source fingerprints are absent or changed"
        # The 36-layer coordinator verifies its local pre-run snapshot against
        # snapshot_after and raises on mismatch, but publishes only the latter.
        if not summary.get("snapshot_after"):
            return False, "K/V intervention snapshot verification is absent"
        rows = summary.get("ranks", [])
        if sorted(row["rank"] for row in rows) != list(range(4)):
            return False, "K/V intervention lacks four rank summaries"
        if not any(row["gate_failure_count"] > 0 for row in rows):
            return False, "K/V intervention has no numerical gate failure"
        for rank in range(4):
            report = json.loads((directory / f"tp4.rank{rank}.json").read_text())
            sidecar = json.loads((directory / "reports" / f"fp32_reconstructed_kv.rank{rank}.json").read_text())
            if not (
                report.get("updates") == 3
                and report.get("tensor_parallel_size") == 4
                and report.get("pipeline_parallel_size") == 1
                and report.get("continuous_trajectory") is True
                and report.get("replay_native_state") is False
                and report.get("passed") is False
                and len(report.get("gradient_summary", [])) == 3
                and sidecar.get("full_width_vjp") is True
                and sidecar.get("failure") == EXPECTED_FAILURE
                and len(sidecar.get("modules", {})) == 216
                and all(item.get("forward_calls", 0) > 0 for item in sidecar["modules"].values())
            ):
                return False, f"Rank {rank} lacks complete 36-layer K/V failure evidence"
        return True, "Completed three-update K/V diagnostic failed the fixed numerical gates"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"K/V intervention is incomplete or invalid: {type(error).__name__}: {error}"


def fingerprints(plan):
    paths = [
        Path(__file__),
        PLAN,
        SEQUENCE,
        ROOT / "tools/run_tp_pp_replay_native_state_witness.py",
        ROOT / "tools/queue_tp_pp_sharded_vae_gpu.py",
        ROOT / "tools/run_tp_pp_two_qualifications_coordinator.py",
        Path("/opt/cosmos-tp-pp-full-vjp-20261119/full_vjp.py"),
        Path("/opt/cosmos-tp-pp-full-vjp-20261119/run_full_model_vjp.py"),
        ROOT / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml",
        BASE / "native_parameter_order_from_meta.json",
        Path("/opt/cosmos-tp-pp-attention-native-fp32-reference-20261119/manifest.json"),
        Path("/opt/cosmos-tp-pp-attention-native-fp32-reference-20261119/outputs.pt"),
        Path("/share-mixed/project/eai_pwm/cache/megatron_tp_pp/canonical_iter1200_tp4_pp1_20261119/manifest.json"),
        *sorted((BASE / "recovered_math_inputs").glob("rank*.pt")),
        *sorted(SNAPSHOT.rglob("*.py")),
    ]
    if not plan.get("command") or not paths[-1].is_file():
        raise ValueError("Replay command or isolated snapshot is unavailable")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def verify_results(directory):
    from tools.run_tp_pp_replay_native_state_witness import read_report

    ranks = []
    for rank in range(4):
        report = read_report(directory / f"tp4.rank{rank}.json")
        witnesses = [
            json.loads((directory / "witness" / f"replay_state.update{u}.rank{rank}.json").read_text()) for u in (1, 2)
        ]
        sidecar = json.loads((directory / "reports" / f"fp32_reconstructed_kv.rank{rank}.json").read_text())
        if not all(w.get("master_moments_step_compute_exact") and w.get("verified_shards") == 811 for w in witnesses):
            raise ValueError(f"Rank {rank} did not witness all 811 replayed parameters")
        if not (
            sidecar.get("completed")
            and sidecar.get("full_width_vjp")
            and sidecar.get("failure") is None
            and len(sidecar.get("modules", {})) == 216
            and all(value.get("forward_calls", 0) > 0 for value in sidecar["modules"].values())
        ):
            raise ValueError(f"Rank {rank} did not exercise complete full-width K/V")
        ranks.append(
            {
                "rank": rank,
                "gradient_summary": report["gradient_summary"],
                "pointwise_gates_passed": report["pointwise_gates_passed"],
                "gate_failure_count": len(report["gate_failures"]),
            }
        )
    return ranks


def environment(plan, devices):
    spec = importlib.util.spec_from_file_location("_replay_sequence", SEQUENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.ENV.copy()
    env.update(plan["env_overrides"])
    env["CUDA_VISIBLE_DEVICES"] = ",".join(device["uuid"] for device in devices)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--case-timeout", type=int, default=14400)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    before = fingerprints(plan)
    # Capture process trees after the potentially slow snapshot hashing so no
    # unobserved process can be introduced during that read window.
    snapshots = [
        queue.snapshot(pid, fragment)
        for pid, fragment in (
            (574178, "run_36_after_vjp.py"),
            (585516, "run_tp_pp_two_qualifications_coordinator.py"),
            (599884, "run_tp_pp_original_capacity_reference_coordinator.py"),
            (609054, "queue_tp_pp_sharded_vae_gpu.py"),
        )
    ]
    for item in snapshots:
        queue.validate_snapshot(item, item["predecessor"]["pid"])
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "queue_result.json"
    result = {
        "completed": False,
        "launched": False,
        "skipped": None,
        "failure": None,
        "scope": "fixed-native-state diagnostic; not continuous training acceptance",
        "predecessors": snapshots,
        "queued_at": time.time(),
        "gpu_isolation_limitation": "Host and container GPU PID namespaces differ",
        "source_sha256": before,
    }
    queue.save(status, result)
    try:
        result["observed_predecessor_tree"] = queue.await_predecessors(snapshots, timeout=args.timeout, poll=3)
        run, reason = intervention_decision(INTERVENTION)
        result["decision"] = reason
        if not run:
            result["skipped"] = reason
            result["completed"] = True
            return
        with LOCK.open("a+") as lock:
            queue.acquire(lock, timeout=args.timeout, poll=3)
            result["lock_acquired_at"] = time.time()
            result["gpu_before"] = queue.await_idle(timeout=args.timeout, poll=3)
            if fingerprints(plan) != before:
                raise RuntimeError("Replay source, reference metadata or input fingerprints changed")
            case_dir = Path(plan["env_overrides"]["FULL_VJP_REPORT_DIR"]).parent
            if case_dir.exists():
                raise FileExistsError(case_dir)
            case_dir.mkdir(parents=True, exist_ok=False)
            memory_before = safety.memory_state()
            if memory_before["conservative_reclaimable_headroom"] < safety.MIN_HEADROOM:
                raise RuntimeError("Insufficient cgroup/host headroom before replay")
            env = environment(plan, result["gpu_before"][-1]["inventory"])
            with (args.output / "replay.log").open("x") as log:
                process = subprocess.Popen(
                    plan["command"],
                    cwd=plan["cwd"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                identity = queue.process_identity(process.pid)
                result.update(launched=True, child=identity, started_at=time.time(), memory_before=memory_before)
                queue.save(status, result)
                deadline = time.monotonic() + args.case_timeout
                stop_reason = None
                observed = {(identity["pid"], identity["create_time"]): identity}
                try:
                    while process.poll() is None:
                        for item in tuple(observed.values()):
                            for child in queue.descendants(item):
                                observed[(child["pid"], child["create_time"])] = child
                        memory = safety.memory_state()
                        if memory["conservative_reclaimable_headroom"] < safety.MIN_HEADROOM:
                            stop_reason = "cgroup/host headroom below threshold"
                            break
                        if any(
                            memory["events"].get(k, 0) > memory_before["events"].get(k, 0) for k in ("oom", "oom_kill")
                        ):
                            stop_reason = "cgroup OOM event increased"
                            break
                        if time.monotonic() > deadline:
                            stop_reason = "replay timed out"
                            break
                        time.sleep(3)
                finally:
                    if stop_reason and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                    # Keep the lock while any observed child remains alive,
                    # including if this coordinator itself is interrupted.
                    while process.poll() is None or any(queue.alive(item) for item in observed.values()):
                        try:
                            for item in tuple(observed.values()):
                                for child in queue.descendants(item):
                                    observed[(child["pid"], child["create_time"])] = child
                            time.sleep(3)
                        except KeyboardInterrupt:
                            continue
                    result["returncode"] = process.wait()
                    result["observed_child_tree"] = list(observed.values())
                result.update(ended_at=time.time(), stop_reason=stop_reason, memory_after=safety.memory_state())
            result["gpu_after"] = queue.await_idle(timeout=args.timeout, poll=3)
            if stop_reason or result["returncode"] != 0:
                raise RuntimeError(f"Replay did not complete normally: {stop_reason or result['returncode']}")
            if fingerprints(plan) != before:
                raise RuntimeError("Replay source, reference metadata or input fingerprints changed during run")
            result["ranks"] = verify_results(case_dir)
            result["completed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        queue.save(status, result)


if __name__ == "__main__":
    main()
