# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Queue a clean VAE-inclusive TP4/FSDP4 comparison after attention diagnosis.

This script is inert until explicitly launched. It uses a cooperative lock and
polls compute users; an uncooperative process could still run between polls,
so the report describes observed exclusivity rather than a hardware guarantee.
Outputs are created in a new directory and are never overwritten.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

if __package__:
    from . import tp_pp_gpu_process_monitor as gpu_monitor
else:
    # Direct `python tools/...py` adds tools/, not the repository root, to
    # sys.path. Import the sibling directly without requiring PYTHONPATH.
    import tp_pp_gpu_process_monitor as gpu_monitor

LOCK = Path("/tmp/cosmos-tp-pp-norm-post-fused.lock")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import coordinator helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live(identity):
    try:
        process = psutil.Process(identity["pid"])
        return abs(process.create_time() - identity["create_time"]) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def matching_completion(events_path, predecessor):
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # A writer may be in the middle of appending an event.
            continue
        terminal = event.get("attention_isolation_finished") or "attention_isolation_native_failed" in event
        if terminal and event.get("time", 0) >= predecessor["create_time"]:
            return event
    return None


def query_gpu_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    inventory = {}
    for line in result.stdout.splitlines():
        index, uuid = (field.strip() for field in line.split(","))
        inventory[int(index)] = uuid
    if sorted(inventory) != [0, 1, 2, 3] or len(set(inventory.values())) != 4:
        raise ValueError(f"Exactly four physical GPUs are required: {inventory}")
    return inventory


def query_gpu_users(inventory):
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    known = set(inventory.values())
    users = []
    for line in result.stdout.splitlines():
        uuid, pid = (field.strip() for field in line.split(","))
        if uuid not in known:
            raise ValueError(f"Unknown GPU UUID in compute process list: {uuid}")
        users.append({"gpu_uuid": uuid, "pid": int(pid)})
    return users


def save(path, data):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def stop_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def check_predecessor(identity_path, predecessor):
    if json.loads(identity_path.read_text()) != predecessor or live(predecessor):
        raise RuntimeError("Attention coordinator identity changed or resumed")


def run_benchmark(
    sequence, backend, directory, fixture, inventory, result, status_path, poll_seconds, check_dependency, monitor
):
    bundle = sequence.TP4 if backend == "tp" else sequence.TP2
    command = sequence.distributed(
        "tools/run_tp4_fsdp4_raw_microbatch_benchmark.py",
        "--raw-fixture",
        fixture,
        "--backend",
        backend,
        "--bundle",
        bundle,
        "--prepared-spool",
        sequence.SPOOL,
        "--output",
        directory / backend,
        "--toml",
        sequence.TOML,
        "--updates",
        7,
        "--warmup-updates",
        2,
    )
    env = sequence.ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(inventory[index] for index in range(4))
    phase = {"command": command, "log": str(directory / f"{backend}.log"), "polls": 0}
    result["runs"][backend] = phase
    save(status_path, result)
    with (directory / f"{backend}.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=sequence.WORK, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        phase["pid"] = process.pid
        phase["create_time"] = psutil.Process(process.pid).create_time()
        save(status_path, result)
        try:
            while process.poll() is None:
                check_dependency()
                users = query_gpu_users(inventory)
                phase["polls"] += 1
                snapshot = {"time": time.time(), "users": users}
                try:
                    snapshot["evidence"] = monitor.check(users, process.pid)
                except BaseException as error:
                    snapshot["failure"] = f"{type(error).__name__}: {error}"
                    phase["failure_snapshot"] = snapshot
                    raise
                finally:
                    with (directory / f"{backend}_gpu_observations.jsonl").open("a") as audit:
                        audit.write(json.dumps(snapshot) + "\n")
                try:
                    root = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)] if p.is_running())
                    if rss > 780 * 1024**3:
                        raise RuntimeError(f"Host-memory guard exceeded during {backend}: {rss}")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(poll_seconds)
            phase["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"{backend} benchmark failed with exit code {process.returncode}")
            if not phase["polls"]:
                raise RuntimeError("No GPU-user monitoring snapshots were collected")
            monitor.finish()
        finally:
            stop_process_group(process)
            phase["returncode"] = process.returncode
            save(status_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-helper", type=Path, required=True)
    parser.add_argument("--attention-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600)
    parser.add_argument("--pid-monitor-mode", choices=("strict", "restricted"), default="strict")
    parser.add_argument("--host-proc", type=Path)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("Poll interval and timeout must be positive")

    def interrupt(signum, _frame):
        raise SystemExit(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "coordinator_result.json"
    result = {
        "completed": False,
        "failure": None,
        "runs": {},
        "exclusivity": "cooperative lock and periodic compute-user polling",
        "pid_monitor_mode": args.pid_monitor_mode,
        "provisional": args.pid_monitor_mode == "restricted",
        "monitor_limitations": (
            "Restricted mode cannot bind host NVML PIDs to local FD owners or exclude transient interference/PID replacement between polls"
            if args.pid_monitor_mode == "restricted"
            else "External interference between polling snapshots cannot be excluded"
        ),
    }
    save(status, result)
    try:
        host_proc = gpu_monitor.validate_host_proc(args.host_proc) if args.pid_monitor_mode == "strict" else None
        sequence = load_module(args.sequence_helper, "_raw_clean_sequence")
        predecessor = json.loads(args.attention_identity.read_text())
        result["predecessor"] = predecessor
        deadline = time.monotonic() + args.timeout_seconds
        events_path = sequence.OUTPUT / "sequence_events.jsonl"
        while True:
            if json.loads(args.attention_identity.read_text()) != predecessor:
                raise RuntimeError("Attention coordinator identity changed; refusing stale dependency")
            event = matching_completion(events_path, predecessor)
            if not live(predecessor):
                if event is None:
                    raise RuntimeError("Attention coordinator exited without its completion event")
                # Diagnostic failure is still a terminal dependency: it must
                # not silently cancel an independent performance comparison.
                result["attention_terminal_event"] = event
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Attention coordinator did not finish")
            time.sleep(args.poll_seconds)
        fixture = sequence.OUTPUT / "raw_microbatch_fixture"
        manifest = json.loads((fixture / "manifest.json").read_text())
        if not manifest["passed"] or [row["samples"] for row in manifest["records"]] != [9, 11, 13]:
            raise ValueError("Raw fixture failed validation")
        result["raw_fixture"] = str(fixture)
        result["fixture_records"] = manifest["records"]
        with LOCK.open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for cooperative GPU lock")
                    time.sleep(args.poll_seconds)
            inventory = query_gpu_inventory()
            result["gpu_inventory"] = inventory
            devices = gpu_monitor.query_device_nodes(inventory) if args.pid_monitor_mode == "restricted" else None
            while query_gpu_users(inventory):
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPUs remained occupied")
                time.sleep(args.poll_seconds)
            save(status, result)
            for backend in ("tp", "fsdp"):
                check_predecessor(args.attention_identity, predecessor)
                if query_gpu_users(inventory):
                    raise RuntimeError(f"GPU was occupied before {backend} benchmark")
                monitor = (
                    gpu_monitor.StrictMonitor(inventory, host_proc)
                    if args.pid_monitor_mode == "strict"
                    else gpu_monitor.RestrictedMonitor(inventory, devices)
                )
                run_benchmark(
                    sequence,
                    backend,
                    args.output,
                    fixture,
                    inventory,
                    result,
                    status,
                    args.poll_seconds,
                    lambda: check_predecessor(args.attention_identity, predecessor),
                    monitor,
                )
                # Let CUDA processes exit completely before switching backends.
                while query_gpu_users(inventory):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"GPU was not released after {backend}")
                    time.sleep(args.poll_seconds)
            compare = load_module(
                sequence.WORK / "tools/compare_tp4_fsdp4_microbatch_benchmark.py", "_raw_clean_compare"
            )
            comparison = compare.compare(args.output / "tp", args.output / "fsdp")
            comparison["exclusivity_observation"] = result["exclusivity"]
            comparison["pid_monitor_mode"] = args.pid_monitor_mode
            comparison["provisional"] = result["provisional"]
            comparison["monitor_limitations"] = result["monitor_limitations"]
            comparison["performance_acceptance"] = False
            (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        result["completed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(status, result)


if __name__ == "__main__":
    main()
