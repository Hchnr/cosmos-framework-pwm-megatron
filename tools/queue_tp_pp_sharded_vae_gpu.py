# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed cooperative queue for a later isolated VAE GPU diagnostic.

'snapshot' records an already running predecessor and observed descendants.
'queue' defaults to a dry run. '--execute' explicitly enables command launch;
the command is never a claim of model correctness or of GPU exclusivity.
Only cooperative users of the same lock are serialized. NVML host PIDs can
belong to another PID namespace and polling cannot exclude transient users.
"""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("/tmp/cosmos-tp-pp-norm-post-fused.lock")
PREDECESSOR_FRAGMENT = "run_tp_pp_two_qualifications_coordinator.py"
DEFAULT_SOURCES = (
    ROOT / "tools/qualify_tp_pp_sharded_vae_gpu.py",
    ROOT / "tools/tp_pp_sharded_vae_experiment.py",
    ROOT / "tools/queue_tp_pp_sharded_vae_gpu.py",
)


def boot_identity():
    return {"hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def source_fingerprints(paths):
    """Record resolved identity/content; detect a file replaced while hashing."""
    result = {}
    for candidate in paths:
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        label = str(path)
        if label in result:
            continue
        target = path.resolve(strict=True)
        before = target.stat()
        if not target.is_file():
            raise ValueError(f"Source is not a regular file: {path}")
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = target.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity or path.resolve(strict=True) != target:
            raise RuntimeError(f"Source changed while hashing: {path}")
        result[label] = {"resolved": str(target), "sha256": digest}
    return result


def process_identity(pid):
    process = psutil.Process(pid)
    return {"pid": process.pid, "create_time": process.create_time(), "command": process.cmdline()}


def alive(identity):
    try:
        process = psutil.Process(identity["pid"])
        return abs(process.create_time() - identity["create_time"]) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def descendants(identity):
    if not alive(identity):
        return []
    try:
        result = []
        for child in psutil.Process(identity["pid"]).children(recursive=True):
            try:
                result.append(process_identity(child.pid))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return result
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return []


def snapshot(pid, command_fragment=PREDECESSOR_FRAGMENT):
    item = process_identity(pid)
    if not alive(item) or command_fragment not in " ".join(item["command"]):
        raise ValueError("Predecessor is not the expected live coordinator")
    children = descendants(item)
    if not alive(item):
        raise RuntimeError("Predecessor exited during identity capture")
    return {
        "version": 1,
        "captured_at": time.time(),
        "node": boot_identity(),
        "command_fragment": command_fragment,
        "predecessor": item,
        "observed_descendants": children,
        "observation_limit": "Only process identities visible during snapshots can be tracked.",
    }


def validate_snapshot(data, expected_pid):
    if data.get("version") != 1 or data.get("node") != boot_identity():
        raise ValueError("Predecessor node/boot identity is missing or changed")
    parent = data.get("predecessor", {})
    fragment = data.get("command_fragment", PREDECESSOR_FRAGMENT)
    if (
        not isinstance(fragment, str)
        or not fragment
        or parent.get("pid") != expected_pid
        or fragment not in " ".join(parent.get("command", []))
    ):
        raise ValueError("Predecessor identity does not match the expected coordinator")
    children = data.get("observed_descendants")
    if not isinstance(children, list):
        raise ValueError("Predecessor descendants were not captured")
    for item in [parent, *children]:
        if not isinstance(item.get("pid"), int) or not isinstance(item.get("create_time"), (float, int)):
            raise ValueError("Incomplete predecessor/descendant identity")
    # Starting after parent exit loses the opportunity to observe descendants
    # spawned since capture. Never infer their absence from the original list.
    if not alive(parent):
        raise RuntimeError("Predecessor no longer alive at queue startup; descendant continuity is unknown")


def await_predecessors(snapshot_data, *, timeout, poll, status=None):
    snapshots = snapshot_data if isinstance(snapshot_data, list) else [snapshot_data]
    if not snapshots:
        raise ValueError("At least one predecessor snapshot is required")
    start_node = snapshots[0]["node"]
    if any(item["node"] != start_node for item in snapshots):
        raise ValueError("Predecessors were not captured on the same node and boot")
    initial = [identity for item in snapshots for identity in (item["predecessor"], *item["observed_descendants"])]
    observed = {(item["pid"], item["create_time"]): item for item in initial}
    deadline = time.monotonic() + timeout
    while True:
        if boot_identity() != start_node:
            raise RuntimeError("Node or boot identity changed while waiting")
        for item in tuple(observed.values()):
            for child in descendants(item):
                observed[(child["pid"], child["create_time"])] = child
        remaining = [item for item in observed.values() if alive(item)]
        if status:
            status(list(observed.values()), remaining)
        if not remaining:
            return list(observed.values())
        if time.monotonic() >= deadline:
            raise TimeoutError("Predecessor or an observed descendant remains alive")
        time.sleep(poll)


def gpu_state():
    def query(arguments):
        command = ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"]
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()

    inventory = []
    for line in query(["--query-gpu=index,uuid,memory.used,utilization.gpu"]):
        index, uuid, memory, utilization = (part.strip() for part in line.split(","))
        inventory.append(
            {"index": int(index), "uuid": uuid, "memory_used_mib": int(memory), "utilization": int(utilization)}
        )
    inventory.sort(key=lambda item: item["index"])
    if [row["index"] for row in inventory] != [0, 1, 2, 3] or len({row["uuid"] for row in inventory}) != 4:
        raise ValueError("Expected exactly four distinct GPU UUIDs")
    known = {row["uuid"] for row in inventory}
    users = []
    for line in query(["--query-compute-apps=gpu_uuid,pid"]):
        uuid, pid = (part.strip() for part in line.split(","))
        if uuid not in known:
            raise ValueError(f"Unrecognized GPU UUID {uuid}")
        users.append({"uuid": uuid, "host_pid": int(pid)})
    return {"time": time.time(), "inventory": inventory, "compute_users": users}


def idle(state):
    return not state["compute_users"] and all(device["utilization"] == 0 for device in state["inventory"])


def await_idle(*, timeout, poll, consecutive=2):
    deadline = time.monotonic() + timeout
    last_inventory, samples = None, []
    while True:
        state = gpu_state()
        inventory = tuple((row["index"], row["uuid"]) for row in state["inventory"])
        if last_inventory is not None and last_inventory != inventory:
            raise RuntimeError("GPU inventory changed during idle checks")
        last_inventory = inventory
        samples = [*samples, state] if idle(state) else []
        if len(samples) >= consecutive:
            return samples
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs did not become idle: {state}")
        time.sleep(poll)


def acquire(lock, *, timeout, poll):
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Could not acquire the cooperative GPU lock") from None
            time.sleep(poll)


def await_launched_tree(process, root, *, poll, status=None):
    """Keep the lock while the child or any observed descendant is alive."""
    observed = {(root["pid"], root["create_time"]): root}
    while True:
        for item in tuple(observed.values()):
            for child in descendants(item):
                observed[(child["pid"], child["create_time"])] = child
        remaining = [item for item in observed.values() if alive(item)]
        code = process.poll()
        if status:
            status(list(observed.values()), remaining)
        if code is not None and not remaining:
            return code
        time.sleep(poll)


def save(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    capture = actions.add_parser("snapshot", help="Capture a live predecessor identity; does not inspect GPUs")
    capture.add_argument("--pid", type=int, required=True)
    capture.add_argument("--command-fragment", default=PREDECESSOR_FRAGMENT)
    capture.add_argument("--output", type=Path, required=True)
    queue = actions.add_parser("queue", help="Prepare a queue entry; --execute is required to launch")
    queue.add_argument("--identity", type=Path, action="append", required=True)
    queue.add_argument("--expected-pid", type=int, action="append", required=True)
    queue.add_argument("--output", type=Path, required=True)
    queue.add_argument("--timeout", type=float, default=14400)
    queue.add_argument("--poll", type=float, default=2)
    queue.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="PATH",
        help="Also pin a source file by SHA-256 (repeatable); core VAE sources are always pinned",
    )
    queue.add_argument(
        "--require-env",
        action="append",
        default=[],
        metavar="NAME",
        help="Require an inherited environment variable; record its SHA-256 without logging its value (repeatable)",
    )
    queue.add_argument("--execute", action="store_true")
    queue.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.action == "snapshot":
        data = snapshot(args.pid, args.command_fragment)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        return
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.timeout <= 0 or not 0 < args.poll <= 60:
        parser.error("A command, positive timeout, and poll interval in (0, 60] are required")
    if len(args.identity) != len(args.expected_pid) or len(set(args.expected_pid)) != len(args.expected_pid):
        parser.error("Specify one unique --expected-pid per --identity, in the same order")
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "queue_result.json"
    result = {
        "completed": False,
        "mode": "execute" if args.execute else "dry_run",
        "command": command,
        "failure": None,
        "predecessor_gate": "process-tree termination only; predecessor success or qualification is not established",
        "gpu_exclusivity": "provisional: cooperative lock, two idle polls; transient or foreign users remain possible",
    }
    save(result_path, result)
    try:
        snapshots = [json.loads(path.read_text()) for path in args.identity]
        result["predecessor_snapshots"] = snapshots
        for item, pid in zip(snapshots, args.expected_pid, strict=True):
            validate_snapshot(item, pid)
        if not all(alive(item["predecessor"]) for item in snapshots):
            raise RuntimeError("A predecessor exited before all process trees could be observed")
        data = snapshots[0]
        source_paths = (*DEFAULT_SOURCES, *args.source)
        result["source_sha256_on_queue"] = source_fingerprints(source_paths)
        result["required_environment"] = {}
        for name in args.require_env:
            if not name.isidentifier() or name not in os.environ or not os.environ[name]:
                raise ValueError(f"Required inherited environment variable is absent or invalid: {name}")
            result["required_environment"][name] = {
                "sha256": hashlib.sha256(os.environ[name].encode()).hexdigest(),
                "source": "inherited unchanged",
            }
        if not args.execute:
            result["completed"] = True
            return

        def update_wait(observed, remaining):
            result.update(observed_predecessor_tree=observed, live_predecessor_tree=remaining)
            save(result_path, result)

        await_predecessors(snapshots, timeout=args.timeout, poll=args.poll, status=update_wait)
        with LOCK.open("a+") as lock:
            acquire(lock, timeout=args.timeout, poll=args.poll)
            if boot_identity() != data["node"]:
                raise RuntimeError("Node or boot changed after predecessor completion")
            if any(alive(item) for item in result["observed_predecessor_tree"]):
                raise RuntimeError("A predecessor identity became live after lock acquisition")
            result["gpu_before"] = await_idle(timeout=args.timeout, poll=args.poll)
            result["source_sha256_before_launch"] = source_fingerprints(source_paths)
            if result["source_sha256_before_launch"] != result["source_sha256_on_queue"]:
                raise RuntimeError("A pinned qualification source changed while waiting; refusing to launch")
            environment = os.environ.copy()
            overrides = {
                "CUDA_VISIBLE_DEVICES": ",".join(row["uuid"] for row in result["gpu_before"][-1]["inventory"]),
                "LD_LIBRARY_PATH": "",
                "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            if set(args.require_env) & set(overrides):
                raise ValueError("A required inherited variable conflicts with a queue execution override")
            environment.update(overrides)
            result["execution_environment_overrides"] = overrides
            save(result_path, result)
            with (args.output / "qualification.log").open("x") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                result["launched"] = process_identity(process.pid)
                save(result_path, result)

                def update_child_tree(observed, remaining):
                    result.update(observed_child_tree=observed, live_child_tree=remaining)
                    save(result_path, result)

                # No signals are sent to any process. Keep the lock until our
                # command and observed descendants have terminated, even if
                # this coordinator is interrupted.
                try:
                    result["returncode"] = await_launched_tree(
                        process, result["launched"], poll=args.poll, status=update_child_tree
                    )
                except BaseException:
                    while True:
                        try:
                            await_launched_tree(process, result["launched"], poll=args.poll, status=update_child_tree)
                            break
                        except KeyboardInterrupt:
                            continue
                    raise
            if result["returncode"] != 0:
                raise RuntimeError(f"Qualification exited {result['returncode']}")
            result["gpu_after"] = await_idle(timeout=args.timeout, poll=args.poll)
            result["source_sha256_after"] = source_fingerprints(source_paths)
            if result["source_sha256_after"] != result["source_sha256_on_queue"]:
                raise RuntimeError("A pinned qualification source changed during the run")
            result["completed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(result_path, result)


if __name__ == "__main__":
    main()
