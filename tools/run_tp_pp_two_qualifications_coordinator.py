# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Queue three isolated GPU qualifications after the two correctness coordinators.

The lock is cooperative. NVML host PIDs may not match this container's PID
namespace; idle-before/after checks do not establish exclusive ownership while
running. A failed predecessor is not proof of either numerical qualification.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("/tmp/cosmos-tp-pp-norm-post-fused.lock")
MIN_HEADROOM = 24 * 1024**3
SOURCES = (
    "tools/run_tp_pp_two_qualifications_coordinator.py",
    "tools/qualify_tp_pp_adaptive_padding_gpu.py",
    "tools/run_tp_pp_adaptive_padding.py",
    "tools/tp_pp_adaptive_qualification_data.py",
    "tools/qualify_tp_pp_checkpointed_chunked_mlp_math.py",
    "tools/qualify_tp_pp_nested_checkpoint_memory.py",
    "tools/run_tp_pp_checkpointed_chunked_mlp_math.py",
    "tools/qualify_tp_pp_flat_checkpoint_math.py",
    "tools/run_tp_pp_flat_checkpoint_math.py",
    "tools/run_tp_pp_chunked_mlp_math.py",
    "tests/megatron/tp_pp_network_smoke.py",
    "tests/megatron/tp_pp_network_io_test.py",
    "cosmos_framework/model/generator/mot/small_action_tower.py",
    "cosmos_framework/model/generator/mot/unified_mot.py",
    "cosmos_framework/model/generator/megatron_tp_pp/io.py",
    "cosmos_framework/model/generator/megatron_tp_pp/stage.py",
    "cosmos_framework/training_backends/megatron_tp_pp/driver.py",
    "cosmos_framework/training_backends/megatron_tp_pp/optimizer.py",
    "cosmos_framework/utils/generator/fused_adam.py",
)


def identity(pid, command_fragment):
    process = psutil.Process(pid)
    command = process.cmdline()
    if command_fragment not in " ".join(command) or process.status() == psutil.STATUS_ZOMBIE:
        raise ValueError(f"Wrong or terminated predecessor {pid}: {command}")
    return {"pid": pid, "create_time": process.create_time(), "command": command}


def alive(item):
    try:
        process = psutil.Process(item["pid"])
        return abs(process.create_time() - item["create_time"]) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def record_children(item, observed):
    if not alive(item):
        return
    try:
        for process in psutil.Process(item["pid"]).children(recursive=True):
            key = (process.pid, process.create_time())
            observed[key] = {"pid": key[0], "create_time": key[1]}
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return


def sources():
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return {"head": head, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "files_sha256": hashes}


def gpu_state():
    def query(args):
        return subprocess.run(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.splitlines()

    inventory = []
    for line in query(["--query-gpu=index,uuid,memory.used,utilization.gpu"]):
        index, uuid, memory, utilization = (part.strip() for part in line.split(","))
        inventory.append(
            {"index": int(index), "uuid": uuid, "memory_used_mib": int(memory), "utilization_percent": int(utilization)}
        )
    inventory.sort(key=lambda item: item["index"])
    if [item["index"] for item in inventory] != [0, 1, 2, 3] or len({item["uuid"] for item in inventory}) != 4:
        raise ValueError(f"Expected exactly four distinct GPUs: {inventory}")
    uuids = {item["uuid"] for item in inventory}
    users = []
    for line in query(["--query-compute-apps=gpu_uuid,pid"]):
        uuid, pid = (part.strip() for part in line.split(","))
        if uuid not in uuids:
            raise ValueError(f"Unknown GPU UUID: {uuid}")
        users.append({"uuid": uuid, "host_pid": int(pid)})
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "inventory": inventory, "compute_processes": users}


def wait_idle(timeout=3600):
    deadline = time.monotonic() + timeout
    while True:
        state = gpu_state()
        if not state["compute_processes"]:
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs remain busy: {state}")
        time.sleep(3)


def memory_state():
    """Use conservative reclaimable inactive cache, never all page cache."""
    base = Path("/sys/fs/cgroup")
    current = int((base / "memory.current").read_text())
    max_text = (base / "memory.max").read_text().strip()
    limit = None if max_text == "max" else int(max_text)
    stats = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
    events = {
        key: int(value) for key, value in (line.split() for line in (base / "memory.events").read_text().splitlines())
    }
    selected = {
        key: int(stats.get(key, 0)) for key in ("anon", "file", "inactive_file", "active_file", "slab_reclaimable")
    }
    cgroup_headroom = (
        None if limit is None else max(0, limit - current) + selected["inactive_file"] + selected["slab_reclaimable"]
    )
    host_available = psutil.virtual_memory().available
    usable = host_available if cgroup_headroom is None else min(cgroup_headroom, host_available)
    return {
        "current": current,
        "max": limit,
        "stat": selected,
        "events": events,
        "host_available": host_available,
        "conservative_reclaimable_headroom": usable,
    }


def save(path, result):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, path)


def run_case(name, command, output, inventory, timeout, gpu_count=2):
    env = os.environ.copy()
    env.update(
        LD_LIBRARY_PATH="",
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES=",".join(item["uuid"] for item in inventory[:gpu_count]),
    )
    log_path = output / f"{name}.log"
    start = time.time()
    memory_before = memory_state()
    if memory_before["conservative_reclaimable_headroom"] < MIN_HEADROOM:
        raise RuntimeError(f"Insufficient conservative cgroup/host headroom: {memory_before}")
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        pid = process.pid
        created = psutil.Process(pid).create_time()
        deadline = time.monotonic() + timeout
        min_headroom = memory_before["conservative_reclaimable_headroom"]
        peak_tree_rss = 0
        stop_reason = None
        try:
            while process.poll() is None:
                memory = memory_state()
                min_headroom = min(min_headroom, memory["conservative_reclaimable_headroom"])
                if min_headroom < MIN_HEADROOM:
                    stop_reason = "cgroup/host conservative memory headroom below threshold"
                    break
                if any(
                    memory["events"].get(key, 0) > memory_before["events"].get(key, 0) for key in ("oom", "oom_kill")
                ):
                    stop_reason = "cgroup OOM event counter increased"
                    break
                try:
                    parent = psutil.Process(pid)
                    peak_tree_rss = max(
                        peak_tree_rss,
                        sum(
                            child.memory_info().rss
                            for child in (parent, *parent.children(recursive=True))
                            if child.is_running()
                        ),
                    )
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                if time.monotonic() > deadline:
                    stop_reason = f"{name} timed out after {timeout}s"
                    break
                time.sleep(3)
        finally:
            if process.poll() is None:
                os.killpg(pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL)
                    process.wait(timeout=30)
        returncode = process.wait()
    return {
        "command": command,
        "pid": pid,
        "create_time": created,
        "returncode": returncode,
        "stop_reason": stop_reason,
        "start_time": start,
        "end_time": time.time(),
        "log": str(log_path),
        "memory_before": memory_before,
        "memory_after": memory_state(),
        "minimum_conservative_headroom": min_headroom,
        "peak_child_tree_rss": peak_tree_rss,
    }


def read_rank(path, rank, *, require_scope=True):
    data = json.loads(path.read_text())
    if not data.get("passed") or (require_scope and data.get("full_model_acceptance") is not False):
        raise ValueError(f"Qualification report failed: {path}")
    if "rank" in data and data["rank"] != rank:
        raise ValueError(f"Wrong rank in {path}")
    return data


def gate_adaptive(result, output):
    if result["returncode"]:
        raise ValueError("Adaptive qualification process failed")
    for rank in range(2):
        data = read_rank(output / f"adaptive.rank{rank}.json", rank)
        if data.get("updates") != 7 or len(data.get("updates_checked", [])) != 7 or not data.get("compile_events"):
            raise ValueError(f"Incomplete adaptive coverage on rank {rank}")
    return {"passed": True, "ranks": 2, "full_model_acceptance": False}


def gate_chunk(result, output):
    if result["returncode"]:
        raise ValueError("Checkpointed chunk qualification process failed")
    counts = []
    for rank in range(2):
        read_rank(output / f"chunk.rank{rank}.json", rank, require_scope=False)
        flat = read_rank(output / f"chunk.flat_checkpoint.rank{rank}.json", rank)
        chunk = read_rank(output / f"chunk.checkpointed_chunk.rank{rank}.json", rank)
        if (
            flat.get("chunk_tokens") != 1
            or min(flat.get("native_checkpoint_calls", 0), flat.get("tp_checkpoint_calls", 0)) < 4
            or chunk.get("chunk_checkpoints", 0) <= 0
        ):
            raise ValueError(f"Missing checkpoint witness on rank {rank}")
        counts.append(chunk["chunk_checkpoints"])
    return {"passed": True, "checkpointed_chunk_counts": counts, "full_model_acceptance": False}


def gate_nested_memory(result, output):
    if result["returncode"]:
        raise ValueError("Nested checkpoint memory process failed")
    data = json.loads((output / "nested_memory.json").read_text())
    if not (
        data.get("passed")
        and data.get("cuda_peak_reduced") is True
        and data.get("outputs_and_all_gradients_close") is True
    ):
        raise ValueError("Nested checkpoint output, gradient or peak-memory gate failed")
    if any(data[mode].get("outer_block_calls", 0) < 2 for mode in ("plain", "nested")):
        raise ValueError("Outer checkpoint recomputation was not observed")
    return {
        "passed": True,
        "plain_peak_bytes": data["plain"]["peak_allocated_bytes"],
        "nested_peak_bytes": data["nested"]["peak_allocated_bytes"],
        "full_model_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pid", type=int, required=True)
    parser.add_argument("--last-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    predecessors = [identity(args.first_pid, "run_after_streamed.py"), identity(args.last_pid, "run_36_after_vjp.py")]
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "coordinator_result.json"
    result = {
        "coordinator_pid": os.getpid(),
        "predecessors": predecessors,
        "completed": False,
        "failure": None,
        "scope": "three isolated tiny diagnostics; none proves full-model correctness or throughput",
        "gpu_isolation_limitation": "shared cooperative lock, NVML process list; host and container PID namespaces differ",
        "cases": {},
    }
    save(result_path, result)
    observed = {}
    try:
        while any(alive(item) for item in predecessors) or any(alive(item) for item in observed.values()):
            for item in predecessors:
                record_children(item, observed)
            for item in list(observed.values()):
                record_children(item, observed)
            time.sleep(3)
        result["observed_predecessor_children"] = list(observed.values())
        with LOCK.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for name in ("adaptive", "checkpointed_chunk", "nested_memory"):
                case = {"passed": False}
                result["cases"][name] = case
                case["gpu_before"] = wait_idle()
                before = sources()
                case["source_sha256_before"] = before
                output = args.output.resolve()
                python = str(ROOT / ".venv/bin/python")
                base = [python, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
                command = base + (
                    [
                        str(ROOT / "tools/qualify_tp_pp_adaptive_padding_gpu.py"),
                        "--output",
                        str(output / "adaptive.json"),
                    ]
                    if name == "adaptive"
                    else [
                        str(ROOT / "tools/qualify_tp_pp_checkpointed_chunked_mlp_math.py"),
                        "--tp",
                        "2",
                        "--pp",
                        "1",
                        "--chunk-tokens",
                        "1",
                        "--output",
                        str(output / "chunk.json"),
                    ]
                )
                if name == "nested_memory":
                    command = [
                        python,
                        str(ROOT / "tools/qualify_tp_pp_nested_checkpoint_memory.py"),
                        "--output",
                        str(output / "nested_memory.json"),
                    ]
                save(result_path, result)
                try:
                    case.update(
                        run_case(
                            name,
                            command,
                            output,
                            case["gpu_before"]["inventory"],
                            args.timeout,
                            1 if name == "nested_memory" else 2,
                        )
                    )
                    case["gpu_after"] = wait_idle()
                    case["source_sha256_after"] = sources()
                    if case["source_sha256_after"] != before:
                        raise RuntimeError("Sources changed during GPU qualification")
                    before_events = case["memory_before"]["events"]
                    after_events = case["memory_after"]["events"]
                    if any(after_events.get(key, 0) > before_events.get(key, 0) for key in ("oom", "oom_kill")):
                        raise RuntimeError("Cgroup OOM event counter increased during GPU qualification")
                    if case.get("stop_reason"):
                        raise RuntimeError(f"Process stopped by guard: {case['stop_reason']}")
                    gate = {
                        "adaptive": gate_adaptive,
                        "checkpointed_chunk": gate_chunk,
                        "nested_memory": gate_nested_memory,
                    }[name]
                    case["gate"] = gate(case, output)
                    case["passed"] = True
                except (Exception, SystemExit) as error:
                    case["failure"] = f"{type(error).__name__}: {error}"
                    # If the idle check itself fails, abort rather than launch a second case on occupied GPUs.
                    wait_idle()
                    if case.get("stop_reason") or "OOM event" in case["failure"] or "headroom" in case["failure"]:
                        raise RuntimeError(
                            f"Memory/timeout guard requires queue termination: {case['failure']}"
                        ) from error
                save(result_path, result)
        result["completed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(result_path, result)


if __name__ == "__main__":
    main()
