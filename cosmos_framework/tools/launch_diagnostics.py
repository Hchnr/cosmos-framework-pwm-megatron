# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Persist node health independently of torchrun and record its exit status."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def cgroup_path(proc: Path = Path("/proc")) -> Path | None:
    try:
        membership = next(
            line[3:] for line in (proc / "self/cgroup").read_text().splitlines() if line.startswith("0::")
        )
        for line in (proc / "self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            try:
                return Path(fields[4]) / Path(membership).relative_to(fields[3])
            except ValueError:
                continue
    except (OSError, StopIteration, ValueError):
        pass
    return None


def memory_snapshot(cgroup: Path | None, proc: Path = Path("/proc")) -> dict:
    result = {}
    try:
        info = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
        result["host_available_bytes"] = int(info["MemAvailable"].split()[0]) * 1024
    except (OSError, ValueError, KeyError):
        pass
    if cgroup is not None:
        for name in (
            "memory.current",
            "memory.max",
            "memory.high",
            "memory.low",
            "memory.peak",
            "memory.events",
            "memory.stat",
            "cpu.stat",
            "cpu.max",
            "pids.current",
            "pids.max",
            "pids.events",
        ):
            try:
                value = (cgroup / name).read_text().strip()
                if name in {"memory.events", "memory.stat", "cpu.stat", "pids.events"}:
                    result[name] = {key: int(number) for key, number in (line.split() for line in value.splitlines())}
                elif name == "cpu.max":
                    result[name] = value
                else:
                    result[name] = value if value == "max" else int(value)
            except (OSError, ValueError):
                pass
        for name in ("cpu.pressure", "memory.pressure", "io.pressure"):
            try:
                result[name] = (cgroup / name).read_text().strip()
            except OSError:
                pass
    try:
        # Inspect local tmpfs only; querying a stalled shared mount can block diagnostics.
        shm = os.statvfs("/dev/shm")
        result["shm"] = {
            "capacity_bytes": shm.f_blocks * shm.f_frsize,
            "available_bytes": shm.f_bavail * shm.f_frsize,
            "free_bytes": shm.f_bfree * shm.f_frsize,
            "free_inodes": shm.f_ffree,
        }
    except OSError:
        pass
    return result


def run(command: list[str], output: Path, interval: float) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    cgroup = cgroup_path()
    with output.open("a", buffering=1) as stream:

        def record(event, **fields):
            payload = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
            try:
                stream.write(json.dumps(payload) + "\n")
                stream.flush()
            except OSError as error:
                # Telemetry failure must not terminate otherwise healthy training.
                print(f"launch diagnostics write failed: {error}", file=sys.stderr)

        record("start", hostname=socket.gethostname(), pid=os.getpid(), cgroup=str(cgroup), command=command)
        try:
            child = subprocess.Popen(command)
        except OSError as error:
            record("launch_error", error=str(error))
            return 127

        def forward(signum, _frame):
            record("signal", signal=signum)
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                record("memory", **memory_snapshot(cgroup))
                try:
                    status = child.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    continue
            record("exit", returncode=status, **memory_snapshot(cgroup))
            return status if status >= 0 else 128 - status
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("a command and a positive interval are required")
    filename = f"{time.time_ns()}_{socket.gethostname()}_{os.getpid()}.jsonl"
    output = args.output_dir / filename
    print(f">>> Node diagnostics: {output}", flush=True)
    sys.exit(run(command, output, args.interval))


if __name__ == "__main__":
    main()
