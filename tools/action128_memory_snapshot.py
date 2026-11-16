"""Read-only host/cgroup/process memory snapshot; run inside a training container.

No CUDA imports, process attachment, cache flushing, or training changes.
RSS sums double-count shared pages; use PSS and cgroup counters for attribution.
Output contains process IDs/roles, not command arguments or environment values.
"""

import argparse
import datetime
import json
import os
import socket
import time
from pathlib import Path


def counters(path, *, kilobytes=False):
    result = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                result[parts[0].rstrip(":")] = int(parts[1]) * (1024 if kilobytes else 1)
    except (OSError, ValueError):
        pass
    return result


def snapshot(include_pss=False):
    started = time.monotonic()
    result = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "host_meminfo_bytes": counters(Path("/proc/meminfo"), kilobytes=True),
        "cgroups": [],
        "processes": [],
        "errors": [],
    }
    # Resolve the current cgroup within the visible v2 mount and include visible
    # ancestors: the enclosing pod may have a lower limit than the container.
    memberships = Path("/proc/self/cgroup").read_text().splitlines()
    group = next((line[3:] for line in memberships if line.startswith("0::")), None)
    if group is not None:
        for mount in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = mount.split(" - ", 1)
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            mount_root, mount_path = Path(fields[3]), Path(fields[4])
            try:
                relative = Path(group).relative_to(mount_root)
            except ValueError:
                result["errors"].append("cgroup membership outside visible mount")
                continue
            current = mount_path / relative
            while current.is_relative_to(mount_path):
                row = {"path": str(current)}
                for name in (
                    "memory.current",
                    "memory.max",
                    "memory.high",
                    "memory.peak",
                    "memory.swap.current",
                    "memory.swap.max",
                    "memory.pressure",
                ):
                    try:
                        value = (current / name).read_text().strip()
                        row[name] = int(value) if value.isdigit() else value
                    except OSError:
                        pass
                for name in ("memory.stat", "memory.events"):
                    row[name] = counters(current / name)
                result["cgroups"].append(row)
                if current == mount_path:
                    break
                current = current.parent
            break
    # Inventory the container's visible processes. Read expensive PSS only for
    # training ranks and their descendants, never attach a debugger.
    inventory = {}
    roots = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = (path / "status").read_text()
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            pid = int(path.name)
            role = "other"
            if "--helper" in command and "media_decode" in command:
                role = "decoder"
            elif "torchrun" in command:
                role = "launcher"
            elif "cosmos_framework.scripts.train" in command or "tools.action128_spike_probe" in command:
                role = "training_or_worker"
                roots.add(pid)
            row = {"pid": pid, "ppid": int(fields["PPid"]), "name": fields["Name"].strip(), "role": role}
            for key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmLck", "VmPin"):
                if key in fields:
                    row[key] = int(fields[key].split()[0]) * 1024
            inventory[pid] = row
        except (OSError, ValueError, KeyError):
            continue
    selected = set(roots)
    while True:
        children = {pid for pid, row in inventory.items() if row["ppid"] in selected}
        if children <= selected:
            break
        selected |= children
    for pid, row in inventory.items():
        if pid in selected:
            if row["role"] == "training_or_worker":
                row["role"] = "worker" if row["ppid"] in selected else "trainer"
            if include_pss:
                values = counters(Path(f"/proc/{pid}/smaps_rollup"), kilobytes=True)
                if not values:
                    result["errors"].append(f"smaps_rollup unavailable for pid {pid}")
                row["smaps_bytes"] = values
        result["processes"].append(row)
    shm = os.statvfs("/dev/shm")
    result["dev_shm"] = {
        "total_bytes": shm.f_blocks * shm.f_frsize,
        "used_bytes": (shm.f_blocks - shm.f_bfree) * shm.f_frsize,
    }
    result["sampling_seconds"] = time.monotonic() - started
    result["notes"] = (
        "Host counters and cgroup counters differ; RSS is not additive physical memory. file includes shmem."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pss", action="store_true", help="Read smaps_rollup for training process trees; may take seconds"
    )
    args = parser.parse_args()
    print(json.dumps(snapshot(args.pss), indent=2))
