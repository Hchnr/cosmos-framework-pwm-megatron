# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed GPU-user checks for host-PID and container-PID environments."""

import os
import stat
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import psutil


def process_tree(owner_pid):
    owner = psutil.Process(owner_pid)
    return {owner_pid, *(child.pid for child in owner.children(recursive=True))}


def stat_starttime(proc_dir):
    text = (proc_dir / "stat").read_text()
    # comm may contain spaces or ')' so use the final close parenthesis.
    return int(text[text.rfind(")") + 2 :].split()[19])


def validate_host_proc(host_proc, local_proc=Path("/proc")):
    if host_proc is None:
        raise RuntimeError("Strict GPU PID monitoring requires an explicit host procfs")
    host_proc = Path(host_proc)
    try:
        host_ns = (host_proc / "1/ns/pid").stat()
        local_ns = (local_proc / "self/ns/pid").stat()
        if (host_ns.st_dev, host_ns.st_ino) == (local_ns.st_dev, local_ns.st_ino):
            raise RuntimeError("Host procfs PID namespace cannot be distinguished from the container namespace")
        (host_proc / "1/status").read_text()
    except OSError as error:
        raise RuntimeError(f"Host procfs cannot be verified: {error}") from error
    return host_proc


def map_host_pid(host_pid, host_proc, local_proc=Path("/proc")):
    host = host_proc / str(host_pid)
    try:
        fields = dict(line.split(":", 1) for line in (host / "status").read_text().splitlines() if ":" in line)
        ns_pids = [int(value) for value in fields["NSpid"].split()]
        if not ns_pids or ns_pids[0] != host_pid:
            raise RuntimeError("Host procfs NSpid does not start with the NVML PID")
        host_ns = (host / "ns/pid").stat()
        current_ns = (local_proc / "self/ns/pid").stat()
        if (host_ns.st_dev, host_ns.st_ino) != (current_ns.st_dev, current_ns.st_ino):
            return None
        local_pid = ns_pids[-1]
        local = local_proc / str(local_pid)
        local_ns = (local / "ns/pid").stat()
        if (host_ns.st_dev, host_ns.st_ino) != (local_ns.st_dev, local_ns.st_ino):
            raise RuntimeError("Host and local process PID namespaces differ")
        if stat_starttime(host) != stat_starttime(local):
            raise RuntimeError("Host and local process start times differ")
        return local_pid
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise RuntimeError(f"Cannot verify NVML host PID {host_pid}: {error}") from error


def strict_check(users, owner_pid, host_proc, local_proc=Path("/proc")):
    owned = process_tree(owner_pid)
    mapping = []
    foreign = []
    for user in users:
        local_pid = map_host_pid(user["pid"], host_proc, local_proc)
        item = {**user, "local_pid": local_pid}
        mapping.append(item)
        if local_pid not in owned:
            foreign.append(item)
    if foreign:
        raise RuntimeError(f"Foreign GPU users observed: {foreign}")
    return {"mapping": mapping}


class StrictMonitor:
    def __init__(self, inventory, host_proc):
        self.host_proc = host_proc
        self.observed = set()
        self.expected = set(inventory.values())

    def check(self, users, owner_pid):
        details = strict_check(users, owner_pid, self.host_proc)
        self.observed.update(user["gpu_uuid"] for user in users)
        return details

    def finish(self):
        missing = self.expected - self.observed
        if missing:
            raise RuntimeError(f"No verified GPU context observed for device(s): {sorted(missing)}")


def query_device_nodes(inventory):
    output = subprocess.run(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True, timeout=20).stdout
    root = ET.fromstring(output)
    by_uuid = {gpu.findtext("uuid"): int(gpu.findtext("minor_number")) for gpu in root.findall("gpu")}
    if set(by_uuid) != set(inventory.values()):
        raise RuntimeError("NVML inventory and device minor numbers differ")
    devices = {}
    for uuid, minor in by_uuid.items():
        path = Path(f"/dev/nvidia{minor}")
        info = path.stat()
        if not stat.S_ISCHR(info.st_mode) or os.minor(info.st_rdev) != minor:
            raise RuntimeError(f"GPU device node cannot be verified: {path}")
        devices[uuid] = info.st_rdev
    return devices


def tree_device_fds(owner_pid, devices, local_proc=Path("/proc")):
    owned = process_tree(owner_pid)
    owners = {uuid: [] for uuid in devices}
    reverse = {device: uuid for uuid, device in devices.items()}
    for pid in owned:
        try:
            for fd in (local_proc / str(pid) / "fd").iterdir():
                try:
                    info = fd.stat()
                except FileNotFoundError:
                    continue
                if stat.S_ISCHR(info.st_mode) and info.st_rdev in reverse:
                    uuid = reverse[info.st_rdev]
                    if pid not in owners[uuid]:
                        owners[uuid].append(pid)
        except FileNotFoundError:
            # Exited worker; the next GPU snapshot will resolve its context.
            if psutil.pid_exists(pid):
                raise RuntimeError(f"Cannot read live process GPU FDs: {pid}")
        except PermissionError as error:
            raise RuntimeError(f"Cannot inspect GPU FDs for process {pid}") from error
    return owners


class RestrictedMonitor:
    """Provisional counts/FD evidence; cannot bind an NVML PID to a local PID."""

    def __init__(self, inventory, devices):
        self.devices = devices
        self.observed = {uuid: set() for uuid in inventory.values()}

    def check(self, users, owner_pid):
        owners = tree_device_fds(owner_pid, self.devices)
        per_gpu = {uuid: set() for uuid in self.devices}
        for user in users:
            per_gpu[user["gpu_uuid"]].add(user["pid"])
        all_pids = set()
        for uuid, pids in per_gpu.items():
            if len(pids) > 1:
                raise RuntimeError(f"Multiple GPU compute PIDs observed on {uuid}: {sorted(pids)}")
            if pids and not owners[uuid]:
                raise RuntimeError(f"GPU context on {uuid} lacks a device FD in the owner process tree")
            if self.observed[uuid] and pids and pids != self.observed[uuid]:
                raise RuntimeError(f"GPU compute PID changed on {uuid}")
            if pids & all_pids:
                raise RuntimeError("One GPU compute PID appeared on multiple devices")
            all_pids.update(pids)
            self.observed[uuid].update(pids)
        return {"host_pids_by_gpu": {k: sorted(v) for k, v in per_gpu.items()}, "local_fd_owners_by_gpu": owners}

    def finish(self):
        missing = [uuid for uuid, pids in self.observed.items() if not pids]
        if missing:
            raise RuntimeError(f"No GPU context observed for device(s): {missing}")
