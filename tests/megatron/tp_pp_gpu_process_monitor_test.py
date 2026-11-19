# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os

import pytest

from tools import tp_pp_gpu_process_monitor as monitor


def fixture_procs(tmp_path):
    host = tmp_path / "host"
    local = tmp_path / "local"
    for root, pid in ((host, "1"), (host, "1363191"), (local, "self"), (local, "445222")):
        (root / pid / "ns").mkdir(parents=True)
    (host / "1/ns/pid").write_text("host pid ns")
    (host / "1/status").write_text("NSpid:\t1\n")
    (local / "self/ns/pid").write_text("container pid ns")
    os.link(local / "self/ns/pid", local / "445222/ns/pid")
    os.link(local / "self/ns/pid", host / "1363191/ns/pid")
    (host / "1363191/status").write_text("NSpid:\t1363191\t445222\n")
    stat_line = "123 (name with ) chars) S " + "0 " * 18 + "999 "
    (host / "1363191/stat").write_text(stat_line)
    (local / "445222/stat").write_text(stat_line)
    return host, local


def test_host_pid_mapping_requires_host_proc_and_matching_identity(tmp_path):
    host, local = fixture_procs(tmp_path)
    assert monitor.validate_host_proc(host, local) == host
    assert monitor.map_host_pid(1363191, host, local) == 445222
    (local / "445222/stat").write_text("123 (foo) S " + "0 " * 18 + "1000 ")
    with pytest.raises(RuntimeError, match="start times differ"):
        monitor.map_host_pid(1363191, host, local)
    with pytest.raises(RuntimeError, match="explicit host procfs"):
        monitor.validate_host_proc(None, local)
    with pytest.raises(RuntimeError, match="cannot be verified"):
        monitor.validate_host_proc(tmp_path / "missing", local)


def test_strict_monitor_rejects_unmapped_or_foreign_pids(tmp_path, monkeypatch):
    host, local = fixture_procs(tmp_path)
    monkeypatch.setattr(monitor, "process_tree", lambda pid: {445222})
    users = [{"gpu_uuid": "GPU-a", "pid": 1363191}]
    assert monitor.strict_check(users, 1, host, local)["mapping"][0]["local_pid"] == 445222
    monkeypatch.setattr(monitor, "process_tree", lambda pid: {123})
    with pytest.raises(RuntimeError, match="Foreign GPU users"):
        monitor.strict_check(users, 1, host, local)
    with pytest.raises(RuntimeError, match="Cannot verify NVML host PID"):
        monitor.strict_check([{"gpu_uuid": "GPU-a", "pid": 999}], 1, host, local)


def test_restricted_monitor_rejects_extra_or_replaced_gpu_pids(monkeypatch):
    inventory = {0: "a", 1: "b", 2: "c", 3: "d"}
    device_nodes = {uuid: n for n, uuid in inventory.items()}
    monkeypatch.setattr(monitor, "tree_device_fds", lambda owner, nodes: {uuid: [10] for uuid in nodes})
    checker = monitor.RestrictedMonitor(inventory, device_nodes)
    checker.check([{"gpu_uuid": "a", "pid": 100}], 10)
    with pytest.raises(RuntimeError, match="Multiple GPU compute"):
        checker.check([{"gpu_uuid": "a", "pid": 100}, {"gpu_uuid": "a", "pid": 200}], 10)
    with pytest.raises(RuntimeError, match="PID changed"):
        checker.check([{"gpu_uuid": "a", "pid": 200}], 10)
    with pytest.raises(RuntimeError, match="multiple devices"):
        checker.check([{"gpu_uuid": "a", "pid": 100}, {"gpu_uuid": "b", "pid": 100}], 10)
    with pytest.raises(RuntimeError, match="No GPU context observed"):
        checker.finish()


def test_restricted_monitor_requires_fds_and_all_four_gpus(monkeypatch):
    inventory = {0: "a", 1: "b", 2: "c", 3: "d"}
    devices = {uuid: n for n, uuid in inventory.items()}
    monkeypatch.setattr(monitor, "tree_device_fds", lambda owner, nodes: {uuid: [] for uuid in nodes})
    checker = monitor.RestrictedMonitor(inventory, devices)
    with pytest.raises(RuntimeError, match="lacks a device FD"):
        checker.check([{"gpu_uuid": "a", "pid": 100}], 10)
    monkeypatch.setattr(monitor, "tree_device_fds", lambda owner, nodes: {uuid: [10] for uuid in nodes})
    checker.check([{"gpu_uuid": uuid, "pid": 100 + index} for index, uuid in inventory.items()], 10)
    checker.finish()
