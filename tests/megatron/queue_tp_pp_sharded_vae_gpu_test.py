# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
import sys

import psutil
import pytest

from tools import queue_tp_pp_sharded_vae_gpu as queue


def example_snapshot():
    return {
        "version": 1,
        "node": {"hostname": "node", "boot_id": "boot"},
        "predecessor": {"pid": 585516, "create_time": 10.0, "command": [queue.PREDECESSOR_FRAGMENT]},
        "observed_descendants": [{"pid": 11, "create_time": 11.0, "command": ["child"]}],
    }


def test_identity_fails_closed_for_node_or_reused_or_missing_parent(monkeypatch):
    data = example_snapshot()
    monkeypatch.setattr(queue, "boot_identity", lambda: data["node"])
    monkeypatch.setattr(queue, "alive", lambda _: True)
    queue.validate_snapshot(data, 585516)
    with pytest.raises(ValueError, match="expected coordinator"):
        queue.validate_snapshot(data, 585517)
    data["node"] = {"hostname": "other", "boot_id": "new"}
    monkeypatch.setattr(queue, "boot_identity", lambda: {"hostname": "node", "boot_id": "boot"})
    with pytest.raises(ValueError, match="node/boot"):
        queue.validate_snapshot(data, 585516)
    data = example_snapshot()
    monkeypatch.setattr(queue, "alive", lambda _: False)
    with pytest.raises(RuntimeError, match="no longer alive"):
        queue.validate_snapshot(data, 585516)
    data["observed_descendants"] = None
    with pytest.raises(ValueError, match="descendants"):
        queue.validate_snapshot(data, 585516)


def test_process_identity_pid_reuse_and_access_denied(monkeypatch):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 11.0

        def status(self):
            return psutil.STATUS_RUNNING

    monkeypatch.setattr(queue.psutil, "Process", FakeProcess)
    assert not queue.alive({"pid": 9, "create_time": 10.0})
    assert queue.alive({"pid": 9, "create_time": 11.0})

    def denied(_):
        raise psutil.AccessDenied()

    monkeypatch.setattr(queue.psutil, "Process", denied)
    with pytest.raises(psutil.AccessDenied):
        queue.alive({"pid": 9, "create_time": 11.0})


def test_wait_tracks_descendant_after_parent_exit(monkeypatch):
    data = example_snapshot()
    parent, first_child = data["predecessor"], data["observed_descendants"][0]
    grandchild = {"pid": 12, "create_time": 12.0, "command": ["grandchild"]}
    tick = [0]
    monkeypatch.setattr(queue, "boot_identity", lambda: data["node"])
    monkeypatch.setattr(queue.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(queue.time, "sleep", lambda _: tick.__setitem__(0, tick[0] + 1))
    monkeypatch.setattr(queue, "alive", lambda item: tick[0] < {585516: 1, 11: 2, 12: 4}[item["pid"]])
    monkeypatch.setattr(queue, "descendants", lambda item: [grandchild] if item == first_child and tick[0] == 0 else [])
    rows = []
    observed = queue.await_predecessors(data, timeout=10, poll=1, status=lambda items, live: rows.append(live))
    assert {item["pid"] for item in observed} == {parent["pid"], first_child["pid"], grandchild["pid"]}
    assert tick[0] == 4 and rows[-1] == []


def test_wait_tracks_both_predecessor_trees(monkeypatch):
    first = example_snapshot()
    second = {
        **example_snapshot(),
        "command_fragment": "run_tp_pp_original_capacity_reference_coordinator.py",
        "predecessor": {
            "pid": 599884,
            "create_time": 20.0,
            "command": ["run_tp_pp_original_capacity_reference_coordinator.py"],
        },
        "observed_descendants": [{"pid": 22, "create_time": 22.0, "command": ["child"]}],
    }
    tick = [0]
    monkeypatch.setattr(queue, "boot_identity", lambda: first["node"])
    monkeypatch.setattr(queue.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(queue.time, "sleep", lambda _: tick.__setitem__(0, tick[0] + 1))
    expires = {585516: 1, 11: 2, 599884: 3, 22: 5}
    monkeypatch.setattr(queue, "alive", lambda item: tick[0] < expires[item["pid"]])
    monkeypatch.setattr(queue, "descendants", lambda _: [])
    queue.validate_snapshot(first, 585516)
    queue.validate_snapshot(second, 599884)
    observed = queue.await_predecessors([first, second], timeout=10, poll=1)
    assert {item["pid"] for item in observed} == set(expires)
    assert tick[0] == 5


def test_launched_process_tree_retains_lock_until_descendants_finish(monkeypatch):
    parent = {"pid": 1, "create_time": 1.0}
    child = {"pid": 2, "create_time": 2.0}
    tick = [0]

    class FakeProcess:
        def poll(self):
            return 0 if tick[0] >= 1 else None

    monkeypatch.setattr(queue.time, "sleep", lambda _: tick.__setitem__(0, tick[0] + 1))
    monkeypatch.setattr(queue, "alive", lambda item: tick[0] < {1: 1, 2: 3}[item["pid"]])
    monkeypatch.setattr(queue, "descendants", lambda item: [child] if item == parent and tick[0] == 0 else [])
    assert queue.await_launched_tree(FakeProcess(), parent, poll=1) == 0
    assert tick[0] == 3


def test_idle_requires_two_consecutive_observations_and_same_inventory(monkeypatch):
    def state(users=(), utilization=0):
        return {
            "inventory": [{"index": i, "uuid": f"GPU-{i}", "utilization": utilization} for i in range(4)],
            "compute_users": list(users),
        }

    states = iter([state(users=["busy"]), state(), state(utilization=5), state(), state()])
    monkeypatch.setattr(queue, "gpu_state", lambda: next(states))
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    assert len(queue.await_idle(timeout=10, poll=0.01)) == 2

    states = iter(
        [state(), {**state(), "inventory": [{"index": i, "uuid": "changed", "utilization": 0} for i in range(4)]}]
    )
    monkeypatch.setattr(queue, "gpu_state", lambda: next(states))
    with pytest.raises(RuntimeError, match="inventory changed"):
        queue.await_idle(timeout=10, poll=0.01)


def test_dry_run_never_queries_or_launches_gpu(monkeypatch, tmp_path):
    data = example_snapshot()
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(data))
    second = {
        **data,
        "command_fragment": "run_tp_pp_original_capacity_reference_coordinator.py",
        "predecessor": {
            "pid": 599884,
            "create_time": 20.0,
            "command": ["run_tp_pp_original_capacity_reference_coordinator.py"],
        },
    }
    identity2 = tmp_path / "identity2.json"
    identity2.write_text(json.dumps(second))
    output = tmp_path / "out"
    monkeypatch.setattr(queue, "boot_identity", lambda: data["node"])
    monkeypatch.setattr(queue, "alive", lambda _: True)
    monkeypatch.setattr(queue, "DEFAULT_SOURCES", ())
    monkeypatch.setenv("WAN_VAE_PATH", "/example/vae")
    monkeypatch.setattr(queue, "gpu_state", lambda: pytest.fail("dry run queried GPU"))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("dry run launched a process"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "queue",
            "--identity",
            str(identity),
            "--expected-pid",
            "585516",
            "--identity",
            str(identity2),
            "--expected-pid",
            "599884",
            "--output",
            str(output),
            "--require-env",
            "WAN_VAE_PATH",
            "--",
            "mock-command",
            "--flag",
        ],
    )
    queue.main()
    result = json.loads((output / "queue_result.json").read_text())
    assert result["completed"] and result["mode"] == "dry_run"
    assert [row["predecessor"]["pid"] for row in result["predecessor_snapshots"]] == [585516, 599884]
    assert result["command"] == ["mock-command", "--flag"]
    assert result["required_environment"]["WAN_VAE_PATH"]["sha256"] == hashlib.sha256(b"/example/vae").hexdigest()
    assert "gpu_before" not in result and "launched" not in result


def test_missing_required_environment_fails_without_gpu(monkeypatch, tmp_path):
    data = example_snapshot()
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(data))
    output = tmp_path / "out"
    monkeypatch.setattr(queue, "boot_identity", lambda: data["node"])
    monkeypatch.setattr(queue, "alive", lambda _: True)
    monkeypatch.setattr(queue, "DEFAULT_SOURCES", ())
    monkeypatch.delenv("WAN_VAE_PATH", raising=False)
    monkeypatch.setattr(queue, "gpu_state", lambda: pytest.fail("queried GPU"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "queue",
            "--identity",
            str(identity),
            "--expected-pid",
            "585516",
            "--output",
            str(output),
            "--require-env",
            "WAN_VAE_PATH",
            "--execute",
            "--",
            "never",
        ],
    )
    with pytest.raises(ValueError, match="WAN_VAE_PATH"):
        queue.main()
    result = json.loads((output / "queue_result.json").read_text())
    assert not result["completed"] and "WAN_VAE_PATH" in result["failure"]


def test_source_change_after_queue_fails_before_command_launch(monkeypatch, tmp_path):
    data = example_snapshot()
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(data))
    output = tmp_path / "out"
    first, second, extra = (tmp_path / name for name in ("worker.py", "primitive.py", "extra.py"))
    for path in (first, second, extra):
        path.write_text("original")
    monkeypatch.setattr(queue, "DEFAULT_SOURCES", (first, second))
    monkeypatch.setattr(queue, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(queue, "boot_identity", lambda: data["node"])
    monkeypatch.setattr(queue, "alive", lambda _: True)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched changed source"))

    def predecessor_finished(_data, *, timeout, poll, status):
        status([], [])
        return []

    def idle_then_change(**_kwargs):
        extra.write_text("changed while waiting")
        return [{"inventory": [{"uuid": f"GPU-{i}"} for i in range(4)], "compute_users": []}]

    monkeypatch.setattr(queue, "await_predecessors", predecessor_finished)
    monkeypatch.setattr(queue, "await_idle", idle_then_change)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "queue",
            "--identity",
            str(identity),
            "--expected-pid",
            "585516",
            "--output",
            str(output),
            "--source",
            str(extra),
            "--execute",
            "--",
            "never",
        ],
    )
    with pytest.raises(RuntimeError, match="source changed"):
        queue.main()
    result = json.loads((output / "queue_result.json").read_text())
    assert not result["completed"] and "source changed" in result["failure"]
    assert set(result["source_sha256_on_queue"]) == {str(first), str(second), str(extra)}
    assert result["source_sha256_on_queue"][str(extra)] != result["source_sha256_before_launch"][str(extra)]


def test_missing_identity_records_failure_without_launch(monkeypatch, tmp_path):
    output = tmp_path / "out"
    monkeypatch.setattr(queue, "gpu_state", lambda: pytest.fail("queried GPU"))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched process"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "queue",
            "--identity",
            str(tmp_path / "absent"),
            "--expected-pid",
            "585516",
            "--output",
            str(output),
            "--",
            "never",
        ],
    )
    with pytest.raises(FileNotFoundError):
        queue.main()
    result = json.loads((output / "queue_result.json").read_text())
    assert not result["completed"] and "FileNotFoundError" in result["failure"]
