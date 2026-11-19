# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_tp4_fsdp4_raw_clean_coordinator as coordinator


def test_direct_script_entry_without_pythonpath_is_importable(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/run_tp4_fsdp4_raw_clean_coordinator.py"
    env = {**os.environ, "PYTHONPATH": "", "LD_LIBRARY_PATH": ""}
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "--pid-monitor-mode" in result.stdout


def test_direct_script_strict_preflight_fails_closed_without_host_proc(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/run_tp4_fsdp4_raw_clean_coordinator.py"
    env = {**os.environ, "PYTHONPATH": "", "LD_LIBRARY_PATH": ""}
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sequence-helper",
            "unused",
            "--attention-identity",
            "unused",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    saved = json.loads((output / "coordinator_result.json").read_text())
    assert "requires an explicit host procfs" in saved["failure"] and saved["runs"] == {}


def test_completion_must_be_later_than_predecessor(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            (
                json.dumps({"time": 1, "attention_isolation_finished": True}),
                json.dumps({"time": 5, "attention_isolation_finished": True}),
                '{"unfinished"',
            )
        )
    )
    assert coordinator.matching_completion(events, {"create_time": 2})["time"] == 5
    assert coordinator.matching_completion(events, {"create_time": 6}) is None


@pytest.mark.parametrize(
    "terminal",
    [
        {"time": 5, "attention_isolation_native_failed": 1},
        {"time": 5, "attention_isolation_finished": True, "numerical_returncode": 1},
        {"time": 5, "attention_isolation_finished": True, "analysis_returncodes": {"0": 1}},
    ],
)
def test_attention_terminal_failures_do_not_cancel_performance(tmp_path, terminal):
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(terminal) + "\n")
    assert coordinator.matching_completion(events, {"create_time": 2}) == terminal


def test_gpu_inventory_and_unknown_device_fail_closed(monkeypatch):
    response = SimpleNamespace(stdout="0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n")
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *a, **kw: response)
    inventory = coordinator.query_gpu_inventory()
    assert inventory == {0: "GPU-a", 1: "GPU-b", 2: "GPU-c", 3: "GPU-d"}
    response.stdout = "GPU-b, 123\n"
    assert coordinator.query_gpu_users(inventory) == [{"gpu_uuid": "GPU-b", "pid": 123}]
    response.stdout = "GPU-unknown, 123\n"
    with pytest.raises(ValueError, match="Unknown GPU UUID"):
        coordinator.query_gpu_users(inventory)
    response.stdout = "0, GPU-a\n1, GPU-b\n"
    with pytest.raises(ValueError, match="Exactly four"):
        coordinator.query_gpu_inventory()


def test_benchmark_stops_on_foreign_user_and_keeps_diagnostics(tmp_path, monkeypatch):
    class FakeProcess:
        pid = os.getpid()
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15

    monkeypatch.setattr(coordinator.subprocess, "Popen", lambda *a, **kw: FakeProcess())
    monkeypatch.setattr(coordinator.os, "killpg", lambda *a: None)
    monkeypatch.setattr(coordinator, "query_gpu_users", lambda inventory: [{"gpu_uuid": "GPU-a", "pid": 987654321}])
    monkeypatch.setattr(coordinator.time, "sleep", lambda seconds: None)
    sequence = SimpleNamespace(
        TP4="tp4",
        TP2="tp2",
        SPOOL="spool",
        TOML="config",
        WORK=tmp_path,
        ENV={},
        distributed=lambda *args: list(map(str, args)),
    )
    result = {"runs": {}}
    status = tmp_path / "coordinator_result.json"

    class Monitor:
        def check(self, users, owner):
            raise RuntimeError("Foreign GPU users")

    with pytest.raises(RuntimeError, match="Foreign GPU users"):
        coordinator.run_benchmark(
            sequence,
            "tp",
            tmp_path,
            tmp_path / "fixture",
            {0: "GPU-a", 1: "GPU-b", 2: "GPU-c", 3: "GPU-d"},
            result,
            status,
            0.01,
            lambda: None,
            Monitor(),
        )
    persisted = json.loads(status.read_text())
    run = persisted["runs"]["tp"]
    assert run["returncode"] == -15 and run["failure_snapshot"]["users"][0]["pid"] == 987654321
    assert (tmp_path / "tp.log").is_file()
    assert run["command"][-4:] == ["--updates", "7", "--warmup-updates", "2"]


@pytest.mark.parametrize(
    "terminal",
    [
        {"time": 5, "attention_isolation_native_failed": 1},
        {"time": 5, "attention_isolation_finished": True, "numerical_returncode": 1, "analysis_returncodes": {"0": 1}},
    ],
)
def test_failed_attention_still_runs_both_performance_cases(tmp_path, monkeypatch, terminal):
    identity = tmp_path / "attention.json"
    identity.write_text(json.dumps({"pid": 987654321, "create_time": 2}))
    (tmp_path / "sequence_events.jsonl").write_text(json.dumps(terminal) + "\n")
    fixture = tmp_path / "raw_microbatch_fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text(
        json.dumps({"passed": True, "records": [{"samples": n} for n in (9, 11, 13)]})
    )
    sequence = SimpleNamespace(OUTPUT=tmp_path, WORK=tmp_path)
    compare = SimpleNamespace(compare=lambda *args: {"measurement_completed": True})
    monkeypatch.setattr(
        coordinator, "load_module", lambda path, name: sequence if name == "_raw_clean_sequence" else compare
    )
    monkeypatch.setattr(coordinator, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(coordinator, "query_gpu_inventory", lambda: {i: f"GPU-{i}" for i in range(4)})
    monkeypatch.setattr(coordinator, "query_gpu_users", lambda inventory: [])
    monkeypatch.setattr(coordinator.gpu_monitor, "query_device_nodes", lambda inventory: inventory)
    monkeypatch.setattr(coordinator, "live", lambda identity: False)
    seen = []
    monkeypatch.setattr(coordinator, "run_benchmark", lambda sequence, backend, *args: seen.append(backend))
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--sequence-helper",
            "unused",
            "--attention-identity",
            str(identity),
            "--output",
            str(output),
            "--pid-monitor-mode",
            "restricted",
        ],
    )
    coordinator.main()
    result = json.loads((output / "coordinator_result.json").read_text())
    assert result["completed"] and result["attention_terminal_event"] == terminal
    assert seen == ["tp", "fsdp"]


def test_missing_attention_terminal_event_fails_with_saved_reason(tmp_path, monkeypatch):
    identity = tmp_path / "attention.json"
    identity.write_text(json.dumps({"pid": 987654321, "create_time": 2}))
    (tmp_path / "sequence_events.jsonl").write_text("")
    monkeypatch.setattr(coordinator, "load_module", lambda path, name: SimpleNamespace(OUTPUT=tmp_path))
    monkeypatch.setattr(coordinator, "live", lambda identity: False)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--sequence-helper",
            "unused",
            "--attention-identity",
            str(identity),
            "--output",
            str(output),
            "--pid-monitor-mode",
            "restricted",
        ],
    )
    with pytest.raises(RuntimeError, match="without its completion event"):
        coordinator.main()
    result = json.loads((output / "coordinator_result.json").read_text())
    assert not result["completed"] and "without its completion event" in result["failure"]
    assert result["runs"] == {}
