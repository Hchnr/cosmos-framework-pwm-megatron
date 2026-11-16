"""Exercise exit reporting without starting GPUs or importing training code."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cosmos_framework.tools.launch_diagnostics import cgroup_path, memory_snapshot, run


def test_memory_events_and_container_limit(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    cgroup = tmp_path / "cgroup"
    (cgroup / "pod").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/pod\n")
    (proc / "self/mountinfo").write_text(f"1 0 0:1 / {cgroup} rw - cgroup2 cgroup rw\n")
    (proc / "meminfo").write_text("MemAvailable: 4096 kB\n")
    (cgroup / "pod/memory.current").write_text("1024")
    (cgroup / "pod/memory.max").write_text("2048")
    (cgroup / "pod/memory.events").write_text("oom 3\noom_kill 2\n")
    snapshot = memory_snapshot(cgroup_path(proc), proc)
    assert snapshot["memory.max"] == 2048
    assert snapshot["memory.events"]["oom_kill"] == 2
    assert snapshot["host_available_bytes"] == 4096 * 1024


@pytest.mark.parametrize("status", [0, 7])
def test_child_exit_is_persisted(tmp_path, status):
    path = tmp_path / "events.jsonl"
    assert run([sys.executable, "-c", f"raise SystemExit({status})"], path, 0.05) == status
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["event"] == "start"
    assert records[-1]["event"] == "exit"
    assert records[-1]["returncode"] == status


def test_killed_child_is_reported_as_signal(tmp_path):
    path = tmp_path / "events.jsonl"
    command = [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"]
    assert run(command, path, 0.05) == 137
    assert json.loads(path.read_text().splitlines()[-1])["returncode"] == -signal.SIGKILL


def test_monitor_samples_while_child_is_idle(tmp_path):
    path = tmp_path / "events.jsonl"
    assert run([sys.executable, "-c", "import time; time.sleep(0.2)"], path, 0.03) == 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(record["event"] == "memory" for record in records) >= 3


def test_termination_is_forwarded_to_child(tmp_path):
    ready = tmp_path / "ready"
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.tools.launch_diagnostics",
        "--output-dir",
        str(tmp_path),
        "--interval=0.03",
        "--",
        sys.executable,
        "-c",
        f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(30)",
    ]
    monitor = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        monitor.send_signal(signal.SIGTERM)
        assert monitor.wait(timeout=5) == 143
        records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
        assert any(record["event"] == "signal" and record["signal"] == signal.SIGTERM for record in records)
        assert records[-1]["returncode"] == -signal.SIGTERM
    finally:
        if monitor.poll() is None:
            monitor.send_signal(signal.SIGTERM)
            monitor.wait(timeout=5)


def test_launcher_reports_failure_with_errexit(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    launcher = tmp_path / "launch.sh"
    launcher.write_text(f'set -euo pipefail\nsource "{repo}/examples/_sft_launcher_common.sh"\n')
    stub = tmp_path / "torchrun"
    stub.write_text("#!/bin/sh\nexit 23\n")
    stub.chmod(0o755)
    toml = tmp_path / "recipe.toml"
    toml.write_text("")
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TOML_FILE": str(toml),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "REQUIRE_WAN_VAE": "0",
    }
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 23, result.stderr
    assert "Done (exit 23)" in result.stdout
