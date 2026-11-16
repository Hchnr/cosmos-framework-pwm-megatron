"""Supervise a SwanLab mirror by discovering W&B files, independently of training.

Run with the separate SwanLab Python environment on one tools node. No training
imports, process signals, or writes to the original W&B files are used.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STOP = False
WORKER = Path(__file__).with_name("swanlab_wandb_bridge.py").resolve()


def stop(*_):
    global STOP
    STOP = True


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def source_order(path):
    match = re.search(r"run-(\d{8}_\d{6})-", path.parent.name)
    return (match[1] if match else "", path.parent.name)


def discover(output_root):
    # Limit discovery to actual training job directories, excluding validation
    # logs and sidecar copies under OUTPUT_ROOT/logs.
    files = list(output_root.glob("cosmos3_action/action128/*/wandb/*/run-*.wandb"))
    return max(files, key=source_order).resolve() if files else None


def legacy_state_dir(output_root, source):
    for audit in (output_root / "logs").glob("launch_*/monitor/swanlab_bridge/extraction.json"):
        data = read_json(audit)
        if data.get("source") and Path(data["source"]).resolve() == source:
            if (audit.parent / "online_state.json").is_file():
                return audit.parent.resolve()
    return None


def verified_worker(pid, source, state_dir):
    try:
        process = Path("/proc") / str(int(pid))
        argv = [os.fsdecode(v) for v in (process / "cmdline").read_bytes().split(b"\0") if v]
        cwd = (process / "cwd").resolve()
        scripts = [(cwd / v).resolve() for v in argv if Path(v).name in ["bridge.py", "swanlab_wandb_bridge.py"]]
        if (state_dir / "bridge.py").resolve() in scripts:
            return read_json(state_dir / "extraction.json").get("source") == str(source)
        if WORKER not in scripts or "--source" not in argv or "--state-dir" not in argv:
            return False
        actual_source = (cwd / argv[argv.index("--source") + 1]).resolve()
        actual_state = (cwd / argv[argv.index("--state-dir") + 1]).resolve()
        return actual_source == source and actual_state == state_dir
    except (OSError, ValueError, TypeError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--proxy-file", type=Path, default=Path("/share-mixed/project/eai_pwm/.secrets/swanlab_proxy"))
    parser.add_argument("--key-file", type=Path, default=Path("/share-mixed/project/eai_pwm/.secrets/swanlab_api_key"))
    parser.add_argument("--workspace")
    parser.add_argument("--project", default="cosmos3_action")
    parser.add_argument("--name", help="Display name for a newly created SwanLab run")
    parser.add_argument("--step-log", type=Path, help="Rank0 training log used for step wall-time metrics")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true", help="Detach while retaining the shared manager lock")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    root = args.state_root.resolve() if args.state_root else output_root / "logs/swanlab_bridge"
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / "manager.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "manager_already_running"}), flush=True)
            return
        if args.background:
            for package in ("swanlab", "wandb"):
                if importlib.util.find_spec(package) is None:
                    parser.error(f"missing {package}; use the separate SwanLab Python environment")
            tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            logfile = root / f"manager_{tag}_{os.getpid()}.private.log"
            fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid:
                os.close(fd)
                print(json.dumps({"state": "manager_started", "pid": pid, "log": str(logfile)}), flush=True)
                return
            # The child retains the parent's flock, so concurrent node launchers
            # cannot start another manager during detachment.
            os.setsid()
            null = os.open(os.devnull, os.O_RDONLY)
            os.dup2(null, 0)
            os.close(null)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.close(fd)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    registry = read_json(root / "sources.json")
    children = []
    while not STOP:
        status = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "manager_pid": os.getpid(),
            "training_signaled": False,
            "discovery": "wandb_file_direct",
        }
        try:
            source = discover(output_root)
            if source is None:
                status["state"] = "waiting_for_wandb_file"
            else:
                key = str(source)
                entry = registry.get(key, {})
                if not entry:
                    state_dir = legacy_state_dir(output_root, source)
                    if state_dir is None:
                        suffix = hashlib.sha256(key.encode()).hexdigest()[:8]
                        state_dir = root / "sessions" / (source.parent.name + "-" + suffix)
                    entry = {"state_dir": str(state_dir), "source": key}
                    registry[key] = entry
                state_dir = Path(entry["state_dir"]).resolve()
                state = read_json(state_dir / "online_state.json")
                pid = next(
                    (p for p in [entry.get("pid"), state.get("pid")] if verified_worker(p, source, state_dir)), None
                )
                status.update(source_wandb_file=key, state_dir=str(state_dir))
                if verified_worker(pid, source, state_dir):
                    entry["pid"] = pid
                    status.update(
                        state="following",
                        uploader_pid=pid,
                        cloud_url=state.get("url"),
                        source_last_step=state.get("local_source_last_step"),
                        server_confirmed_steps=state.get("server_confirmed_steps", {}),
                        uploader_status_utc=state.get("utc"),
                    )
                elif source.stat().st_size < 256:
                    status["state"] = "waiting_for_wandb_header"
                elif time.time() - entry.get("last_started", 0) < 60:
                    status["state"] = "retry_backoff"
                elif args.dry_run:
                    status["state"] = "would_start_uploader"
                else:
                    state_dir.mkdir(parents=True, exist_ok=True)
                    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    logfile = state_dir / ("online_" + tag + ".private.log")
                    fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    command = [
                        sys.executable,
                        str(WORKER),
                        "--mode",
                        "online",
                        "--source",
                        key,
                        "--state-dir",
                        str(state_dir),
                        "--proxy-file",
                        str(args.proxy_file),
                        "--key-file",
                        str(args.key_file),
                        "--project",
                        args.project,
                        "--interval",
                        str(args.interval),
                        "--duration",
                        "28800",
                    ]
                    if args.workspace:
                        command.extend(["--workspace", args.workspace])
                    if args.name:
                        command.extend(["--name", args.name])
                    if args.step_log:
                        command.extend(["--step-log", str(args.step_log.resolve())])
                    with os.fdopen(fd, "w") as stream:
                        child = subprocess.Popen(
                            command,
                            stdin=subprocess.DEVNULL,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    children.append(child)
                    entry.update(pid=child.pid, last_started=time.time())
                    status.update(state="starting_uploader", uploader_pid=child.pid)
            if not args.dry_run:
                save_json(root / "sources.json", registry)
        except (OSError, ValueError, KeyError) as error:
            status.update(state="retrying", error_type=type(error).__name__)
        if not args.dry_run:
            save_json(root / "manager_status.json", status)
        print(json.dumps(status), flush=True)
        # Reap owned uploader processes without signaling any process.
        children = [child for child in children if child.poll() is None]
        if args.once:
            return
        until = time.monotonic() + args.interval
        while not STOP and time.monotonic() < until:
            time.sleep(1)


if __name__ == "__main__":
    main()
