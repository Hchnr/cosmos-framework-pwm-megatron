"""Run a command while sampling whole-device NVML memory, including startup.

NVML includes CUDA contexts and allocator caches that allocated-tensor peaks omit.
Sampling can miss short spikes; retain PyTorch allocated/reserved peaks as well.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pynvml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("a command and positive sampling interval are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("refusing to overwrite an existing memory trace")
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    initial = [pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in handles]
    peaks = [memory.used for memory in initial]
    process = None
    started = time.time()

    def terminate(signum, _frame):
        # Ensure an interrupted monitor cannot leave its training process group
        # running independently after the monitor exits.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        with args.output.open("x") as trace:
            process = subprocess.Popen(command, start_new_session=True)
            while True:
                memories = [pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in handles]
                used = [memory.used for memory in memories]
                peaks = [max(a, b) for a, b in zip(peaks, used, strict=True)]
                trace.write(json.dumps({"time": time.time(), "used_bytes": used}) + "\n")
                trace.flush()
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(args.interval)
        report = {
            "returncode": returncode,
            "elapsed_seconds": time.time() - started,
            "sampling_interval_seconds": args.interval,
            "devices": [
                {
                    "index": index,
                    "total_bytes": memory.total,
                    "initial_used_bytes": memory.used,
                    "peak_used_bytes": peak,
                    "peak_used_fraction": peak / memory.total,
                    "minimum_headroom_bytes": memory.total - peak,
                }
                for index, (memory, peak) in enumerate(zip(initial, peaks, strict=True))
            ],
        }
        args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        pynvml.nvmlShutdown()
    raise SystemExit(returncode if returncode >= 0 else 128 - returncode)


if __name__ == "__main__":
    main()
