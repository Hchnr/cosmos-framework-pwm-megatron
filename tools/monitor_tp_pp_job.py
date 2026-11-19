"""Sample a named local torchrun job's host memory without changing training."""

import argparse
import json
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.interval <= 0 or args.startup_timeout <= 0:
        parser.error("Sampling interval and startup timeout must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started, seen = time.monotonic(), False
    with args.output.open("x") as handle:
        while True:
            processes = {}
            for process in psutil.process_iter(["cmdline"]):
                command = process.info["cmdline"] or []
                if any(part.endswith("/monitor_tp_pp_job.py") for part in command):
                    continue
                named_cli = False
                if "--job-name" in command and command.index("--job-name") + 1 < len(command):
                    named_cli = command[command.index("--job-name") + 1] == args.job_name
                if f"job.name={args.job_name}" not in command and not named_cli:
                    continue
                try:
                    processes[process.pid] = process
                    processes.update({child.pid: child for child in process.children(recursive=True)})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if not processes:
                if seen:
                    break
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError("Named training job did not start")
            else:
                seen = True
                rss, cpu, alive = 0, 0.0, []
                for pid, process in processes.items():
                    try:
                        rss += process.memory_info().rss
                        times = process.cpu_times()
                        cpu += times.user + times.system
                        alive.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                cgroup = Path("/sys/fs/cgroup/memory.current")
                handle.write(
                    json.dumps(
                        {
                            "unix_time": time.time(),
                            "elapsed_seconds": time.monotonic() - started,
                            "pids": sorted(alive),
                            "summed_rss_bytes": rss,
                            "cpu_seconds": cpu,
                            "cgroup_memory_current_bytes": int(cgroup.read_text()) if cgroup.is_file() else None,
                            "rss_counts_shared_pages_per_process": True,
                            "cgroup_memory_includes_other_processes_and_file_cache": True,
                        }
                    )
                    + "\n"
                )
                handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
