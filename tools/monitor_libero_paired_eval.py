"""Refresh local paired-evaluation artifacts until all scheduled trials finish."""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, default=20)
    args = parser.parse_args()
    last = None
    failures = 0
    while True:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        try:
            for path in [args.root / "launch_manifest.json", *args.root.glob("shard1_launch_*.json")]:
                value = json.loads(path.read_text())
                jobs = value["jobs"] if isinstance(value, dict) else value
                for job in jobs:
                    if job["kind"] != "simulator":
                        continue
                    command = job["command"]
                    trial_ids = command[command.index("--trials") + 1].split(",")
                    output = Path(command[command.index("--output") + 1])
                    if all((output / f"trial_{int(trial):03d}.json").exists() for trial in trial_ids):
                        continue
                    process = Path(f"/proc/{job['pid']}/stat")
                    if not process.exists() or process.read_text().split(") ")[1].split()[0] == "Z":
                        raise RuntimeError(f"Simulator exited before completing trials: {job['log']}")
            captures = [args.root / f"{tower}_tower_27.pt" for tower in ["two", "three"]]
            if all(p.exists() for p in captures) and not (args.root / "visuals_27/metrics.json").exists():
                with (args.root / "render_27.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "tools/render_action_tower_comparison.py",
                            "--baseline",
                            str(captures[0]),
                            "--candidate",
                            str(captures[1]),
                            "--fixture",
                            str(args.fixture),
                            "--baseline-label",
                            "2 towers / step700",
                            "--candidate-label",
                            "3 towers / step700",
                            "--output",
                            str(args.root / "visuals_27"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            counts = tuple(
                len(list((args.root / f"closed_loop_{tower}_tower").glob("shard*/trial_*.json")))
                for tower in ["two", "three"]
            )
            if counts != last:
                subprocess.run(
                    [
                        sys.executable,
                        "tools/summarize_libero_paired_trials.py",
                        "--baseline",
                        str(args.root / "closed_loop_two_tower"),
                        "--candidate",
                        str(args.root / "closed_loop_three_tower"),
                        "--output",
                        str(args.root / "closed_loop_visuals"),
                        "--expected-trials",
                        str(args.expected_trials),
                        "--render",
                    ],
                    check=True,
                )
                report = json.loads((args.root / "closed_loop_visuals/metrics.json").read_text())
                completed = report["paired_trials"] == args.expected_trials
                errors = sum(any(p["errors"]) for p in report["pairs"])
                status = f"{timestamp}：step700闭环完成双塔/三塔 {counts[0]}/{counts[1]} 个trial；已配对 {report['paired_trials']}/{args.expected_trials}，配对子集成功数 {report['successes'][0]}/{report['successes'][1]}，配对运行错误 {errors}。"
                print(status, flush=True)
                with args.record.open("a") as f:
                    f.write("\n- 自动评测进度：" + status + "\n")
                marker = "<!-- STEP700_LIVE_STATUS -->"
                document = args.summary.read_text().split(marker)[0].rstrip()
                document += f"\n\n{marker}\n\n## Step700自动评测进度\n\n{status}\n\n状态：{'全部配对评测完成' if completed else '评测继续运行'}。仅比较相同初始状态的已完成配对子集，不把一侧尚未完成的trial计作失败。详情见 [闭环结果页](../../../outputs/libero_bowl_plate/step700/closed_loop_visuals/index.html) 和 [27窗口离线结果](../../../outputs/libero_bowl_plate/step700/visuals_27/index.html)。\n"
                args.summary.write_text(document)
                last = counts
                if completed:
                    # Stop only this run's servers, after both sides have saved
                    # every trial. Verify command lines to avoid PID reuse.
                    manifests = [args.root / "launch_manifest.json", *args.root.glob("shard1_launch_*.json")]
                    stopped = []
                    for path in manifests:
                        value = json.loads(path.read_text())
                        jobs = value["jobs"] if isinstance(value, dict) else value
                        for job in jobs:
                            if job["kind"] != "server":
                                continue
                            try:
                                cmdline = Path(f"/proc/{job['pid']}/cmdline").read_bytes().decode().replace("\0", " ")
                                if (
                                    "cosmos_framework.scripts.action_policy_server_libero" in cmdline
                                    and str(args.root.resolve()) in cmdline
                                ):
                                    os.kill(job["pid"], signal.SIGTERM)
                                    stopped.append(job["pid"])
                            except (FileNotFoundError, ProcessLookupError):
                                pass
                    (args.root / "evaluation_complete.json").write_text(
                        json.dumps(dict(time_utc=timestamp, stopped_server_pids=stopped, errors=errors), indent=2)
                        + "\n"
                    )
                    return
            failures = 0
        except Exception as exc:
            failures += 1
            print(f"Monitor attempt failed ({failures}): {exc!r}", flush=True)
            if failures >= 3:
                with args.record.open("a") as f:
                    f.write(f"\n- 评测汇总监控失败：{timestamp} {exc!r}；原始任务/文件保留，需检查monitor日志。\n")
                raise
        time.sleep(30)


if __name__ == "__main__":
    main()
