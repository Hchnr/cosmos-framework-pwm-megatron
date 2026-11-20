# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run synthetic MCore overlap NCCL control and failure witnesses serially."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from tools.queue_tp_pp_sharded_vae_gpu import await_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("tp_pp_overlap_unused_repro.py")
    result = {"scope": "synthetic DDP readiness only; not production-model acceptance", "cases": []}
    try:
        for world in (2, 4):
            for overlap in (False, True):
                label = f"{'overlap' if overlap else 'control'}_{world}"
                before = await_idle(timeout=120, poll=2)
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={world}",
                    str(runner),
                    "--output",
                    str(args.output / label),
                ]
                if overlap:
                    cmd.append("--overlap")
                with (args.output / f"{label}.log").open("x") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        raise TimeoutError(f"{label} exceeded {args.timeout} seconds") from None
                if code != 0:
                    raise RuntimeError(f"{label} returned {code}")
                reports = [
                    json.loads(
                        (args.output / label / f"{'overlap' if overlap else 'control'}.rank{rank}.json").read_text()
                    )
                    for rank in range(world)
                ]
                if not all(
                    item["rank"] == rank
                    and item["world_size"] == world
                    and item["overlap"] == overlap
                    and item["transport"] == "real NCCL"
                    and item["expected_behavior_observed"]
                    and item["full_model_acceptance"] is False
                    for rank, item in enumerate(reports)
                ):
                    raise ValueError(f"{label} did not meet the per-rank outcome contract")
                result["cases"].append({"case": label, "returncode": code, "passed": True, "gpu_before": before})
                (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        result["completed"] = True
    except BaseException as exc:
        result["completed"] = False
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
