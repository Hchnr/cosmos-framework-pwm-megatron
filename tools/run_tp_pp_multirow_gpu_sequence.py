# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serial tiny CUDA qualification of experimental grouped TP row reductions.

The caller must hold the shared GPU lock and verify exclusive GPU ownership.
The sequence checks TP2/TP4 FP32 baseline/candidate full-AC compiled model
outputs and state, records profiler collective calls per update, and compares
BF16 grouped vs separate rows using a one-layer FP32-master AdamW diagnostic.
No full-model correctness or performance acceptance is claimed.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def load_fp32_case(output, world, mode):
    ranks = []
    for rank in range(world):
        report = json.loads(output.with_name(f"{output.stem}.rank{rank}.json").read_text())
        sidecar = json.loads(output.with_name(f"{output.stem}.multirow_{mode}.rank{rank}.json").read_text())
        gradients = report.get("relative_gradient_l2", {})
        updates = {int(key.split("/", 1)[0]) for key in gradients}
        counts = sidecar.get("functional_all_reduce_calls_per_update")
        if not (
            report.get("passed") is True
            and report.get("compiled") is True
            and updates == {0, 1, 2, 3}
            and sidecar.get("passed") is True
            and sidecar.get("mode") == mode
            and sidecar.get("tp") == world
            and sidecar.get("full_ac_compile") is True
            and sidecar.get("shared_input_reduction") is True
            and sidecar.get("full_model_acceptance") is False
            and isinstance(counts, list)
            and len(counts) == 4
            and all(isinstance(value, int) and value > 0 for value in counts)
        ):
            raise ValueError(f"Missing or invalid FP32 {mode} report for TP{world} rank {rank}")
        ranks.append(
            {
                "rank": rank,
                "counts": counts,
                "peak_cuda_allocated_bytes": sidecar["peak_cuda_allocated_bytes"],
                "max_relative_gradient_l2": max(gradients.values()),
            }
        )
    return ranks


def validate_counts(baseline, candidate):
    differences = []
    for old, new in zip(baseline, candidate, strict=True):
        if old["rank"] != new["rank"]:
            raise ValueError("Ranks differ")
        delta = [a - b for a, b in zip(old["counts"], new["counts"], strict=True)]
        if not all(value > 0 for value in delta):
            raise ValueError(f"Profiler did not observe fewer all-reduces at rank {old['rank']}: {delta}")
        differences.append({"rank": old["rank"], "reduced_calls_per_update": delta})
    return differences


def load_bf16_case(output, world):
    ranks = []
    for rank in range(world):
        report = json.loads((output / f"rank{rank}.json").read_text())
        cases = report.get("cases", [])
        if (
            report.get("rank") != rank
            or len(cases) != 4
            or not all(
                row.get("passed") is True
                and row.get("dtype") == "bf16"
                and row.get("full_activation_checkpointing") is True
                and row.get("compile_backend") == "inductor"
                for row in cases
            )
        ):
            raise ValueError(f"BF16 fixed numerical gate failed or report incomplete for TP{world} rank {rank}")
        ranks.append(
            {
                "rank": rank,
                "max_relative_l2": max(row["max_relative_l2"] for row in cases),
                "peak_cuda_allocated_bytes": report["peak_cuda_bytes"],
            }
        )
    return ranks


def run(command, log, timeout):
    started = time.monotonic()
    with log.open("x") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            reason = None
        except subprocess.TimeoutExpired:
            reason = "qualification timeout"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    return {
        "command": command,
        "pid": process.pid,
        "returncode": code,
        "seconds": time.monotonic() - started,
        "stop_reason": reason,
        "log": str(log),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.case_timeout <= 0:
        raise ValueError("Positive case timeout required")
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.output / "summary.json"
    result = {
        "passed": False,
        "scope": "tiny GPU qualification only; not full-model or throughput acceptance",
        "worlds": {},
    }
    save(summary_path, result)
    executable = sys.executable
    try:
        for world in (2, 4):
            summary = {"passed": False, "cases": {}}
            result["worlds"][str(world)] = summary
            launcher = [executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"]
            ranks_by_mode = {}
            for mode in ("baseline", "candidate"):
                output = args.output / f"tp{world}_{mode}.json"
                command = launcher + [
                    str(ROOT / "tools/qualify_tp_pp_multirow_reduction.py"),
                    "--tp",
                    str(world),
                    "--output",
                    str(output),
                    "--shared-input-reduction",
                    "--profile-collectives",
                    "--compile",
                ]
                if mode == "baseline":
                    command.append("--baseline")
                case = run(command, args.output / f"tp{world}_{mode}.log", args.case_timeout)
                summary["cases"][mode] = case
                save(summary_path, result)
                if case["returncode"] != 0:
                    raise RuntimeError(f"TP{world} {mode} qualification exited {case['returncode']}")
                ranks_by_mode[mode] = load_fp32_case(output, world, mode)
                case["ranks"] = ranks_by_mode[mode]
                save(summary_path, result)
            summary["functional_collective_delta"] = validate_counts(
                ranks_by_mode["baseline"], ranks_by_mode["candidate"]
            )
            bf16_dir = args.output / f"tp{world}_bf16"
            command = launcher + [
                str(ROOT / "tests/megatron/tp_pp_multirow_layer_worker.py"),
                "--worktree",
                str(ROOT),
                "--output",
                str(bf16_dir),
                "--backend",
                "nccl",
                "--dtype",
                "bf16",
                "--compile",
                "--full-ac",
                "--compile-backend",
                "inductor",
            ]
            bf16 = run(command, args.output / f"tp{world}_bf16.log", args.case_timeout)
            summary["cases"]["bf16"] = bf16
            save(summary_path, result)
            if bf16["returncode"] != 0:
                raise RuntimeError(f"TP{world} BF16 fixed-gate comparison exited {bf16['returncode']}")
            bf16["ranks"] = load_bf16_case(bf16_dir, world)
            summary["passed"] = True
            save(summary_path, result)
        result["passed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(summary_path, result)


if __name__ == "__main__":
    main()
