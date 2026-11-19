# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run two synthetic production loss/loader qualifications on NCCL serially.

Call only after the shared GPU queue owns the lock and confirms four idle GPUs.
The report is not full-model, real-dataset or training-schedule acceptance.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil

SNAPSHOT = Path(__file__).resolve().parents[1]
EXPECTED_TOPOLOGIES = ("tp2pp1dp2", "tp2pp2dp1")
EXPECTED_CASES = {
    "multiple vision items",
    "absent vision dummy",
    "no noisy vision",
    "vision_gen false",
    "uncommitted mismatch",
    "loss",
    "gradient",
    "RNG",
}


def source_digest():
    digest = hashlib.sha256()
    paths = sorted(
        path
        for base in (SNAPSHOT / "cosmos_framework", SNAPSHOT / "tests", SNAPSHOT / "tools")
        for path in base.rglob("*.py")
        if path.is_file()
    )
    for path in paths:
        digest.update(str(path.relative_to(SNAPSHOT)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"python_files": len(paths), "sha256": digest.hexdigest()}


def check_reports(directory, topology):
    reports = []
    for rank in range(4):
        data = json.loads((directory / f"{topology}.rank{rank}.json").read_text())
        if not (
            data.get("passed") is True
            and data.get("topology") == topology
            and data.get("rank") == rank
            and data.get("backend") == "nccl"
            and data.get("production_logical_rng") is True
            and set(data.get("covered", [])) == EXPECTED_CASES
            and [case.get("vision_gen") for case in data.get("cases", [])] == [True, False]
        ):
            raise ValueError(f"Incomplete or failing rank {rank} report for {topology}")
        reports.append(data)
    return reports


def stop_tree(process):
    """Only signal the child and descendants launched by this sequence."""
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in reversed(children):
            child.send_signal(signal.SIGTERM)
        root.send_signal(signal.SIGTERM)
        _, remaining = psutil.wait_procs([root, *children], timeout=20)
        for child in remaining:
            child.kill()
        psutil.wait_procs(remaining, timeout=10)
    except psutil.NoSuchProcess:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-digest", type=Path, required=True)
    parser.add_argument("--case-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.case_timeout < 1:
        raise ValueError("case-timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "scope": "synthetic production preflight, decoder, loss, RNG and loader on NCCL",
        "full_model_acceptance": False,
        "cases": {},
    }
    try:
        report["source_before"] = source_digest()
        expected = json.loads(args.expected_source_digest.read_text())
        if report["source_before"] != expected:
            raise RuntimeError("Qualification snapshot changed while queued")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(SNAPSHOT), env.get("PYTHONPATH", "")))
        for topology in EXPECTED_TOPOLOGIES:
            case_dir = args.output / topology
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                str(SNAPSHOT / "tools/qualify_tp_pp_sample_count_distributed.py"),
                "--backend",
                "nccl",
                "--topology",
                topology,
                "--output",
                str(case_dir),
            ]
            case = {"command": command, "passed": False}
            report["cases"][topology] = case
            with (args.output / f"{topology}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=SNAPSHOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    case["returncode"] = process.wait(timeout=args.case_timeout)
                except subprocess.TimeoutExpired:
                    stop_tree(process)
                    raise TimeoutError(f"{topology} exceeded {args.case_timeout}s") from None
            if case["returncode"]:
                raise RuntimeError(f"{topology} returned {case['returncode']}")
            reports = check_reports(case_dir, topology)
            case["passed"] = True
            case["rank_results"] = [{"rank": row["rank"], "cases": row["cases"]} for row in reports]
        report["source_after"] = source_digest()
        if report["source_before"] != report["source_after"]:
            raise RuntimeError("Qualification snapshot changed during the run")
        report["passed"] = True
    except BaseException as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (args.output / "summary.json").open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
