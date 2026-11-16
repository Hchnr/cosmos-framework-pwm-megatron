#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Safe one-pass DCP retention for one registered ratio-ablation run."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
RUN_TABLE = REPO / "examples/v2_perf_ratio_ablation_runs_r02.tsv"
OWNER_FILE = ".v2_perf_ratio_ablation_owner"
KEEP_STEPS = {1500, 2000, 3000, 5000, 8000, 8500}
ITER_RE = re.compile(r"iter_([0-9]{9})")


def load_run(experiment: str) -> dict[str, str]:
    matches = []
    for line in RUN_TABLE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if fields[0] == experiment:
            matches.append(fields)
    if len(matches) != 1:
        raise ValueError(f"experiment must be registered exactly once: {experiment}")
    experiment, host, toml, job, output, port, session = matches[0]
    return {
        "experiment": experiment,
        "host": host,
        "toml": toml,
        "job": job,
        "output": output,
        "port": port,
        "session": session,
    }


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def complete(checkpoint: Path) -> tuple[bool, str]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        return False, "missing, non-directory, or symlink"
    for component in ("model", "optim", "scheduler", "trainer"):
        metadata = checkpoint / component / ".metadata"
        if not metadata.is_file() or metadata.stat().st_size == 0:
            return False, f"{component}/.metadata missing or empty"
        for rank in range(8):
            shard = checkpoint / component / f"__{rank}_0.distcp"
            if not shard.is_file():
                return False, f"{component} rank {rank} shard missing"
            if component != "scheduler" or rank == 0:
                if shard.stat().st_size == 0:
                    return False, f"{component} rank {rank} shard empty"
    for rank in range(8):
        state = checkpoint / "dataloader" / f"rank_{rank}.pkl"
        if not state.is_file() or state.stat().st_size == 0:
            return False, f"dataloader rank {rank} state missing or empty"
    return True, "complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = load_run(args.experiment)

    output_root = Path(run["output"]).resolve(strict=False)
    allowed_parent = (REPO / "outputs/train").resolve(strict=False)
    if output_root.parent != allowed_parent:
        raise ValueError(f"output root is outside exact allowed parent: {output_root}")
    owner = output_root / OWNER_FILE
    if not owner.is_file() or owner.read_text().strip() != args.experiment:
        raise ValueError(f"ownership marker missing or mismatched: {output_root}")

    model_dir = output_root / "cosmos3/understanding_verification" / run["job"]
    checkpoint_root = (model_dir / "checkpoints").resolve(strict=False)
    if checkpoint_root != model_dir / "checkpoints":
        raise ValueError(f"non-canonical checkpoint root: {checkpoint_root}")
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "checkpoint_retention.log"
    lock_path = log_dir / ".checkpoint_retention.lock"

    def log(message: str) -> None:
        line = f">>> {timestamp()} [{args.experiment}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if not checkpoint_root.is_dir():
            log("checkpoint root not created yet")
            return 0
        latest_file = checkpoint_root / "latest_checkpoint.txt"
        if not latest_file.is_file():
            log("latest pointer not created yet")
            return 0
        latest_name = latest_file.read_text().splitlines()[0].strip()
        match = ITER_RE.fullmatch(latest_name)
        if not match:
            log(f"BLOCKED invalid latest pointer: {latest_name!r}")
            return 1
        latest_step = int(match.group(1))
        latest_dir = (checkpoint_root / latest_name).resolve(strict=False)
        if latest_dir != checkpoint_root / latest_name:
            log(f"BLOCKED non-canonical latest path: {latest_dir}")
            return 1
        latest_ok, latest_reason = complete(latest_dir)
        if not latest_ok:
            log(f"latest pointer targets incomplete checkpoint; no deletion: {latest_name}: {latest_reason}")
            return 0
        if args.check_only:
            log(f"CHECK complete latest={latest_name}")
            return 0

        candidates = sorted(path for path in checkpoint_root.iterdir() if ITER_RE.fullmatch(path.name))
        for candidate in candidates:
            match = ITER_RE.fullmatch(candidate.name)
            assert match is not None
            step = int(match.group(1))
            if step >= latest_step or step in KEEP_STEPS:
                continue
            candidate_real = candidate.resolve(strict=False)
            if candidate_real != checkpoint_root / candidate.name or candidate.is_symlink():
                log(f"BLOCKED refusing non-canonical candidate: {candidate}")
                continue
            candidate_ok, candidate_reason = complete(candidate_real)
            if not candidate_ok:
                log(f"SKIP incomplete older checkpoint (never delete): {candidate_real}: {candidate_reason}")
                continue
            before_bytes = directory_bytes(candidate_real)
            before_disk = shutil.disk_usage(output_root)
            log(f"DELETE path={candidate_real} step={step} bytes={before_bytes} "
                f"filesystem_before_used={before_disk.used} filesystem_before_free={before_disk.free} "
                f"latest={latest_name}")
            latest_now = latest_file.read_text().splitlines()[0].strip()
            latest_now_ok, _ = complete(checkpoint_root / latest_now)
            if latest_now != latest_name or not latest_now_ok:
                log("BLOCKED latest pointer changed or became incomplete before deletion")
                return 1
            shutil.rmtree(candidate_real)
            after_disk = shutil.disk_usage(output_root)
            log(f"DELETED path={candidate_real} step={step} "
                f"filesystem_after_used={after_disk.used} filesystem_after_free={after_disk.free}")

        if latest_step == 8500:
            expected = {f"iter_{step:09d}" for step in KEEP_STEPS}
            actual = {path.name for path in checkpoint_root.iterdir() if ITER_RE.fullmatch(path.name)}
            if actual != expected:
                log(f"BLOCKED final checkpoint set mismatch actual={sorted(actual)} expected={sorted(expected)}")
                return 1
            for name in sorted(expected):
                ok, reason = complete(checkpoint_root / name)
                if not ok:
                    log(f"BLOCKED final milestone incomplete: {name}: {reason}")
                    return 1
            log(f"FINAL_OK latest=iter_000008500 retained={sorted(actual)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
