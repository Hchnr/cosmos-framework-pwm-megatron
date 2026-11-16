#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read-only completion audit for the four Nano8B V2 ratio ablations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
RUN_TABLE = REPO / "examples/v2_perf_ratio_ablation_runs_r02.tsv"
OWNER_FILE = ".v2_perf_ratio_ablation_owner"
KEEP_STEPS = (1500, 2000, 3000, 5000, 8000, 8500)
EXPECTED_BRANCH = "ybai_0828/understanding_verification"
EXPECTED_COMMIT = "e8bbcc0c17d6a890f032c4b7a10a29e7c596137e"
EXPECTED_INITIAL_DCP = "/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP"
EXPECTED_RESOLVED = {
    "optimizer_lr": "2.0e-5",
    "optimizer_weight_decay": "2.0e-7",
    "trainer_max_iter": "8500",
    "trainer_grad_accum_iter": "1",
    "trainer_logging_iter": "1",
    "scheduler_cycle_lengths": "[8500]",
    "scheduler_warm_up_steps": "[200]",
    "checkpoint_save_iter": "500",
    "max_num_tokens_after_packing": "65536",
    "dataloader_max_sequence_length": "65536",
    "max_samples_per_batch": "48",
    "packing_pool_size": "48",
    "train_vlm_visual": "true",
    "seed": "42",
    "dataset_selection_mode": "exact_periodic",
    "wandb_mode": "online",
}
ITER_RE = re.compile(r"iter_([0-9]{9})$")
STEP_PATTERNS = (
    re.compile(r"Iteration ([0-9]+):"),
    re.compile(r"\[RANK 0\] ([0-9]+) : iter_speed"),
    re.compile(r"global_id=([0-9]+)"),
)
ANOMALY_PATTERNS = {
    "nan_inf": re.compile(
        r"Loss:\s*(-?nan|[-+]?inf)|loss[^A-Za-z0-9]*(?:is|=|:)?\s*(-?nan|[-+]?inf)",
        re.IGNORECASE,
    ),
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError|\bOOM\b", re.IGNORECASE),
    "data_error": re.compile(
        r"ValidationError|ConfigKeyError|FileNotFoundError|DataLoader worker.*"
        r"(?:exited unexpectedly|killed)|map-style resume cannot|dataset fingerprint mismatch|"
        r"schedule mismatch",
        re.IGNORECASE,
    ),
    "storage_error": re.compile(
        r"No space left on device|Input/output error|disk quota|ENOSPC", re.IGNORECASE
    ),
    "node_error": re.compile(
        r"NODE_UNREACHABLE|NCCL (?:WARN|ERROR)|NCCL.*(?:error|failed|timed out)|"
        r"ProcessGroupNCCL.*(?:error|failed|timeout)|host unreachable|node failure",
        re.IGNORECASE,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs() -> list[dict[str, str]]:
    runs = []
    for line in RUN_TABLE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 7:
            raise ValueError(f"invalid run-table row: {line!r}")
        experiment, host, toml, job, output, port, session = fields
        runs.append(
            dict(
                experiment=experiment,
                host=host,
                toml=toml,
                job=job,
                output=output,
                port=port,
                session=session,
            )
        )
    if len(runs) != 4:
        raise ValueError(f"expected four runs, found {len(runs)}")
    return runs


def manifest_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value and value != ">-":
            values[key] = value
    return values


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def dcp_complete(checkpoint: Path) -> tuple[bool, str, dict[str, int]]:
    counts: dict[str, int] = {}
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        return False, "missing, non-directory, or symlink", counts
    for component in ("model", "optim", "scheduler", "trainer"):
        component_dir = checkpoint / component
        metadata = component_dir / ".metadata"
        shards = [component_dir / f"__{rank}_0.distcp" for rank in range(8)]
        counts[component] = int(metadata.is_file()) + sum(item.is_file() for item in shards)
        if not metadata.is_file() or metadata.stat().st_size == 0:
            return False, f"{component}/.metadata missing or empty", counts
        for rank, shard in enumerate(shards):
            if not shard.is_file():
                return False, f"{component} rank {rank} shard missing", counts
            if (component != "scheduler" or rank == 0) and shard.stat().st_size == 0:
                return False, f"{component} rank {rank} shard empty", counts
    states = [checkpoint / "dataloader" / f"rank_{rank}.pkl" for rank in range(8)]
    counts["dataloader"] = sum(item.is_file() for item in states)
    for rank, state in enumerate(states):
        if not state.is_file() or state.stat().st_size == 0:
            return False, f"dataloader rank {rank} state missing or empty", counts
    return True, "complete", counts


def scan_logs(paths: list[Path]) -> tuple[int, dict[str, list[str]]]:
    max_step = 0
    matches = {name: [] for name in ANOMALY_PATTERNS}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                for pattern in STEP_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        max_step = max(max_step, int(match.group(1)))
                for name, pattern in ANOMALY_PATTERNS.items():
                    if len(matches[name]) < 20 and pattern.search(line):
                        matches[name].append(f"{path}:{line_number}:{line.rstrip()[:500]}")
    return max_step, matches


def audit_run(run: dict[str, str]) -> tuple[dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    pending: list[str] = []
    output = Path(run["output"]).resolve(strict=False)
    model_dir = output / "cosmos3/understanding_verification" / run["job"]
    checkpoint_root = model_dir / "checkpoints"
    manifest_path = output / "run_manifest.yaml"
    owner_path = output / OWNER_FILE
    expected_parent = (REPO / "outputs/train").resolve(strict=False)
    if output.parent != expected_parent:
        errors.append(f"output root outside exact allowed parent: {output}")
    if not owner_path.is_file() or owner_path.read_text().strip() != run["experiment"]:
        errors.append(f"owner marker missing or mismatched: {owner_path}")
    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")
        manifest: dict[str, str] = {}
    else:
        manifest = manifest_values(manifest_path)

    toml = Path(run["toml"])
    actual_toml_sha = sha256(toml) if toml.is_file() else None
    checks = {
        "experiment": run["experiment"],
        "toml": run["toml"],
        "git_branch": EXPECTED_BRANCH,
        "git_commit": EXPECTED_COMMIT,
        "initial_checkpoint": EXPECTED_INITIAL_DCP,
        "output_root": str(output),
        "model_dir": str(model_dir),
        "checkpoint_dir": str(checkpoint_root),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest {key}: expected {expected!r}, got {manifest.get(key)!r}")
    if manifest.get("toml_sha256") != actual_toml_sha:
        errors.append(
            f"TOML hash drift: manifest={manifest.get('toml_sha256')} actual={actual_toml_sha}"
        )
    for key, expected in EXPECTED_RESOLVED.items():
        if manifest.get(key) != expected:
            errors.append(f"resolved {key}: expected {expected!r}, got {manifest.get(key)!r}")

    latest_path = checkpoint_root / "latest_checkpoint.txt"
    latest = latest_path.read_text().splitlines()[0].strip() if latest_path.is_file() else None
    actual_dirs = sorted(
        item.name for item in checkpoint_root.iterdir() if item.is_dir() and ITER_RE.fullmatch(item.name)
    ) if checkpoint_root.is_dir() else []
    expected_dirs = [f"iter_{step:09d}" for step in KEEP_STEPS]
    dcp_reports = []
    if latest != "iter_000008500":
        pending.append(f"latest is {latest!r}, not iter_000008500")
    if latest == "iter_000008500" and actual_dirs != expected_dirs:
        errors.append(f"final checkpoint set mismatch: {actual_dirs}")
    for name in expected_dirs:
        path = checkpoint_root / name
        ok, reason, counts = dcp_complete(path)
        dcp_reports.append(
            {"step": int(name.removeprefix("iter_")), "path": str(path), "complete": ok,
             "reason": reason, "file_counts": counts}
        )
        if latest == "iter_000008500" and not ok:
            errors.append(f"final milestone incomplete: {path}: {reason}")

    logs = output / "logs"
    train_log = logs / "train.log"
    monitor_log = logs / "monitor.log"
    recovery_log = logs / "recovery.log"
    max_step, anomalies = scan_logs([train_log, monitor_log, recovery_log])
    if max_step < 8500:
        pending.append(f"maximum logged step is {max_step}, below 8500")
    for category, lines in anomalies.items():
        if lines:
            errors.append(f"{category} evidence found in logs ({len(lines)} retained matches)")

    done_path = output / "monitor_state/done_utc"
    done_utc = done_path.read_text().strip() if done_path.is_file() else None
    if not done_utc:
        pending.append("monitor completion timestamp missing")

    retention_log = logs / "checkpoint_retention.log"
    deletion_records: list[dict[str, object]] = []
    delete_intents: set[int] = set()
    if retention_log.is_file():
        for line in retention_log.read_text(errors="replace").splitlines():
            intent = re.search(r" DELETE path=(\S+) step=([0-9]+).*filesystem_before_", line)
            if intent:
                delete_intents.add(int(intent.group(2)))
            deleted = re.search(
                r" DELETED path=(\S+) step=([0-9]+).*filesystem_after_", line
            )
            if deleted:
                deleted_path = Path(deleted.group(1)).resolve(strict=False)
                step = int(deleted.group(2))
                deletion_records.append({"path": str(deleted_path), "step": step})
                if deleted_path.parent != checkpoint_root:
                    errors.append(f"retention deleted outside exact checkpoint root: {deleted_path}")
                if step in KEEP_STEPS:
                    errors.append(f"retention deleted permanent milestone: {deleted_path}")
                if step not in delete_intents:
                    errors.append(f"DELETED record lacks matching before-disk DELETE record: step {step}")

    monitor_text = monitor_log.read_text(errors="replace") if monitor_log.is_file() else ""
    recoveries = [line for line in monitor_text.splitlines() if "RECOVERY_START" in line]
    report: dict[str, object] = {
        "experiment": run["experiment"],
        "host": run["host"],
        "job": run["job"],
        "tmux_session": run["session"],
        "start_utc": manifest.get("start_utc"),
        "done_utc": done_utc,
        "launcher_pid": manifest.get("launcher_pid"),
        "toml": run["toml"],
        "toml_sha256": actual_toml_sha,
        "worktree_diff_sha256": manifest.get("worktree_diff_sha256"),
        "initial_checkpoint": manifest.get("initial_checkpoint"),
        "output_root": str(output),
        "model_dir": str(model_dir),
        "checkpoint_dir": str(checkpoint_root),
        "latest_checkpoint": latest,
        "checkpoint_directories": actual_dirs,
        "permanent_dcps": dcp_reports,
        "max_logged_step": max_step,
        "recoveries": recoveries,
        "deleted_nonmilestones": deletion_records,
        "anomalies": anomalies,
        "logs": {
            "train": str(train_log),
            "monitor": str(monitor_log),
            "retention": str(retention_log),
            "recovery": str(recovery_log),
        },
    }
    return report, errors, pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = load_runs()
    reports = []
    errors: list[str] = []
    pending: list[str] = []
    for run in runs:
        report, run_errors, run_pending = audit_run(run)
        reports.append(report)
        errors.extend(f"{run['experiment']}: {item}" for item in run_errors)
        pending.extend(f"{run['experiment']}: {item}" for item in run_pending)

    starts = [parse_utc(str(report["start_utc"])) for report in reports if report["start_utc"]]
    intervals = [int((right - left).total_seconds()) for left, right in zip(starts, starts[1:])]
    if len(starts) != 4:
        errors.append("not all four manifests contain a valid start_utc")
    elif any(interval < 120 for interval in intervals):
        errors.append(f"stagger interval below 120 seconds: {intervals}")
    if len({str(report["host"]) for report in reports}) != 4:
        errors.append("the four tasks are not mapped to four distinct hosts")
    if len({str(report["initial_checkpoint"]) for report in reports}) != 1:
        errors.append("initial checkpoints differ")
    if len({str(report["toml"]) for report in reports}) != 4:
        errors.append("TOML paths are not unique")
    if len({str(report["checkpoint_dir"]) for report in reports}) != 4:
        errors.append("checkpoint directories are not unique")

    status = "FAIL" if errors else ("IN_PROGRESS" if pending else "PASS")
    result = {
        "audit_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "launch_intervals_seconds": intervals,
        "errors": errors,
        "pending": pending,
        "runs": reports,
        "scope_note": (
            "No evaluation, comparison, winner selection, baseline rerun, or fifth experiment is "
            "performed by this read-only audit."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.resolve(strict=False)
        allowed_parent = (REPO / "outputs/train").resolve(strict=False)
        if output.parent != allowed_parent:
            raise ValueError(f"audit output must be directly under {allowed_parent}: {output}")
        output.write_text(rendered)
    sys.stdout.write(rendered)
    if errors:
        return 1
    if pending and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
