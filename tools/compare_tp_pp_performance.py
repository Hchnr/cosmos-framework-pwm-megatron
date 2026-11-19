# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare matched TP/PP and native FSDP updates after checking actual inputs."""

import argparse
import json
import math
import statistics
from pathlib import Path


def updates(root):
    ranks = {}
    for path in sorted(root.glob("updates.rank*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        indexed = {row["iteration"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Repeated iterations in {path}; separate independent runs")
        ranks[path.name] = indexed
    if len(ranks) != 4 or len({tuple(sorted(rows)) for rows in ranks.values()}) != 1:
        raise ValueError(f"Expected four ranks with the same completed updates: {root}")
    return ranks


def inputs(root, iteration):
    records = {}
    for path in sorted(root.glob(f"replay_plan.iter{iteration:09d}.rank*.json")):
        for row in json.loads(path.read_text()):
            key = tuple(row["slot"])
            if key in records:
                raise ValueError(f"Duplicate logical slot {key} in {root}")
            records[key] = row
    return records


def critical_host_phases(run, iteration):
    """Keep sequential phases on one rank; other ranks' maxima do not add."""
    rank = max(run, key=lambda name: run[name][iteration]["update_seconds"])
    row = run[rank][iteration]
    names = (
        "parallel_prepare_wall_seconds",
        "forward_backward_host_seconds",
        "optimizer_and_sync_host_seconds",
    )
    if not row.get("parallel_preparation") or not all(name in row for name in names):
        return None
    phases = {name: row[name] for name in names}
    if not all(math.isfinite(value) and value >= 0 for value in phases.values()):
        raise ValueError("Nonfinite or negative host phase durations")
    residual = row["update_seconds"] - sum(phases.values())
    return {
        "rank_log": rank,
        "seconds": phases,
        "fraction_of_update": {name: value / row["update_seconds"] for name, value in phases.items()},
        "unassigned_host_seconds": residual,
        "accounting_consistent": residual >= -1e-6,
        "scope": "sequential host intervals including communication and waits; not isolated GPU kernels",
    }


def compare(tp_root, native_root, *, warmup_updates=3, logical_dp=4, reference_microsteps=3, legacy_tp_loss_scale=1):
    if logical_dp < 1 or reference_microsteps < 1:
        raise ValueError("Logical DP and reference microsteps must be positive")
    if not math.isfinite(legacy_tp_loss_scale) or legacy_tp_loss_scale < 1:
        raise ValueError("Legacy loss reporting scale must be finite and at least one")
    tp, native = updates(tp_root), updates(native_root)
    iterations = sorted(next(iter(tp.values())))
    if iterations != sorted(next(iter(native.values()))):
        raise ValueError("The two runs must compare identical iteration windows")
    if warmup_updates < 0 or len(iterations) <= warmup_updates:
        raise ValueError("No measured updates remain after warmup")
    rows, mismatches = [], []
    for index, iteration in enumerate(iterations):
        left, right = inputs(tp_root, iteration - 1), inputs(native_root, iteration - 1)
        expected = {(iteration - 1, micro, rank) for micro in range(reference_microsteps) for rank in range(logical_dp)}
        if set(left) != expected or set(right) != expected:
            raise ValueError(f"Missing or unexpected slots at update {iteration}")
        for slot in sorted(expected):
            for field in (
                "source",
                "samples",
                "image",
                "tokens",
                "sha256",
                "prepared_noise_sha256",
                "prepared_packed_sha256",
            ):
                if (
                    field not in left[slot]
                    or field not in right[slot]
                    or left[slot][field] is None
                    or left[slot][field] != right[slot][field]
                ):
                    mismatches.append({"slot": slot, "field": field})
        durations = [max(rank[iteration]["update_seconds"] for rank in run.values()) for run in (tp, native)]
        if not all(math.isfinite(value) and value > 0 for value in durations):
            raise ValueError("Nonfinite or nonpositive update durations")
        tp_losses = [rank[iteration]["loss"] for rank in tp.values() if "loss" in rank[iteration]]
        native_losses = [rank[iteration]["loss"] for rank in native.values()]
        if not tp_losses or not all(value is not None and math.isfinite(value) for value in tp_losses + native_losses):
            raise ValueError("Missing or nonfinite losses")
        if legacy_tp_loss_scale != 1 and any("loss_reporting" in rank[iteration] for rank in tp.values()):
            raise ValueError("Cannot apply legacy loss reporting correction to versioned metrics")
        tp_loss_reported, native_loss = statistics.mean(tp_losses), statistics.mean(native_losses)
        tp_loss = tp_loss_reported * legacy_tp_loss_scale
        loss_difference = abs(tp_loss - native_loss) / max(abs(native_loss), 1e-20)
        rows.append(
            {
                "iteration": iteration,
                "warmup": index < warmup_updates,
                "logical_slots": len(expected),
                "samples": sum(item["samples"] for item in left.values()),
                "tokens": {
                    name: sum(item["tokens"][name] for item in left.values()) for name in ("text", "vision", "action")
                },
                "tp_update_seconds": durations[0],
                "native_update_seconds": durations[1],
                "tp_critical_rank_host_phases": critical_host_phases(tp, iteration),
                "native_critical_rank_host_phases": critical_host_phases(native, iteration),
                "tp_loss": tp_loss,
                "tp_loss_as_logged": tp_loss_reported,
                "native_loss": native_loss,
                "loss_relative_difference": loss_difference,
                "loss_passed": loss_difference <= 0.01,
            }
        )
    measured = [row for row in rows if not row["warmup"]]
    tp_seconds = sum(row["tp_update_seconds"] for row in measured)
    native_seconds = sum(row["native_update_seconds"] for row in measured)
    matched = not mismatches
    measured_samples = sum(row["samples"] for row in measured)
    measured_tokens = {name: sum(row["tokens"][name] for row in measured) for name in ("text", "vision", "action")}
    return {
        "scope": "four-GPU short-window matched-input comparison; excludes checkpoint writes",
        "inputs_matched": matched,
        "input_mismatches": mismatches,
        "bf16_loss_trajectory_passed": all(row["loss_passed"] for row in rows),
        "warmup_updates": warmup_updates,
        "logical_data_parallel_size": logical_dp,
        "reference_microsteps": reference_microsteps,
        "global_packed_microbatches_per_update": logical_dp * reference_microsteps,
        "legacy_tp_loss_reporting_scale": legacy_tp_loss_scale,
        "measured_updates": len(measured),
        "updates": rows,
        "tp_mean_update_seconds": tp_seconds / len(measured),
        "native_mean_update_seconds": native_seconds / len(measured),
        "matched_tp_throughput_over_native": native_seconds / tp_seconds if matched else None,
        "matched_samples_per_second": {
            "tp": measured_samples / tp_seconds,
            "native": measured_samples / native_seconds,
        }
        if matched
        else None,
        "matched_tokens_per_second": {
            name: {"tp": count / tp_seconds, "native": count / native_seconds}
            for name, count in measured_tokens.items()
        }
        if matched
        else None,
        "production_576_slot_acceptance": False,
        "full_optimizer_state_correctness_accepted": False,
        "192_gpu_performance_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-run", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--logical-dp", type=int, default=4)
    parser.add_argument("--reference-microsteps", type=int, default=3)
    parser.add_argument(
        "--legacy-tp-loss-scale",
        type=float,
        default=1,
        help="Explicit correction for pre-fix logs whose detached loss aliased MCore's in-place scaled loss",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.tp_run,
        args.native_run,
        warmup_updates=args.warmup_updates,
        logical_dp=args.logical_dp,
        reference_microsteps=args.reference_microsteps,
        legacy_tp_loss_scale=args.legacy_tp_loss_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "updates"}))
    if not result["inputs_matched"] or not result["bf16_loss_trajectory_passed"]:
        raise SystemExit("Matched-input/loss gate failed; see complete report")


if __name__ == "__main__":
    main()
