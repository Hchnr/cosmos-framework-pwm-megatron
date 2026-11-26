# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize the full-source FP32 diagnostic's rank JSON reports."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

GRADIENT_RELATIVE_L2_LIMIT = 1e-4
GRADIENT_COSINE_MINIMUM = 0.999999
OPTIMIZER_TENSOR_FIELDS = ("master", "exp_avg", "exp_avg_sq")


def verdict(failed, covered):
    if failed:
        return "failed"
    return "passed" if covered else "unverified"


def checks(row, update, field):
    return [item for item in row.get(field, []) if item["update"] == update]


def all_passed(rows, field="passed"):
    return all(item[field] is True for item in rows)


def finite_or_none(value):
    return value if math.isfinite(value) else None


def summarize_rank_reports(reports):
    if not reports:
        raise ValueError("No rank reports")
    header_fields = (
        "scope",
        "objective",
        "updates",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "replay_native_state",
        "continuous_trajectory",
        "runtime_constants_sha256",
        "seed",
    )
    reference = reports[0]
    header = {field: reference.get(field) for field in header_fields}
    for rank, row in enumerate(reports):
        if row.get("rank", rank) != rank or any(row.get(field) != header[field] for field in header_fields):
            raise ValueError(f"Rank {rank} has inconsistent run metadata")
    tp, pp = header["tensor_parallel_size"], header["pipeline_parallel_size"]
    if type(tp) is not int or type(pp) is not int or (tp, pp) not in {(1, 2), (2, 2), (4, 1)}:
        raise ValueError("Expected topology metadata from tp_pp_full_source_math")
    if len(reports) != tp * pp:
        raise ValueError(f"Got {len(reports)} rank reports for TP{tp}/PP{pp}")
    if header["updates"] != 3:
        raise ValueError("Expected the full-source diagnostic's three updates")
    if type(header["replay_native_state"]) is not bool or header["continuous_trajectory"] is not (
        not header["replay_native_state"]
    ):
        raise ValueError("Inconsistent update mode")
    for row in reports:
        if len(row.get("gradient_summary", [])) != header["updates"]:
            raise ValueError("Every rank must report each collective gradient summary")
        if row["gradient_summary"] != reference["gradient_summary"]:
            raise ValueError("Ranks disagree on the collective gradient summary")

    updates = []
    for update, gradient in enumerate(reference["gradient_summary"]):
        outputs = []
        gradient_count = 0
        failed_gradients = 0
        backward_covered = True
        backward_failed = False
        optimizer_failed = False
        optimizer_covered = True
        optimizer_fields = Counter()
        failed_optimizer_fields = Counter()

        for row in reports:
            outputs.extend(checks(row, update, "output_errors"))
            grads = checks(row, update, "parameter_gradient_errors")
            states = checks(row, update, "parameter_state_errors")
            compute = checks(row, update, "compute_parameter_checks")
            norms = checks(row, update, "norm_checks")
            steps = checks(row, update, "step_checks")

            parameter_counts = Counter(item["name"] for item in grads)
            state_counts = Counter((item["name"], item["field"]) for item in states)
            gradient_count += len(grads)
            failed_gradients += sum(item["passed"] is not True for item in grads)
            backward_covered &= bool(grads) and all(count == 1 for count in parameter_counts.values())
            backward_failed |= not all_passed(grads)
            optimizer_failed |= not all_passed(states + steps + norms) or not all_passed(
                compute, "compute_equals_master"
            )
            optimizer_fields.update(item["field"] for item in states)
            failed_optimizer_fields.update(item["field"] for item in states if item["passed"] is not True)
            optimizer_covered &= (
                bool(grads)
                and len(norms) == 1
                and Counter(item["name"] for item in steps) == parameter_counts
                and Counter(item["name"] for item in compute) == parameter_counts
                and state_counts
                == Counter(
                    {
                        (name, field): count
                        for name, count in parameter_counts.items()
                        for field in OPTIMIZER_TENSOR_FIELDS
                    }
                )
            )

        relative, cosine = gradient["relative_l2"], gradient["cosine"]
        aggregate_passed = (
            math.isfinite(relative)
            and math.isfinite(cosine)
            and 0 <= relative <= GRADIENT_RELATIVE_L2_LIMIT
            and cosine >= GRADIENT_COSINE_MINIMUM
        )
        gradient_failed = backward_failed or not aggregate_passed
        updates.append(
            {
                "update": update,
                "forward": {
                    "status": verdict(not all_passed(outputs), covered=False),
                    "prediction_check_status": verdict(not all_passed(outputs), bool(outputs)),
                    "prediction_checks": len(outputs),
                    "failed_prediction_checks": sum(item["passed"] is not True for item in outputs),
                    "coverage": "prediction tensors only; layer, input, and loss observations are separate",
                },
                "backward": {
                    "status": verdict(gradient_failed, covered=False),
                    "preclip_parameter_check_status": verdict(gradient_failed, backward_covered),
                    "parameter_shard_checks": gradient_count,
                    "failed_parameter_shard_checks": failed_gradients,
                    "relative_l2": finite_or_none(relative),
                    "cosine": finite_or_none(cosine),
                    "aggregate_passed": aggregate_passed,
                    "coverage": "reported pre-clip parameter gradient shards and canonical aggregate",
                },
                "optimizer": {
                    "status": verdict(optimizer_failed, covered=False),
                    "recorded_update_check_status": verdict(optimizer_failed, optimizer_covered),
                    "tensor_checks": dict(optimizer_fields),
                    "failed_tensor_checks": dict(failed_optimizer_fields),
                    "recorded_update_checks_complete": optimizer_covered,
                    "coverage": "reported compute/master, moments, norm, and step; same-gradient isolation is separate",
                },
            }
        )
    continuous = header["continuous_trajectory"]
    earlier_failure = any(
        update[gate]["status"] == "failed" for update in updates for gate in ("forward", "backward", "optimizer")
    )
    return {
        "format": "cosmos_tp_pp_fp32_differential_summary",
        "run": header,
        "updates": updates,
        "training": {
            "status": verdict(continuous and earlier_failure, covered=False),
            "continuous_updates": len(updates) if continuous else 0,
            "restart_checked": False,
            "coverage": "full-source diagnostic; complete input/RNG/restart evidence is separate",
        },
        "limits": {"gradient_relative_l2": GRADIENT_RELATIVE_L2_LIMIT, "gradient_cosine": GRADIENT_COSINE_MINIMUM},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="Prefix of the .rankN.json files")
    parser.add_argument("--world-size", choices=(2, 4), type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [args.prefix.with_name(f"{args.prefix.name}.rank{rank}.json") for rank in range(args.world_size)]
    artifacts = [file.read_bytes() for file in files]
    report = summarize_rank_reports([json.loads(artifact) for artifact in artifacts])
    report["artifacts"] = [
        {"name": file.name, "sha256": hashlib.sha256(artifact).hexdigest()}
        for file, artifact in zip(files, artifacts, strict=True)
    ]
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
