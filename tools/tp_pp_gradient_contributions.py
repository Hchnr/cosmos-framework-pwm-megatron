# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Analyze the parameter-level contributions to a recorded TP gradient error.

The input must be a complete FP32 TP-only observation with reference squared
norms. Historical reports without those norms cannot be analyzed this way.
The caller must supply one report per rank and the independently pinned
completed coordinator status from the same run.
"""

import argparse
import hashlib
import json
import math
import uuid
from collections import defaultdict
from pathlib import Path


def _finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def analyze(manifest, reports, *, manifest_sha256):
    """Reconstruct squared-error contributions; fail if the global metric disagrees."""
    tp = manifest.get("tp")
    if type(tp) is not int or tp < 1 or manifest.get("pp") != 1 or len(reports) != tp:
        raise ValueError("a complete TP-only report set and PP=1 manifest are required")
    _sha256(manifest_sha256, "bundle manifest SHA256")
    entries = manifest.get("parameters")
    if not isinstance(entries, dict):
        raise ValueError("parameter manifest is required")
    trainable = {name: entry for name, entry in entries.items() if entry.get("optimizer_fields")}
    if not trainable or any(
        "axis" not in entry or (entry["axis"] is not None and (type(entry["axis"]) is not int or entry["axis"] < 0))
        for entry in trainable.values()
    ):
        raise ValueError("trainable parameter layouts are missing or invalid")

    updates = reports[0].get("updates")
    if type(updates) is not int or updates < 1:
        raise ValueError("update count is missing or invalid")
    run_id = reports[0].get("run_id")
    try:
        if not isinstance(run_id, str) or uuid.UUID(run_id).version != 4:
            raise ValueError("invalid run ID")
    except (ValueError, AttributeError) as error:
        raise ValueError("a unique UUID4 run ID is required") from error
    source_sha = _sha256(reports[0].get("source_manifest_sha256"), "source manifest SHA256")
    native_sha = _sha256(reports[0].get("native_status_sha256"), "native status SHA256")
    baseline_sha = reports[0].get("baseline_status_sha256")
    if baseline_sha is not None:
        _sha256(baseline_sha, "baseline status SHA256")
    rows_by_update = [defaultdict(list) for _ in range(updates)]
    summaries = []
    ranks = set()
    for report in reports:
        rank = report.get("rank")
        if type(rank) is not int or not 0 <= rank < tp or rank in ranks:
            raise ValueError("report rank coverage is duplicated or invalid")
        ranks.add(rank)
        if (
            report.get("tensor_parallel_size") != tp
            or report.get("pipeline_parallel_size") != 1
            or report.get("data_parallel_size") != 1
            or report.get("world_size") != tp
            or report.get("updates") != updates
            or report.get("gradient_scope") != "pre_clip"
            or report.get("run_id") != run_id
            or report.get("source_manifest_sha256") != source_sha
            or report.get("native_status_sha256") != native_sha
            or report.get("baseline_status_sha256") != baseline_sha
            or report.get("bundle_manifest_sha256") != manifest_sha256
        ):
            raise ValueError(f"rank {rank} has a different run, provenance, parallelism or gradient scope")
        gradient_summaries = report.get("gradient_summary")
        rows = report.get("parameter_gradient_errors")
        if not isinstance(gradient_summaries, list) or len(gradient_summaries) != updates or not isinstance(rows, list):
            raise ValueError(f"rank {rank} is missing gradient observations")
        summaries.append((rank, gradient_summaries))
        seen = set()
        for row in rows:
            update, name = row.get("update"), row.get("name")
            if type(update) is not int or not 0 <= update < updates or name not in trainable or (update, name) in seen:
                raise ValueError(f"rank {rank} has invalid or duplicate parameter coverage")
            if any(
                row.get(field) != "torch.float32"
                for field in ("gradient_dtype", "reference_dtype", "model_parameter_dtype")
            ):
                raise ValueError(f"rank {rank} has missing or non-FP32 gradient/parameter evidence")
            if row.get("nonfinite", False) is not False:
                raise ValueError(f"rank {rank} reported a non-finite parameter comparison")
            seen.add((update, name))
            reference = _finite_nonnegative(row.get("reference_squared_l2"), "reference squared L2")
            relative = _finite_nonnegative(row.get("relative_l2"), "relative L2")
            error = relative * relative * max(reference, 1e-40)
            if not math.isfinite(error):
                raise ValueError("reconstructed squared error overflowed")
            if relative and not error:
                raise ValueError("reconstructed squared error underflowed")
            rows_by_update[update][name].append((reference, error))
        if len(seen) != updates * len(trainable):
            raise ValueError(f"rank {rank} does not cover all trainable parameters and updates")

    result = []
    for update, parameters in enumerate(rows_by_update):
        ref_terms, error_terms, contributions = [], [], []
        for name, values in parameters.items():
            replicated = trainable[name]["axis"] is None
            divisor = tp if replicated else 1
            ref_terms.extend(reference / divisor for reference, _ in values)
            error = math.fsum(error / divisor for _, error in values)
            error_terms.append(error)
            contributions.append({"name": name, "squared_error": error})
        reference, error = math.fsum(ref_terms), math.fsum(error_terms)
        if reference <= 0:
            raise ValueError("global reference gradient has zero norm")
        reconstructed = math.sqrt(error / reference)
        for rank, summary in summaries:
            recorded = _finite_nonnegative(summary[update].get("relative_l2"), "recorded global relative L2")
            if (not reconstructed) != (not recorded) or not math.isclose(
                reconstructed, recorded, rel_tol=1e-6, abs_tol=0
            ):
                raise ValueError(f"rank {rank} update {update}: parameter contributions do not reproduce global L2")
        for item in contributions:
            item["fraction_of_squared_error"] = item["squared_error"] / error if error else None
        contributions.sort(key=lambda item: item["squared_error"], reverse=True)
        result.append(
            {
                "update": update,
                "global_relative_l2": reconstructed,
                "reference_squared_l2": reference,
                "error_squared_l2": error,
                "zero_total_error": error == 0,
                "parameter_contributions": contributions,
            }
        )
    return {
        "scope": "recorded pre-clip FP32 gradient; TP-only; reference norms and layout required",
        "run_id": run_id,
        "source_manifest_sha256": source_sha,
        "native_status_sha256": native_sha,
        "baseline_status_sha256": baseline_sha,
        "bundle_manifest_sha256": manifest_sha256,
        "limitation": (
            "The caller must verify the coordinator and its input/reference evidence. "
            "Replicated contributions are weighted by 1/TP, as in the original metric; reconstructed sums are "
            "rounded. Parameter-level shares cannot establish individual-element contributions."
        ),
        "updates": result,
    }


def verify_completed_status(status, reports, report_hashes, *, manifest_sha256):
    """Bind all rank report bytes to the completed, posthashed coordinator."""
    if not isinstance(status, dict) or not reports or len(reports) != len(report_hashes):
        raise ValueError("completed coordinator status and all report hashes are required")
    first = reports[0]
    if (
        status.get("phase") != "tp_observation_completed"
        or status.get("source_input_posthash_passed") is not True
        or status.get("run_id") != first.get("run_id")
        or status.get("source_manifest_sha256") != first.get("source_manifest_sha256")
        or status.get("native_status_sha256") != first.get("native_status_sha256")
        or status.get("bundle_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("coordinator status does not identify this completed, posthashed run")
    hashes = status.get("report_sha256_by_rank")
    if not isinstance(hashes, dict) or len(hashes) != len(reports):
        raise ValueError("coordinator rank report manifest is incomplete")
    for report, digest in zip(reports, report_hashes, strict=True):
        _sha256(digest, "rank report SHA256")
        if hashes.get(str(report.get("rank"))) != digest:
            raise ValueError("rank report bytes do not match coordinator status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--status-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({path.resolve() for path in args.reports}) != len(args.reports):
        parser.error("supply a different report file for each rank")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    report_bytes = [path.read_bytes() for path in args.reports]
    reports = [json.loads(raw) for raw in report_bytes]
    status_bytes = args.status.read_bytes()
    _sha256(args.status_sha256, "coordinator status SHA256")
    if hashlib.sha256(status_bytes).hexdigest() != args.status_sha256:
        raise ValueError("coordinator status bytes do not match the independently supplied SHA256")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    result = analyze(manifest, reports, manifest_sha256=manifest_sha256)
    status = json.loads(status_bytes)
    verify_completed_status(
        status,
        reports,
        [hashlib.sha256(raw).hexdigest() for raw in report_bytes],
        manifest_sha256=manifest_sha256,
    )
    result["coordinator_status_sha256"] = args.status_sha256
    result["report_sha256_by_rank"] = status["report_sha256_by_rank"]
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
