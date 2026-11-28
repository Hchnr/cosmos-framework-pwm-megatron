# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize the full-source FP32 diagnostic's rank JSON reports."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
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
    return value if value is not None and math.isfinite(value) else None


def reject_nonfinite(value):
    """Reject nonfinite numeric evidence anywhere in a rank report."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite numeric evidence in rank report")
    if isinstance(value, dict):
        for item in value.values():
            reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_nonfinite(item)


def has_finite_number(row, name):
    value = row.get(name)
    return type(value) in (int, float) and math.isfinite(value)


def tensor_check(row, *, gradient=False):
    """Return failed, passed or unverified using the producer's tensor tolerance."""
    if row.get("passed") is not True or row.get("nonfinite") is True:
        return "failed"
    relative_key = "comparison_relative_l2" if gradient else "relative_l2"
    maximum_key = "comparison_maximum_absolute_error" if gradient else "maximum_absolute_error"
    if not has_finite_number(row, "reference_squared_l2") or not has_finite_number(row, maximum_key):
        return "unverified"
    squared = row["reference_squared_l2"]
    maximum = row[maximum_key]
    if squared < 0 or maximum < 0 or (has_finite_number(row, relative_key) and row[relative_key] < 0):
        return "failed"
    if gradient:
        if not has_finite_number(row, "relative_l2") or not has_finite_number(row, "maximum_absolute_error"):
            return "unverified"
        if row["relative_l2"] < 0 or row["maximum_absolute_error"] < 0:
            return "failed"
        if squared > 1e-16 and row["relative_l2"] > 1e-4:
            return "failed"
        if squared <= 1e-16 and row["maximum_absolute_error"] > 2e-6:
            return "failed"
    if squared <= 1e-16:
        return "passed" if maximum <= 2e-6 else "failed"
    if not has_finite_number(row, relative_key):
        return "unverified"
    relative = row[relative_key]
    return "passed" if 0 <= relative <= 1e-4 else "failed"


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def parallel_coordinates(reports, pp, tp, dp):
    coordinates = []
    for rank, row in enumerate(reports):
        reported = row.get("parallel_ranks", {})
        group = tuple(reported.get(key) for key in ("pp", "tp", "dp"))
        if any(
            type(value) is not int or not 0 <= value < size for value, size in zip(group, (pp, tp, dp), strict=True)
        ):
            raise ValueError(f"Rank {rank} has invalid parallel rank coordinates")
        coordinates.append(group)
    if set(coordinates) != set(product(range(pp), range(tp), range(dp))):
        raise ValueError("Incomplete or duplicate parallel rank coordinates")
    return coordinates


def verify_parameter_coverage(reports, coordinates, bundle_bytes, tp, pp):
    """Verify that each canonical TP shard is fully and uniquely owned across DP."""
    digest = hashlib.sha256(bundle_bytes).hexdigest()
    bundle = json.loads(bundle_bytes)
    if (
        bundle.get("format") != "cosmos_tp_pp_canonical"
        or bundle.get("format_version") != 1
        or bundle.get("complete") is not True
        or bundle.get("pp") != pp
        or bundle.get("tp") != (2 if tp == 1 else tp)
    ):
        raise ValueError("Canonical bundle topology or format differs")
    layouts = {}
    for name, entry in bundle["parameters"].items():
        if not entry["optimizer_fields"]:
            continue
        shape = list(entry["global_shape"])
        axis, stage = entry["axis"], entry["pipeline_rank"]
        if not 0 <= stage < pp or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError(f"Invalid canonical shape or stage for {name}")
        if axis is not None:
            if not 0 <= axis < len(shape) or shape[axis] % tp:
                raise ValueError(f"Invalid canonical axis or split for {name}")
            shape[axis] //= tp
        for shard in range(tp):
            offset = 0 if axis is None else shape[axis] * shard
            layouts[stage, shard, name] = (axis, offset, shape)
    if not layouts:
        raise ValueError("No canonical trainable parameters")

    names_by_rank = []
    ranges = defaultdict(list)
    for rank, (row, (stage, shard, _)) in enumerate(zip(reports, coordinates, strict=True)):
        if row.get("canonical_bundle_manifest_sha256") != digest:
            raise ValueError(f"Rank {rank} did not record the supplied canonical bundle SHA")
        local = set()
        for fragment in row.get("owned_parameter_fragments", []):
            name = fragment["name"]
            layout = layouts.get((stage, shard, name))
            if name in local or layout != (fragment["axis"], fragment["offset"], fragment["local_shape"]):
                raise ValueError(f"Rank {rank} has a duplicate or incorrect parameter layout for {name}")
            start, end = fragment["start"], fragment["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= math.prod(layout[2]):
                raise ValueError(f"Rank {rank} has an invalid optimizer interval for {name}")
            local.add(name)
            ranges[stage, shard, name].append((start, end))
        names_by_rank.append(local)
    for identity, (_, _, shape) in layouts.items():
        covered = 0
        for start, end in sorted(ranges[identity]):
            if start != covered:
                raise ValueError(f"Missing or overlapping optimizer intervals for {identity}")
            covered = end
        if covered != math.prod(shape):
            raise ValueError(f"Incomplete optimizer interval coverage for {identity}")
    return names_by_rank, {
        "status": "passed",
        "bundle_manifest_sha256": digest,
        "canonical_trainable_parameters": len({name for _, _, name in layouts}),
        "tp_parameter_layouts": len(layouts),
        "optimizer_intervals": sum(len(names) for names in names_by_rank),
    }


def summarize_rank_reports(reports, *, data_parallel_size=None, bundle_bytes=None):
    if not reports:
        raise ValueError("No rank reports")
    header_fields = (
        "scope",
        "objective",
        "updates",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "replay_native_state",
        "continuous_trajectory",
        "runtime_constants_sha256",
        "seed",
        "fingerprint_version",
        "prepared_input_hashes",
        "reference_prediction_counts",
        "microbatches_per_dp_replica",
    )
    reference = reports[0]
    header = {field: reference.get(field) for field in header_fields}
    for rank, row in enumerate(reports):
        reject_nonfinite(row)
        if row.get("rank", rank) != rank or any(row.get(field) != header[field] for field in header_fields):
            raise ValueError(f"Rank {rank} has inconsistent run metadata")
    tp = header["tensor_parallel_size"]
    pp = header["pipeline_parallel_size"]
    reported_dp = header["data_parallel_size"]
    if reported_dp is not None and data_parallel_size is not None and reported_dp != data_parallel_size:
        raise ValueError("Data-parallel size disagrees with the rank reports")
    dp = reported_dp if data_parallel_size is None else data_parallel_size
    if dp is None:
        dp = 1
    if any(type(size) is not int or size < 1 for size in (tp, pp, dp)):
        raise ValueError("Expected positive topology sizes")
    if len(reports) != tp * pp * dp:
        raise ValueError(f"Got {len(reports)} rank reports for TP{tp}/PP{pp}/DP{dp}")
    header["data_parallel_size"] = dp
    input_hashes = header["prepared_input_hashes"]
    prediction_counts = header["reference_prediction_counts"]
    fingerprint_version = header["fingerprint_version"]
    if fingerprint_version is not None and (type(fingerprint_version) is not int or fingerprint_version not in (1, 2)):
        raise ValueError("Unsupported fingerprint version")
    coordinates = None
    if bundle_bytes is not None or input_hashes is not None:
        coordinates = parallel_coordinates(reports, pp, tp, dp)
    verified_names, parameter_coverage = (None, {"status": "unverified"})
    if bundle_bytes is not None:
        verified_names, parameter_coverage = verify_parameter_coverage(reports, coordinates, bundle_bytes, tp, pp)
    if header["updates"] != 3:
        raise ValueError("Expected the full-source diagnostic's three updates")
    if header["runtime_constants_sha256"] is not None and not is_sha256(header["runtime_constants_sha256"]):
        raise ValueError("Invalid runtime constants SHA-256")
    if input_hashes is not None and (
        not isinstance(input_hashes, list)
        or type(header["microbatches_per_dp_replica"]) is not int
        or len(input_hashes) != dp * header["microbatches_per_dp_replica"]
        or not input_hashes
        or any(not is_sha256(value) for value in input_hashes)
    ):
        raise ValueError("Prepared input count differs from the physical DP schedule")
    if prediction_counts is not None and (
        input_hashes is None
        or len(prediction_counts) != header["updates"]
        or any(len(slots) != len(input_hashes) for slots in prediction_counts)
    ):
        raise ValueError("Reference prediction counts differ from the physical DP schedule")
    replay = header["replay_native_state"]
    continuous = header["continuous_trajectory"]
    if (replay is not None or continuous is not None) and (type(replay) is not bool or continuous is not (not replay)):
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
        usage_covered = True
        usage_failed = False
        optimizer_failed = False
        optimizer_covered = True
        optimizer_fields = Counter()
        failed_optimizer_fields = Counter()
        loss_checks = []
        output_proofs = []

        for rank, row in enumerate(reports):
            rank_outputs = checks(row, update, "output_errors")
            outputs.extend(rank_outputs)
            output_proofs.extend(tensor_check(item) for item in rank_outputs)
            grads = checks(row, update, "parameter_gradient_errors")
            usage = checks(row, update, "gradient_usage_checks")
            states = checks(row, update, "parameter_state_errors")
            compute = checks(row, update, "compute_parameter_checks")
            norms = checks(row, update, "norm_checks")
            steps = checks(row, update, "step_checks")
            if input_hashes is not None:
                stage, _, replica = coordinates[rank]
                indices = list(range(replica, len(input_hashes), dp))
                slots = checks(row, update, "consumed_slots")
                if [(slot["index"], slot["prepared_input_sha256"]) for slot in slots] != [
                    (index, input_hashes[index]) for index in indices
                ]:
                    raise ValueError(f"Rank {rank} update {update} consumed the wrong logical slots")
                rank_losses = checks(row, update, "loss_checks")
                output_indices = indices if stage == pp - 1 else []
                if [loss["index"] for loss in rank_losses] != output_indices:
                    raise ValueError(f"Rank {rank} update {update} loss checks do not match the logical slots")
                loss_checks.extend(rank_losses)
                if prediction_counts is not None:
                    expected_outputs = Counter(
                        ((update, 0, index), field, sample)
                        for index in output_indices
                        for field, count in prediction_counts[update][index].items()
                        for sample in range(count)
                    )
                    recorded_outputs = Counter(
                        (tuple(item["slot"]), item["field"], item["sample_index"]) for item in rank_outputs
                    )
                    if recorded_outputs != expected_outputs:
                        raise ValueError(f"Rank {rank} update {update} prediction checks do not cover expected outputs")

            parameter_counts = Counter(item["name"] for item in grads)
            state_counts = Counter((item["name"], item["field"]) for item in states)
            if verified_names is not None and parameter_counts != Counter(verified_names[rank]):
                raise ValueError(f"Rank {rank} update {update} gradient checks do not cover the owned intervals")
            gradient_count += len(grads)
            gradient_proofs = [tensor_check(item, gradient=True) for item in grads]
            state_proofs = [tensor_check(item) for item in states]
            failed_gradients += gradient_proofs.count("failed")
            backward_covered &= (
                bool(grads)
                and all(count == 1 for count in parameter_counts.values())
                and all(proof == "passed" for proof in gradient_proofs)
                and all(
                    has_finite_number(item, "relative_l2") and has_finite_number(item, "maximum_absolute_error")
                    for item in grads
                )
            )
            backward_failed |= "failed" in gradient_proofs
            usage_covered &= (
                bool(grads)
                and Counter(item["name"] for item in usage) == parameter_counts
                and all(
                    type(item.get("reference_used")) is bool and type(item.get("actual_used")) is bool for item in usage
                )
            )
            usage_failed |= not all_passed(usage) or any(
                type(item.get("reference_used")) is bool
                and type(item.get("actual_used")) is bool
                and item["reference_used"] != item["actual_used"]
                for item in usage
            )
            optimizer_failed |= (
                "failed" in state_proofs
                or not all_passed(steps + norms)
                or not all_passed(compute, "compute_equals_master")
                or any(
                    type(item.get("actual")) is int
                    and type(item.get("reference")) is int
                    and (item["actual"] < 0 or item["reference"] < 0 or item["actual"] != item["reference"])
                    for item in steps
                )
                or any(
                    has_finite_number(item, "actual")
                    and has_finite_number(item, "reference")
                    and (
                        item["actual"] < 0
                        or item["reference"] < 0
                        or abs(item["actual"] - item["reference"]) > 2e-6 + 1e-4 * abs(item["reference"])
                    )
                    for item in norms
                )
            )
            optimizer_fields.update(item["field"] for item in states)
            failed_optimizer_fields.update(
                item["field"] for item, proof in zip(states, state_proofs, strict=True) if proof == "failed"
            )
            optimizer_covered &= (
                bool(grads)
                and len(norms) == 1
                and all(proof == "passed" for proof in state_proofs)
                and all(has_finite_number(item, "actual") and has_finite_number(item, "reference") for item in norms)
                and all(type(item.get("actual")) is int and type(item.get("reference")) is int for item in steps)
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
            finite_or_none(relative) is not None
            and finite_or_none(cosine) is not None
            and 0 <= relative <= GRADIENT_RELATIVE_L2_LIMIT
            and cosine >= GRADIENT_COSINE_MINIMUM
        )
        gradient_failed = backward_failed or not aggregate_passed
        failed_losses = sum(
            loss.get("passed") is False
            or not has_finite_number(loss, "actual")
            or (
                has_finite_number(loss, "reference")
                and abs(loss["actual"] - loss["reference"]) > 2e-6 + 1e-4 * abs(loss["reference"])
            )
            for loss in loss_checks
        )
        loss_covered = bool(loss_checks) and all(
            loss.get("passed") is True and has_finite_number(loss, "reference") for loss in loss_checks
        )
        updates.append(
            {
                "update": update,
                "forward": {
                    "status": verdict("failed" in output_proofs or bool(failed_losses), covered=False),
                    "prediction_check_status": verdict(
                        "failed" in output_proofs, bool(outputs) and all(proof == "passed" for proof in output_proofs)
                    ),
                    "prediction_coverage_status": verdict(False, prediction_counts is not None),
                    "prediction_checks": len(outputs),
                    "failed_prediction_checks": output_proofs.count("failed"),
                    "logical_slot_coverage_status": verdict(
                        False, input_hashes is not None and fingerprint_version == 2
                    ),
                    "loss_check_status": verdict(bool(failed_losses), loss_covered),
                    "loss_checks": len(loss_checks),
                    "failed_loss_checks": failed_losses,
                    "coverage": "predictions, reported logical slots and optional reference losses; layer checks are separate",
                },
                "backward": {
                    "status": verdict(gradient_failed or usage_failed, covered=False),
                    "preclip_parameter_check_status": verdict(gradient_failed, backward_covered),
                    "gradient_usage_check_status": verdict(usage_failed, usage_covered),
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
    earlier_failure = any(
        update[gate]["status"] == "failed" for update in updates for gate in ("forward", "backward", "optimizer")
    )
    return {
        "format": "cosmos_tp_pp_fp32_differential_summary",
        "run": header,
        "parameter_coverage": parameter_coverage,
        "updates": updates,
        "training": {
            "status": verdict(continuous is True and earlier_failure, covered=False),
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
    parser.add_argument("--data-parallel-size", type=int, help="DP size for legacy reports without DP metadata")
    parser.add_argument(
        "--bundle-manifest", type=Path, help="Canonical bundle manifest for TP/DP interval verification"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [args.prefix.with_name(f"{args.prefix.name}.rank{rank}.json") for rank in range(args.world_size)]
    artifacts = [file.read_bytes() for file in files]
    report = summarize_rank_reports(
        [json.loads(artifact) for artifact in artifacts],
        data_parallel_size=args.data_parallel_size,
        bundle_bytes=args.bundle_manifest.read_bytes() if args.bundle_manifest else None,
    )
    report["artifacts"] = [
        {"name": file.name, "sha256": hashlib.sha256(artifact).hexdigest()}
        for file, artifact in zip(files, artifacts, strict=True)
    ]
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
