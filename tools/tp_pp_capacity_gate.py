# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate the complete original-capacity TP numerical observation on the CPU."""

import json
import math
from collections import Counter

UPDATES = range(3)
STATE_FIELDS = ("master", "exp_avg", "exp_avg_sq")
REL_TOL = 1e-4
ABS_TOL = 2e-6
MIN_COSINE = 0.999999


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metric_record(item, label):
    """Recompute the producer's comparison branch from recorded reference norm."""
    passed = item.get("passed")
    if type(passed) is not bool:
        raise ValueError(f"{label}: missing boolean comparison result")
    if item.get("nonfinite") is True:
        if passed:
            raise ValueError(f"{label}: nonfinite comparison marked passed")
        return False
    norm = item.get("reference_squared_l2")
    relative_key = "comparison_relative_l2" if label == "gradient" else "relative_l2"
    absolute_key = "comparison_maximum_absolute_error" if label == "gradient" else "maximum_absolute_error"
    relative, absolute = item.get(relative_key), item.get(absolute_key)
    if any(not _finite(x) or x < 0 for x in (norm, relative, absolute)):
        raise ValueError(f"{label}: invalid comparison metrics or missing reference norm")
    if label != "gradient" and item.get("local_windows_checked") is not False:
        raise ValueError(f"{label}: unexpected local window comparison")
    computed = absolute <= ABS_TOL if norm <= 1e-16 else relative <= REL_TOL
    if passed is not computed:
        raise ValueError(f"{label}: comparison flag disagrees with tolerance")
    return passed


def _norm_record(item):
    """Recheck torch.isclose; reject results too near the FP32 rounding boundary."""
    actual, reference, passed = item.get("actual"), item.get("reference"), item.get("passed")
    if not _finite(actual) or not _finite(reference) or type(passed) is not bool:
        raise ValueError("invalid gradient norm observation")
    error = abs(actual - reference)
    tolerance = ABS_TOL + REL_TOL * abs(reference)
    if abs(error - tolerance) <= 2e-6 * max(error, tolerance):
        raise ValueError("gradient norm is too close to the FP32 tolerance boundary")
    if passed is not (error < tolerance):
        raise ValueError("gradient norm flag disagrees with tolerance")
    return passed


def _keyed(items, fields, expected, label):
    if not isinstance(items, list):
        raise ValueError(f"{label}: missing records")
    observed = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{label}: invalid record")
        if type(item.get("update")) is not int or any(
            not isinstance(item.get(field), str) for field in fields if field != "update"
        ):
            raise ValueError(f"{label}: invalid record key type")
        key = tuple(item.get(field) for field in fields)
        try:
            duplicate = key in observed
        except TypeError as exc:
            raise ValueError(f"{label}: invalid record key") from exc
        if duplicate:
            raise ValueError(f"{label}: duplicate record {key}")
        observed[key] = item
    if observed.keys() != expected:
        raise ValueError(f"{label}: incomplete or unexpected records")
    return observed


def _output_records(items, pp_rank, pp_size):
    if not isinstance(items, list):
        raise ValueError("missing output observations")
    if pp_rank != pp_size - 1:
        if items:
            raise ValueError("output observations on a non-output stage")
        return True
    expected = Counter(
        {
            (update, (update, 0, index), field, sample): 1
            for update in UPDATES
            for index in (0, 1)
            for field in ("preds_vision", "preds_action")
            for sample in (0, 1)
        }
    )
    observed = Counter()
    passed = True
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("slot"), list):
            raise ValueError("invalid output observation")
        if (
            type(item.get("update")) is not int
            or type(item.get("sample_index")) is not int
            or any(type(index) is not int for index in item["slot"])
        ):
            raise ValueError("invalid output observation key type")
        key = (item.get("update"), tuple(item["slot"]), item.get("field"), item.get("sample_index"))
        try:
            observed[key] += 1
        except TypeError as exc:
            raise ValueError("invalid output observation key") from exc
        if item.get("actual_dtype") != "torch.float32" or item.get("reference_dtype") != "torch.float32":
            raise ValueError("output is not measured in FP32")
        passed &= _metric_record(item, "output")
    if observed != expected:
        raise ValueError("incomplete or unexpected output observations")
    return passed


def _activation_records(items, pp_rank, pp_size):
    # PipelineStage assigns the 36 decoder layers contiguously to stages.
    layer_start, layer_end = 36 * pp_rank // pp_size, 36 * (pp_rank + 1) // pp_size
    expected = {
        (update, str(layer), stream)
        for update in UPDATES
        for layer in range(layer_start, layer_end)
        for stream in ("causal_seq", "full_only_seq", "action_seq")
    }
    observed = _keyed(items, ("update", "layer", "stream"), expected, "decoder activation")
    for item in observed.values():
        dtype_counts = item.get("dtype_counts")
        if not isinstance(dtype_counts, dict) or set(dtype_counts) != {"torch.float32"}:
            raise ValueError("decoder activation is not measured in FP32")
        count = dtype_counts["torch.float32"]
        if type(count) is not int or count < 2:
            raise ValueError("decoder activation lacks both microbatch observations")
        for flag in ("autocast_counts", "matmul_tf32_counts", "cudnn_tf32_counts"):
            if item.get(flag) != {"False": count}:
                raise ValueError(f"decoder runtime has autocast or TF32 enabled: {flag}")


def validate_reports(reports, tp, result, reference, bundle):
    """Return the gate result; raise for an incomplete or inconsistent report.

    A complete observation that fails numerically is returned with passed=False,
    allowing the coordinator to continue with the other parallel topology.
    """
    if tp not in (2, 4) or len(reports) != 4:
        raise ValueError("expected TP2 or TP4 and exactly four rank reports")
    pp_size = 4 // tp
    if bundle.get("tp") != tp or bundle.get("pp") != pp_size or bundle.get("complete") is not True:
        raise ValueError("bundle topology or completeness mismatch")
    parameters = bundle.get("parameters")
    names = reference.get("names")
    if not isinstance(parameters, dict) or not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("invalid reference or bundle parameter manifest")
    expected_names = {f"net.{name}" for name in names}
    if {name for name, item in parameters.items() if item.get("optimizer_fields")} != expected_names:
        raise ValueError("reference and trainable bundle parameter sets differ")
    for name in expected_names:
        stage = parameters[name].get("pipeline_rank")
        if type(stage) is not int or stage not in range(pp_size):
            raise ValueError("invalid bundle pipeline stage")
    runtime_hash = reference.get("runtime_constants_sha256")
    if not isinstance(runtime_hash, str) or len(runtime_hash) != 64:
        raise ValueError("invalid reference runtime constants hash")
    seed = reference.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("native reference is missing a valid requested seed")
    summaries = None
    ranks = []
    any_local_failure = False
    global_passed = None
    for rank, report in enumerate(reports):
        if not isinstance(report, dict):
            raise ValueError(f"rank {rank}: invalid report")
        required = {
            "rank": rank,
            "seed": seed,
            "updates": 3,
            "original_capacity": True,
            "attention_backend": "flex",
            "objective": "native_rf",
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp_size,
            "continuous_trajectory": True,
            "replay_native_state": False,
            "native_adam_arithmetic": True,
            "deterministic_algorithms": True,
            "activation_cpu_offload": True,
            "production_capacity_576_slot_acceptance": False,
            "runtime_constants_sha256": runtime_hash,
            "observe_all_updates": True,
            "retain_full_text": True,
            "native_dtype_evidence": "historical_reference_not_recorded",
        }
        if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in required.items()):
            raise ValueError(f"rank {rank}: incomplete or inconsistent configuration")
        passed = report.get("passed")
        if type(passed) is not bool or report.get("pointwise_gates_passed") is not passed:
            raise ValueError(f"rank {rank}: inconsistent pass flag")
        if global_passed is None:
            global_passed = passed
        elif global_passed is not passed:
            raise ValueError("rank pass flags disagree")
        failures = report.get("gate_failures")
        if not isinstance(failures, list) or any(not isinstance(item, str) or not item for item in failures):
            raise ValueError(f"rank {rank}: invalid gate failures")
        summary = report.get("gradient_summary")
        if not isinstance(summary, list) or len(summary) != 3:
            raise ValueError(f"rank {rank}: incomplete gradient summary")
        summary_pass = True
        for item in summary:
            if not isinstance(item, dict) or not _finite(item.get("relative_l2")) or not _finite(item.get("cosine")):
                raise ValueError(f"rank {rank}: invalid gradient summary")
            summary_pass &= 0 <= item["relative_l2"] <= REL_TOL and item["cosine"] >= MIN_COSINE
        if summaries is None:
            summaries = summary
        elif summaries != summary:
            raise ValueError("global gradient summaries disagree across ranks")
        local_names = {name for name in expected_names if parameters[name]["pipeline_rank"] == rank // tp}
        pairs = {(update, name) for update in UPDATES for name in local_names}
        triples = {(update, name, field) for update, name in pairs for field in STATE_FIELDS}
        grads = _keyed(report.get("parameter_gradient_errors"), ("update", "name"), pairs, "gradient")
        states = _keyed(report.get("parameter_state_errors"), ("update", "name", "field"), triples, "state")
        computes = _keyed(report.get("compute_parameter_checks"), ("update", "name"), pairs, "compute")
        steps = _keyed(report.get("step_checks"), ("update", "name"), pairs, "step")
        norms = _keyed(report.get("norm_checks"), ("update",), {(u,) for u in UPDATES}, "norm")
        _activation_records(report.get("activation_checks"), rank // tp, pp_size)
        output_pass = _output_records(report.get("output_errors"), rank // tp, pp_size)
        local_pass = summary_pass and output_pass
        gradient_relatives = []
        for item in grads.values():
            if any(
                item.get(key) != "torch.float32"
                for key in ("gradient_dtype", "reference_dtype", "model_parameter_dtype")
            ):
                raise ValueError("gradient, model parameter or reference is not FP32")
            local_pass &= _metric_record(item, "gradient")
            relative, absolute = item.get("relative_l2"), item.get("maximum_absolute_error")
            if not _finite(relative) or not _finite(absolute) or relative < 0 or absolute < 0:
                raise ValueError("invalid raw gradient metrics")
            gradient_relatives.append(relative)
        if not gradient_relatives or report.get("maximum_parameter_gradient_relative_l2") != max(gradient_relatives):
            raise ValueError("maximum parameter gradient summary disagrees with observations")
        for item in states.values():
            if item.get("actual_dtype") != "torch.float32" or item.get("reference_dtype") != "torch.float32":
                raise ValueError("optimizer state or reference is not FP32")
            local_pass &= _metric_record(item, "state")
        for item in norms.values():
            local_pass &= _norm_record(item)
        for item in steps.values():
            actual, target, match = item.get("actual"), item.get("reference"), item.get("passed")
            if type(actual) is not int or type(target) is not int or type(match) is not bool:
                raise ValueError("invalid optimizer step observation")
            if match is not (actual == target):
                raise ValueError("optimizer step flag disagrees with values")
            local_pass &= match
        for item in computes.values():
            if item.get("compute_dtype") != "torch.float32" or item.get("master_dtype") != "torch.float32":
                raise ValueError("compute or master parameter is not FP32")
            match = item.get("compute_equals_master")
            if type(match) is not bool:
                raise ValueError("missing compute/master equality result")
            local_pass &= match
        if not local_pass and not failures:
            raise ValueError(f"rank {rank}: failed observations missing gate failures")
        any_local_failure |= not local_pass or bool(failures)
        ranks.append({"rank": rank, "passed": passed, "gate_failures": len(failures)})
    if global_passed == any_local_failure:
        raise ValueError("global pass flag disagrees with observations")
    if type(result.get("returncode")) is not int or (result["returncode"] == 0) != global_passed:
        raise ValueError(f"TP{tp} return code disagrees with rank reports")
    return {
        "passed": global_passed,
        "complete_observation": True,
        "rng_state_comparison_verified": False,
        "ranks": ranks,
    }


def tp_gate(output, tp, result, reference, bundle):
    reports = [json.loads(output.with_name(f"{output.stem}.rank{rank}.json").read_text()) for rank in range(4)]
    return validate_reports(
        reports,
        tp,
        result,
        json.loads((reference / "manifest.json").read_text()),
        json.loads((bundle / "manifest.json").read_text()),
    )
