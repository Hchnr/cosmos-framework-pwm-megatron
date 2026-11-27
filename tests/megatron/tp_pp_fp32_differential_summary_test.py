# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import hashlib
import json

import pytest

from tools.summarize_tp_pp_fp32_differential import summarize_rank_reports


@pytest.fixture
def reports():
    updates = range(3)
    template = {
        "scope": "test",
        "objective": "test",
        "updates": len(updates),
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "replay_native_state": False,
        "continuous_trajectory": True,
        "gradient_summary": [{"relative_l2": 0.0, "cosine": 1.0} for _ in updates],
        "parameter_gradient_errors": [
            {
                "update": u,
                "name": "weight",
                "relative_l2": 0.0,
                "maximum_absolute_error": 0.0,
                "comparison_relative_l2": 0.0,
                "comparison_maximum_absolute_error": 0.0,
                "reference_squared_l2": 1.0,
                "passed": True,
            }
            for u in updates
        ],
        "output_errors": [
            {
                "update": u,
                "passed": True,
                "relative_l2": 0.0,
                "maximum_absolute_error": 0.0,
                "reference_squared_l2": 1.0,
            }
            for u in updates
        ],
        "norm_checks": [{"update": u, "actual": 1.0, "reference": 1.0, "passed": True} for u in updates],
        "step_checks": [{"update": u, "name": "weight", "actual": u, "reference": u, "passed": True} for u in updates],
        "compute_parameter_checks": [{"update": u, "name": "weight", "compute_equals_master": True} for u in updates],
        "parameter_state_errors": [
            {
                "update": u,
                "name": "weight",
                "field": field,
                "passed": True,
                "relative_l2": 0.0,
                "maximum_absolute_error": 0.0,
                "reference_squared_l2": 1.0,
            }
            for u in updates
            for field in ("master", "exp_avg", "exp_avg_sq")
        ],
    }
    return [dict(copy.deepcopy(template), rank=rank) for rank in range(4)]


def test_incomplete_coverage_does_not_pass(reports):
    # The first pipeline stage has no prediction tensors.
    reports[0]["output_errors"] = []
    reports[1]["norm_checks"].pop()

    summary = summarize_rank_reports(reports)
    first, _, last = summary["updates"]
    assert first["forward"]["prediction_check_status"] == "passed"
    assert first["forward"]["status"] == "unverified"
    assert first["backward"]["preclip_parameter_check_status"] == "passed"
    assert first["optimizer"]["recorded_update_check_status"] == "passed"
    assert last["optimizer"]["recorded_update_check_status"] == "unverified"
    assert summary["training"]["status"] == "unverified"


def test_failing_rank_is_retained_and_replay_is_not_a_trajectory(reports):
    reports[1]["step_checks"][0]["passed"] = False

    summary = summarize_rank_reports(reports)
    assert summary["updates"][0]["optimizer"]["status"] == "failed"
    assert summary["training"]["status"] == "failed"

    for row in reports:
        row.update(replay_native_state=True, continuous_trajectory=False)
    summary = summarize_rank_reports(reports)
    assert summary["updates"][0]["optimizer"]["status"] == "failed"
    assert summary["training"]["continuous_updates"] == 0
    assert summary["training"]["status"] == "unverified"


def test_report_distinguishes_an_unused_gradient_from_a_present_zero(reports):
    for row in reports:
        row["gradient_usage_checks"] = [
            {"update": u, "name": "weight", "reference_used": False, "actual_used": False, "passed": True}
            for u in range(3)
        ]
    reports[0]["gradient_usage_checks"][0].update(actual_used=True, passed=False)
    summary = summarize_rank_reports(reports)
    assert summary["updates"][0]["backward"]["preclip_parameter_check_status"] == "passed"
    assert summary["updates"][0]["backward"]["gradient_usage_check_status"] == "failed"
    assert summary["updates"][0]["backward"]["status"] == "failed"
    assert summary["updates"][1]["backward"]["gradient_usage_check_status"] == "passed"


def test_undefined_collective_gradient_metrics_are_reported_as_failures(reports):
    for row in reports:
        row["gradient_summary"][0] = {"relative_l2": None, "cosine": None}
    backward = summarize_rank_reports(reports)["updates"][0]["backward"]
    assert backward["status"] == "failed"
    assert backward["relative_l2"] is None
    assert backward["cosine"] is None


def test_missing_rank_or_disagreeing_collective_summary_is_rejected(reports):
    with pytest.raises(ValueError, match="Got 3 rank reports"):
        summarize_rank_reports(reports[:-1])
    reports[0]["gradient_summary"][0]["relative_l2"] = 1
    with pytest.raises(ValueError, match="disagree"):
        summarize_rank_reports(reports)


def test_data_parallel_ranks_are_required_and_legacy_dp_can_be_supplied(reports):
    for row in reports:
        row["tensor_parallel_size"] = 1

    with pytest.raises(ValueError, match="TP1/PP2/DP1"):
        summarize_rank_reports(reports)
    assert summarize_rank_reports(reports, data_parallel_size=2)["run"]["data_parallel_size"] == 2

    for row in reports:
        row["data_parallel_size"] = 2
    assert summarize_rank_reports(reports)["run"]["data_parallel_size"] == 2
    with pytest.raises(ValueError, match="Got 2 rank reports"):
        summarize_rank_reports(reports[:2])
    with pytest.raises(ValueError, match="disagrees with the rank reports"):
        summarize_rank_reports(reports, data_parallel_size=1)


def test_legacy_report_without_update_mode_has_unverified_trajectory(reports):
    for row in reports:
        del row["replay_native_state"]
        del row["continuous_trajectory"]
    reports[1]["step_checks"][0]["passed"] = False

    summary = summarize_rank_reports(reports)
    assert summary["updates"][0]["optimizer"]["status"] == "failed"
    assert summary["training"]["continuous_updates"] == 0
    assert summary["training"]["status"] == "unverified"


@pytest.fixture
def dp2_bundle(reports):
    raw = json.dumps(
        {
            "format": "cosmos_tp_pp_canonical",
            "format_version": 1,
            "complete": True,
            "tp": 2,
            "pp": 1,
            "parameters": {
                "weight": {"global_shape": [3, 8], "axis": 1, "pipeline_rank": 0, "optimizer_fields": ["moment"]},
                "frozen": {"global_shape": [8], "axis": None, "pipeline_rank": 0, "optimizer_fields": []},
            },
        }
    ).encode()
    for rank, row in enumerate(reports):
        dp, tp = divmod(rank, 2)
        row.update(
            pipeline_parallel_size=1,
            data_parallel_size=2,
            parallel_ranks={"pp": 0, "tp": tp, "dp": dp},
            canonical_bundle_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            owned_parameter_fragments=[
                {
                    "name": "weight",
                    "axis": 1,
                    "offset": tp * 4,
                    "local_shape": [3, 4],
                    "start": dp * 6,
                    "end": (dp + 1) * 6,
                }
            ],
        )
    return raw


def test_dp2_parameter_coverage_requires_canonical_fragments_for_every_update(reports, dp2_bundle):
    summary = summarize_rank_reports(reports, bundle_bytes=dp2_bundle)
    assert summary["parameter_coverage"] == {
        "status": "passed",
        "bundle_manifest_sha256": hashlib.sha256(dp2_bundle).hexdigest(),
        "canonical_trainable_parameters": 1,
        "tp_parameter_layouts": 2,
        "optimizer_intervals": 4,
    }
    assert summarize_rank_reports(reports)["parameter_coverage"]["status"] == "unverified"
    reports[3]["parameter_gradient_errors"].pop()
    with pytest.raises(ValueError, match="gradient checks do not cover"):
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("start", 5, "overlapping optimizer intervals"),
        ("end", 11, "Incomplete optimizer interval coverage"),
        ("offset", 0, "incorrect parameter layout"),
    ],
)
def test_dp2_parameter_coverage_rejects_missing_duplicate_or_wrong_tp_fragments(
    reports, dp2_bundle, field, value, error
):
    reports[3]["owned_parameter_fragments"][0][field] = value
    with pytest.raises(ValueError, match=error):
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle)


def test_dp2_parameter_coverage_rejects_wrong_bundle_or_rank(reports, dp2_bundle):
    with pytest.raises(ValueError, match="did not record the supplied canonical bundle SHA"):
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle + b"\n")
    reports[3]["parallel_ranks"]["dp"] = 0
    with pytest.raises(ValueError, match="duplicate parallel rank coordinates"):
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle)


def test_dp2_slot_coverage_and_legacy_loss_reference(reports, dp2_bundle):
    hashes = ["1" * 64, "2" * 64]
    for row in reports:
        replica = row["parallel_ranks"]["dp"]
        row.update(
            prepared_input_hashes=hashes,
            microbatches_per_dp_replica=1,
            reference_prediction_counts=[[{"vision": 1}, {"vision": 1}] for _ in range(3)],
            consumed_slots=[
                {"update": update, "index": replica, "prepared_input_sha256": hashes[replica]} for update in range(3)
            ],
            loss_checks=[{"update": update, "index": replica, "actual": 1.0, "passed": None} for update in range(3)],
            output_errors=[
                {"update": update, "slot": [update, 0, replica], "field": "vision", "sample_index": 0, "passed": True}
                for update in range(3)
            ],
        )

    summary = summarize_rank_reports(reports, bundle_bytes=dp2_bundle)
    assert summary["updates"][0]["forward"]["logical_slot_coverage_status"] == "passed"
    assert summary["updates"][0]["forward"]["prediction_coverage_status"] == "passed"
    assert summary["updates"][0]["forward"]["loss_check_status"] == "unverified"
    reports[3]["loss_checks"][0].update(reference=2.0, passed=True)
    assert (
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle)["updates"][0]["forward"]["loss_check_status"]
        == "failed"
    )
    reports[3]["loss_checks"][0]["passed"] = False
    assert summarize_rank_reports(reports, bundle_bytes=dp2_bundle)["updates"][0]["forward"]["status"] == "failed"
    removed = reports[3]["output_errors"].pop()
    with pytest.raises(ValueError, match="prediction checks do not cover expected outputs"):
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle)
    reports[3]["output_errors"].append(removed)
    reports[3]["consumed_slots"][0]["index"] = 0
    with pytest.raises(ValueError, match="consumed the wrong logical slots"):
        summarize_rank_reports(reports, bundle_bytes=dp2_bundle)


@pytest.mark.parametrize("invalid_hash", [None, "", "1" * 63, "g" * 64, "A" * 64, 7])
def test_rejects_invalid_prepared_input_fingerprints(reports, invalid_hash):
    for rank, row in enumerate(reports):
        row.update(
            prepared_input_hashes=[invalid_hash],
            microbatches_per_dp_replica=1,
            parallel_ranks={"pp": rank // 2, "tp": rank % 2, "dp": 0},
        )
    with pytest.raises(ValueError, match="Prepared input count"):
        summarize_rank_reports(reports)


def test_rejects_invalid_runtime_constants_fingerprint(reports):
    for row in reports:
        row["runtime_constants_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="Invalid runtime constants SHA-256"):
        summarize_rank_reports(reports)


def test_contradictory_usage_and_step_cannot_pass(reports):
    for row in reports:
        row["gradient_usage_checks"] = [
            {"update": u, "name": "weight", "reference_used": False, "actual_used": False, "passed": True}
            for u in range(3)
        ]
    reports[0]["gradient_usage_checks"][0]["actual_used"] = True
    reports[1]["step_checks"][0].update(actual=999, reference=1200, passed=True)
    result = summarize_rank_reports(reports)["updates"][0]
    assert result["backward"]["gradient_usage_check_status"] == "failed"
    assert result["optimizer"]["recorded_update_check_status"] == "failed"


@pytest.mark.parametrize("section,field", [("norm_checks", "actual"), ("parameter_gradient_errors", "relative_l2")])
def test_nonfinite_evidence_cannot_pass(reports, section, field):
    reports[0][section][0][field] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite numeric evidence"):
        summarize_rank_reports(reports)


def test_missing_numeric_details_are_unverified(reports):
    del reports[0]["parameter_gradient_errors"][0]["relative_l2"]
    del reports[1]["step_checks"][0]["actual"]
    result = summarize_rank_reports(reports)["updates"][0]
    assert result["backward"]["preclip_parameter_check_status"] == "unverified"
    assert result["optimizer"]["recorded_update_check_status"] == "unverified"


@pytest.mark.parametrize(
    "section,relative_key,maximum_key,group,status_key",
    [
        ("output_errors", "relative_l2", "maximum_absolute_error", "forward", "prediction_check_status"),
        (
            "parameter_gradient_errors",
            "comparison_relative_l2",
            "comparison_maximum_absolute_error",
            "backward",
            "preclip_parameter_check_status",
        ),
        (
            "parameter_state_errors",
            "relative_l2",
            "maximum_absolute_error",
            "optimizer",
            "recorded_update_check_status",
        ),
    ],
)
def test_contradictory_tensor_error_cannot_pass(reports, section, relative_key, maximum_key, group, status_key):
    reports[0][section][0].update({relative_key: 999.0, maximum_key: 999.0, "passed": True})
    assert summarize_rank_reports(reports)["updates"][0][group][status_key] == "failed"


def test_near_zero_reference_uses_absolute_tolerance(reports):
    for row in reports:
        row["parameter_state_errors"][0].update(reference_squared_l2=0.0, relative_l2=None, maximum_absolute_error=1e-6)
    assert summarize_rank_reports(reports)["updates"][0]["optimizer"]["recorded_update_check_status"] == "passed"
    reports[0]["parameter_state_errors"][0]["maximum_absolute_error"] = 3e-6
    assert summarize_rank_reports(reports)["updates"][0]["optimizer"]["recorded_update_check_status"] == "failed"


def test_negative_norm_cannot_pass(reports):
    reports[0]["norm_checks"][0].update(actual=-1.0, reference=-1.0, passed=True)
    assert summarize_rank_reports(reports)["updates"][0]["optimizer"]["recorded_update_check_status"] == "failed"
