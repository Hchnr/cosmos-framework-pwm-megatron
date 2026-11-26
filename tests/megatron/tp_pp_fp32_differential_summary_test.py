# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

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
        "parameter_gradient_errors": [{"update": u, "name": "weight", "passed": True} for u in updates],
        "output_errors": [{"update": u, "passed": True} for u in updates],
        "norm_checks": [{"update": u, "passed": True} for u in updates],
        "step_checks": [{"update": u, "name": "weight", "passed": True} for u in updates],
        "compute_parameter_checks": [{"update": u, "name": "weight", "compute_equals_master": True} for u in updates],
        "parameter_state_errors": [
            {"update": u, "name": "weight", "field": field, "passed": True}
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
