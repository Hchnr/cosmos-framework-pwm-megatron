# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.queue_tp_pp_replay_native_state import EXPECTED_FAILURE, intervention_decision, verify_results


def complete_failed_intervention(directory):
    (directory / "reports").mkdir()
    modules = {f"layer{i}": {"forward_calls": 2} for i in range(216)}
    summary = {
        "passed": False,
        "tp4": {"returncode": 1},
        "source_sha256_before": {"a": "b"},
        "source_sha256_after": {"a": "b"},
        "snapshot_after": {"sha256": "x"},
        "ranks": [{"rank": rank, "gate_failure_count": 1} for rank in range(4)],
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    for rank in range(4):
        report = {
            "updates": 3,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "continuous_trajectory": True,
            "replay_native_state": False,
            "passed": False,
            "gradient_summary": [{}, {}, {}],
        }
        sidecar = {"full_width_vjp": True, "failure": EXPECTED_FAILURE, "modules": modules}
        (directory / f"tp4.rank{rank}.json").write_text(json.dumps(report))
        (directory / "reports" / f"fp32_reconstructed_kv.rank{rank}.json").write_text(json.dumps(sidecar))
    return summary


def test_complete_numerical_failure_runs_replay(tmp_path):
    complete_failed_intervention(tmp_path)
    run, reason = intervention_decision(tmp_path)
    assert run and "failed the fixed numerical gates" in reason


@pytest.mark.parametrize("case", ["passed", "missing", "runtime_failure", "different_source", "unexercised"])
def test_passed_incomplete_or_infrastructure_failure_skips_replay(tmp_path, case):
    summary = complete_failed_intervention(tmp_path)
    if case == "passed":
        summary["passed"] = True
    elif case == "missing":
        (tmp_path / "tp4.rank3.json").unlink()
    elif case == "runtime_failure":
        summary["error"] = "TimeoutError"
    elif case == "different_source":
        summary["source_sha256_after"] = {"a": "changed"}
    else:
        path = tmp_path / "reports/fp32_reconstructed_kv.rank0.json"
        sidecar = json.loads(path.read_text())
        sidecar["modules"]["layer0"]["forward_calls"] = 0
        path.write_text(json.dumps(sidecar))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert not intervention_decision(tmp_path)[0]


def test_replay_result_requires_full_state_witness_and_not_acceptance(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "witness").mkdir()
    for rank in range(4):
        report = {
            "updates": 3,
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "replay_native_state": True,
            "continuous_trajectory": False,
            "passed": False,
            "gradient_summary": [{}, {}, {}],
            "pointwise_gates_passed": False,
            "gate_failures": ["first"],
        }
        (tmp_path / f"tp4.rank{rank}.json").write_text(json.dumps(report))
        sidecar = {
            "completed": True,
            "failure": None,
            "full_width_vjp": True,
            "modules": {f"module{i}": {"forward_calls": 1} for i in range(216)},
        }
        (tmp_path / "reports" / f"fp32_reconstructed_kv.rank{rank}.json").write_text(json.dumps(sidecar))
        for update in (1, 2):
            witness = {"verified_shards": 811, "master_moments_step_compute_exact": True}
            (tmp_path / "witness" / f"replay_state.update{update}.rank{rank}.json").write_text(json.dumps(witness))
    assert len(verify_results(tmp_path)) == 4
    (tmp_path / "witness/replay_state.update2.rank3.json").write_text(
        json.dumps({"verified_shards": 1, "master_moments_step_compute_exact": True})
    )
    with pytest.raises(ValueError, match="811"):
        verify_results(tmp_path)
