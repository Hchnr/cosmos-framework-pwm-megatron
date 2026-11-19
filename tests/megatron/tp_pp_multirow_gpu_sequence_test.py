# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.run_tp_pp_multirow_gpu_sequence import load_bf16_case, load_fp32_case, validate_counts


def test_fp32_requires_all_updates_and_profiled_rank_counts(tmp_path):
    output = tmp_path / "candidate.json"
    for rank in range(2):
        report = {"passed": True, "compiled": True, "relative_gradient_l2": {f"{i}/p": 1e-7 for i in range(4)}}
        sidecar = {
            "passed": True,
            "mode": "candidate",
            "tp": 2,
            "full_ac_compile": True,
            "full_model_acceptance": False,
            "functional_all_reduce_calls_per_update": [70, 70, 70, 70],
            "peak_cuda_allocated_bytes": 123,
        }
        (tmp_path / f"candidate.rank{rank}.json").write_text(json.dumps(report))
        (tmp_path / f"candidate.multirow_candidate.rank{rank}.json").write_text(json.dumps(sidecar))
    result = load_fp32_case(output, 2, "candidate")
    assert [row["counts"] for row in result] == [[70] * 4, [70] * 4]
    (tmp_path / "candidate.multirow_candidate.rank1.json").write_text("{}")
    with pytest.raises(ValueError, match="invalid FP32"):
        load_fp32_case(output, 2, "candidate")


def test_profile_gate_requires_fewer_calls_on_every_rank_and_update():
    baseline = [{"rank": 0, "counts": [12, 12, 12, 12]}, {"rank": 1, "counts": [12, 12, 12, 12]}]
    candidate = [{"rank": 0, "counts": [8, 8, 8, 8]}, {"rank": 1, "counts": [8, 8, 8, 8]}]
    assert validate_counts(baseline, candidate)[0]["reduced_calls_per_update"] == [4, 4, 4, 4]
    candidate[1]["counts"][2] = 12
    with pytest.raises(ValueError, match="fewer all-reduces"):
        validate_counts(baseline, candidate)


def test_bf16_gate_rejects_incomplete_or_failed_update(tmp_path):
    rows = [
        {
            "passed": True,
            "dtype": "bf16",
            "full_activation_checkpointing": True,
            "compile_backend": "inductor",
            "max_relative_l2": 1e-6,
        }
        for _ in range(4)
    ]
    for rank in range(2):
        (tmp_path / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "cases": rows, "peak_cuda_bytes": 123}))
    assert len(load_bf16_case(tmp_path, 2)) == 2
    rows[-1]["passed"] = False
    (tmp_path / "rank1.json").write_text(json.dumps({"rank": 1, "cases": rows, "peak_cuda_bytes": 123}))
    with pytest.raises(ValueError, match="BF16 fixed numerical gate"):
        load_bf16_case(tmp_path, 2)
