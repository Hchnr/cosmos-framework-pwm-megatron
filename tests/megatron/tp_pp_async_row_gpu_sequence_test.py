# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.run_tp_pp_async_row_gpu_sequence import verify_tiny


def report(rank, *, passed=True, backend="inductor"):
    return {
        "rank": rank,
        "peak_cuda_bytes": 1024,
        "cases": [
            {
                "passed": passed,
                "dtype": "fp32",
                "full_activation_checkpointing": True,
                "compile_backend": backend,
                "max_relative_l2": 0.0,
            }
            for _ in range(4)
        ],
    }


def test_tiny_gate_checks_all_ranks_and_all_updates(tmp_path):
    for rank in range(2):
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(report(rank)))
    assert len(verify_tiny(tmp_path, 2, compiled=True, dtype="fp32")) == 2
    (tmp_path / "rank1.json").write_text(json.dumps(report(1, passed=False)))
    with pytest.raises(ValueError, match="numerical gate"):
        verify_tiny(tmp_path, 2, compiled=True, dtype="fp32")


def test_tiny_gate_rejects_wrong_backend(tmp_path):
    (tmp_path / "rank0.json").write_text(json.dumps(report(0, backend="aot_eager")))
    with pytest.raises(ValueError, match="Inductor"):
        verify_tiny(tmp_path, 1, compiled=True, dtype="fp32")
