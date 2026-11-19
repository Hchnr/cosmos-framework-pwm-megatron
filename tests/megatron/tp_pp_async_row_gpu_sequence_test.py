# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.run_tp_pp_async_row_gpu_sequence import fingerprints, verify_snapshot, verify_tiny


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


def test_snapshot_rejects_source_change_and_wrong_path(tmp_path):
    source = tmp_path / "cosmos_framework/module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    manifest = {"snapshot": str(tmp_path.resolve()), "files": fingerprints(tmp_path)}
    assert verify_snapshot(tmp_path, manifest) == manifest["files"]
    with pytest.raises(ValueError, match="path"):
        verify_snapshot(tmp_path / "other", manifest)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path, manifest)
