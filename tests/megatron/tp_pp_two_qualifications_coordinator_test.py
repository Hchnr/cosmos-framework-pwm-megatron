# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.run_tp_pp_two_qualifications_coordinator import gate_adaptive, gate_chunk, gate_nested_memory


def write(path, data):
    path.write_text(json.dumps(data))


def test_adaptive_gate_requires_successful_process_and_both_complete_reports(tmp_path):
    data = {
        "passed": True,
        "full_model_acceptance": False,
        "updates": 7,
        "updates_checked": [1] * 7,
        "compile_events": [1],
    }
    for rank in range(2):
        write(tmp_path / f"adaptive.rank{rank}.json", {**data, "rank": rank})
    assert gate_adaptive({"returncode": 0}, tmp_path)["passed"]
    with pytest.raises(ValueError, match="process failed"):
        gate_adaptive({"returncode": 1}, tmp_path)
    write(tmp_path / "adaptive.rank1.json", {**data, "updates_checked": []})
    with pytest.raises(ValueError, match="Incomplete"):
        gate_adaptive({"returncode": 0}, tmp_path)


def test_nested_chunk_gate_checks_both_rank_counts(tmp_path):
    for rank in range(2):
        write(tmp_path / f"chunk.rank{rank}.json", {"passed": True, "rank": rank})
        write(
            tmp_path / f"chunk.flat_checkpoint.rank{rank}.json",
            {
                "passed": True,
                "full_model_acceptance": False,
                "chunk_tokens": 1,
                "native_checkpoint_calls": 4,
                "tp_checkpoint_calls": 4,
            },
        )
        write(
            tmp_path / f"chunk.checkpointed_chunk.rank{rank}.json",
            {"passed": True, "full_model_acceptance": False, "chunk_checkpoints": 3},
        )
    assert gate_chunk({"returncode": 0}, tmp_path)["checkpointed_chunk_counts"] == [3, 3]
    write(
        tmp_path / "chunk.checkpointed_chunk.rank1.json",
        {"passed": True, "full_model_acceptance": False, "chunk_checkpoints": 0},
    )
    with pytest.raises(ValueError, match="witness"):
        gate_chunk({"returncode": 0}, tmp_path)


def test_nested_memory_gate_requires_cuda_peak_and_recomputation(tmp_path):
    report = {
        "passed": True,
        "cuda_peak_reduced": True,
        "outputs_and_all_gradients_close": True,
        "plain": {"outer_block_calls": 2, "peak_allocated_bytes": 20},
        "nested": {"outer_block_calls": 2, "peak_allocated_bytes": 10},
    }
    path = tmp_path / "nested_memory.json"
    write(path, report)
    assert gate_nested_memory({"returncode": 0}, tmp_path)["passed"]
    report["nested"]["outer_block_calls"] = 1
    write(path, report)
    with pytest.raises(ValueError, match="recomputation"):
        gate_nested_memory({"returncode": 0}, tmp_path)
