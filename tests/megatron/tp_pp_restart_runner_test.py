# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Restart acceptance must preserve the canonical topology and global update size."""

import json

import pytest

from tools.run_tp_pp_restart_acceptance import start_topology, training_entry, validate_microbatch_gc


def gc_records(root):
    for rank in range(4):
        rows = [
            {
                "iteration": iteration,
                "interval": 16,
                "schedule_completed": True,
                "completed_backwards": 48,
                "collections": [{"completed_backwards": tick} for tick in [16, 32, 48]],
            }
            for iteration in [1201, 1202, 1203]
        ]
        (root / f"microbatch_gc.rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_gc_requires_execution_on_every_rank_and_update(tmp_path):
    gc_records(tmp_path)
    assert validate_microbatch_gc(tmp_path, [1201, 1202, 1203], interval=16, microbatches=48)


@pytest.mark.parametrize("failure", ["missing", "disabled", "partial", "duplicate"])
def test_gc_request_without_complete_execution_is_rejected(tmp_path, failure):
    gc_records(tmp_path)
    path = tmp_path / "microbatch_gc.rank2.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "missing":
        path.unlink()
    else:
        if failure == "disabled":
            rows[1]["interval"] = 0
        elif failure == "partial":
            rows[1]["collections"].pop()
        else:
            rows.append(rows[0])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="GC policy"):
        validate_microbatch_gc(tmp_path, [1201, 1202, 1203], interval=16, microbatches=48)


@pytest.mark.parametrize(
    "whole,blocks,module",
    [
        (False, 0, "cosmos_framework.scripts.train"),
        (True, 0, "tools.run_tp_pp_with_activation_offload"),
        (False, 2, "tools.run_tp_pp_with_selective_offload"),
    ],
)
def test_offload_modes_select_explicit_training_entry(whole, blocks, module):
    assert training_entry(activation_cpu_offload=whole, selective_blocks=blocks) == module


@pytest.mark.parametrize("whole,blocks", [(True, 2), (False, -1)])
def test_incompatible_offload_modes_are_rejected(whole, blocks):
    with pytest.raises(ValueError):
        training_entry(activation_cpu_offload=whole, selective_blocks=blocks)


def test_canonical_true_dp_start_preserves_all_twelve_slots():
    topology = start_topology({"tp": 2, "pp": 1}, None, 4, 12)
    assert topology == {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "logical_data_parallel_size": 4,
        "global_packed_microbatches_per_update": 12,
    }


def checkpoint():
    return {
        "format": "cosmos_megatron_tp_pp",
        "world_size": 4,
        "topology": {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "logical_data_parallel_size": 4,
            "global_packed_microbatches_per_update": 12,
        },
    }


def test_existing_runtime_start_retains_its_topology():
    saved = checkpoint()
    assert start_topology({"tp": 2, "pp": 2}, saved, None, None) == saved["topology"]


@pytest.mark.parametrize(
    "bundle,saved,logical_dp,slots",
    [
        ({"tp": 2, "pp": 1}, checkpoint(), None, None),
        ({"tp": 2, "pp": 2}, checkpoint(), 4, 3),
        ({"tp": 2, "pp": 2}, checkpoint(), 192, 576),
        ({"tp": 2, "pp": 1}, None, None, 12),
        ({"tp": 2, "pp": 1}, None, 3, 12),
        ({"tp": 2, "pp": 1}, None, 4, 10),
        ({"tp": 2, "pp": 3}, None, 4, 12),
    ],
)
def test_incompatible_start_is_rejected_before_launch(bundle, saved, logical_dp, slots):
    with pytest.raises(ValueError):
        start_topology(bundle, saved, logical_dp, slots)
