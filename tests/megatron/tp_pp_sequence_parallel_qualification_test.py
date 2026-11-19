# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from tools.qualify_tp_pp_sequence_parallel_toy import gate


def test_adjoint_gate_checks_every_rank_and_replicated_loss_scale(tmp_path):
    for rank in range(2):
        (tmp_path / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": rank,
                    "world": 2,
                    "backend": "nccl",
                    "production_acceptance": False,
                    "loss_scale_negative_control": {"gather_loss_divisor_1": 2.0, "gather_loss_divisor_2": 1.0},
                }
            )
        )
    assert gate(tmp_path, "adjoint", 2)["passed"]
    report = json.loads((tmp_path / "rank1.json").read_text())
    report["loss_scale_negative_control"]["gather_loss_divisor_2"] = 2.0
    (tmp_path / "rank1.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        gate(tmp_path, "adjoint", 2)
