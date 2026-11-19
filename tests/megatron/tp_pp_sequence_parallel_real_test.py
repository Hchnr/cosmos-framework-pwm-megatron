# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real decoder and boundaries under Gloo; NCCL is a separate queued gate."""

import json

from tools.qualify_tp_pp_sequence_parallel_real import qualify


def test_real_sequence_parallel_gloo_two_and_four_ranks(tmp_path):
    output = tmp_path / "qualification"
    summary = qualify("gloo", output)
    assert summary["rank_reports"] == 6
    for world in (2, 4):
        for rank in range(world):
            report = json.loads((output / f"tp{world}/rank{rank}.json").read_text())
            assert [step["no_target"] for step in report["steps"]] == [False, False, False, True]
            assert report["steps"][-1]["gradients"]["none"] > 0
            assert report["steps"][0]["layer_sync_once_counts"]["head_norm"] > 0
            assert report["steps"][0]["layer_sync_once_counts"]["sequence_norm"] > 0
            assert report["steps"][0]["distributed_api_trace"]
