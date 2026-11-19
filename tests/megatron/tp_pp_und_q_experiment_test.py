# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest

from tools.run_tp_pp_und_q_experiment import gate_replay


def replay_report():
    exact = {"exact": True}
    common = {"captured_native_reproduced": exact, "captured_tp_reproduced": [exact] * 4}
    return {
        "gpu_replay": {
            "projection": {**common, "native_full_vs_tp_sharded": {"exact": False}},
            "headwise": {"q_norm": common, "rope": common},
        }
    }


def test_only_reproduced_q_width_difference_enables_intervention():
    report = replay_report()
    assert gate_replay(report)
    report["gpu_replay"]["projection"]["native_full_vs_tp_sharded"] = {"exact": True}
    assert not gate_replay(report)


@pytest.mark.parametrize("stage", ["projection", "q_norm", "rope"])
def test_failed_capture_reproduction_rejects_experiment(stage):
    report = copy.deepcopy(replay_report())
    section = report["gpu_replay"]["projection"] if stage == "projection" else report["gpu_replay"]["headwise"][stage]
    section["captured_tp_reproduced"][0] = {"exact": False}
    with pytest.raises(ValueError, match="did not reproduce"):
        gate_replay(report)
