# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from tools.run_tp_pp_full_width_und_q import mark_und_q


class Linear:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(3, 4), requires_grad=False)
        self.bias = None
        self.column = True
        self.tp_size = 2


def stage_with(linear):
    return SimpleNamespace(layers={"0": SimpleNamespace(self_attn=SimpleNamespace(q_proj=linear))})


def test_marks_frozen_q_without_replacing_parameter():
    linear = Linear()
    parameter = linear.weight
    observation = SimpleNamespace(data={"modules": {}}, save=lambda: None)
    assert mark_und_q(stage_with(linear), observation, Linear) == 1
    assert linear.weight is parameter and linear._fp32_reconstructed_kv
    assert observation.data["modules"]["layers.0.self_attn.q_proj"]["kind"] == "frozen_und_q"


@pytest.mark.parametrize("defect", ["trainable", "bias", "row", "already_marked"])
def test_refuses_incompatible_und_projection(defect):
    linear = Linear()
    if defect == "trainable":
        linear.weight.requires_grad_(True)
    elif defect == "bias":
        linear.bias = torch.zeros(3)
    elif defect == "row":
        linear.column = False
    else:
        linear._fp32_reconstructed_kv = True
    with pytest.raises(ValueError):
        mark_und_q(stage_with(linear), SimpleNamespace(data={"modules": {}}, save=lambda: None), Linear)
