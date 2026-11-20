# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import torch
from torch import nn

from tools.run_tp_pp_native_block_diagnostic import clear_native_block_partial_flags


def test_native_norm_flags_after_meta_materialization():
    with torch.device("meta"):
        layer = nn.Linear(8, 8)
    for parameter in layer.parameters():
        parameter._cosmos_native_block_diagnostic = True
    layer.to_empty(device="cpu")
    # Driver flags are assigned after materialization, unlike the old marker.
    for parameter in layer.parameters():
        parameter._cosmos_tp_partial_grad = True
        if hasattr(parameter, "_cosmos_native_block_diagnostic"):
            del parameter._cosmos_native_block_diagnostic
    adapter = nn.Linear(8, 8)
    adapter.weight._cosmos_tp_partial_grad = True
    stage = SimpleNamespace(layers=nn.ModuleDict({"0": layer}), input_adapter=adapter)
    assert clear_native_block_partial_flags(stage) == 2
    assert not any(parameter._cosmos_tp_partial_grad for parameter in layer.parameters())
    assert adapter.weight._cosmos_tp_partial_grad
