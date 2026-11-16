# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest
import torch

from cosmos_framework.utils.generator.fused_adam import FusedAdam


def test_fp32_state_load_is_lossless_and_reuses_storage():
    param = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    optimizer = FusedAdam.__new__(FusedAdam)
    torch.optim.Optimizer.__init__(optimizer, [param], {"lr": 0.01})
    optimizer.capturable = False
    values = torch.tensor([0.12345678, 1.0012345, 0.0001234567])
    saved = optimizer.state_dict()
    saved["state"][0] = {key: values.clone() for key in ("exp_avg", "exp_avg_sq", "master_param")}
    seen = []
    optimizer.register_load_state_dict_post_hook(lambda opt: seen.append(opt.state[param]["exp_avg"].dtype))
    optimizer.load_state_dict(saved)
    assert seen == [torch.float32]
    for key, tensor in saved["state"][0].items():
        assert torch.equal(optimizer.state[param][key], values)
        assert optimizer.state[param][key].data_ptr() == tensor.data_ptr()
    assert len(optimizer._optimizer_load_state_dict_pre_hooks) == 0
    assert len(optimizer._optimizer_load_state_dict_post_hooks) == 1


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FusedAdam kernel requires CUDA")
def test_bf16_master_checkpoint_matches_uninterrupted_update(monkeypatch):
    # Single-rank test; distributed group-step broadcast is an identity.
    monkeypatch.setattr("cosmos_framework.utils.generator.fused_adam.distributed.broadcast", lambda *args: None)
    initial = torch.linspace(-1, 1, 257, device="cuda", dtype=torch.bfloat16)
    param = torch.nn.Parameter(initial.clone())
    optimizer = FusedAdam([param], lr=0.00013, capturable=True, master_weights=True)
    for step in range(3):
        param.grad = torch.full_like(param, 0.123 + step * 0.017)
        optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = FusedAdam([restored_param], capturable=True, master_weights=True)
    restored.load_state_dict(checkpoint)
    for step in range(3):
        for p, opt in ((param, optimizer), (restored_param, restored)):
            p.grad = torch.full_like(p, 0.089 + step * 0.013)
            opt.step()
        assert torch.equal(param, restored_param)
        for key in ("exp_avg", "exp_avg_sq", "master_param"):
            assert torch.equal(optimizer.state[param][key], restored.state[restored_param][key])
        assert restored.param_groups_master[0]["params"][0] is restored.state[restored_param]["master_param"]
