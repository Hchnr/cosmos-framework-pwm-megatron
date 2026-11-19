# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared-input SUM preserves unused branches and reduces only after accumulation."""

import copy

import pytest
import torch


@pytest.mark.parametrize("active", [(0, 1, 2), (0, 2), ()])
def test_shared_sum_preserves_gradients_and_none_edges(monkeypatch, active):
    from cosmos_framework.model.generator.megatron_tp_pp import compile as communication

    generator = torch.Generator().manual_seed(52)
    values = torch.randn(5, 7, dtype=torch.float64, generator=generator)
    modules = torch.nn.ModuleList([torch.nn.Linear(7, 3, bias=False, dtype=torch.float64) for _ in range(3)])
    for module in modules:
        with torch.no_grad():
            module.weight.copy_(torch.randn(3, 7, dtype=torch.float64, generator=generator))
    candidates = copy.deepcopy(modules)
    calls = []

    def sum_across_two_ranks(gradient, group):
        assert group == "qualification"
        calls.append(gradient.detach().clone())
        return gradient * 2

    monkeypatch.setattr(communication, "_sum", sum_across_two_ranks)
    observations = []
    for shared, layers in ((False, modules), (True, candidates)):
        calls.clear()
        inputs = values.clone().requires_grad_()
        common = communication.CopyToTensorParallel.apply(inputs, "qualification") if shared else None
        outputs = [
            layer(common if shared else communication.CopyToTensorParallel.apply(inputs, "qualification"))
            for layer in layers
        ]
        anchor = torch.zeros((), dtype=torch.float64, requires_grad=True)
        sum((outputs[index].square().sum() for index in active), anchor).backward()
        observations.append((inputs.grad, len(calls)))
        assert all((layer.weight.grad is None) == (index not in active) for index, layer in enumerate(layers))
    expected, actual = observations
    assert expected[1] == len(active)
    assert actual[1] == int(bool(active))
    if active:
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-14, atol=1e-14)
    else:
        assert expected[0] is None and actual[0] is None
    for source, candidate in zip(modules, candidates, strict=True):
        if source.weight.grad is not None:
            torch.testing.assert_close(source.weight.grad, candidate.weight.grad, rtol=0, atol=0)
