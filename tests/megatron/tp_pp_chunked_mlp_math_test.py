# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
from types import SimpleNamespace

import pytest
import torch

from tools.run_tp_pp_chunked_mlp_math import install


@pytest.mark.parametrize("tokens", [0, 1, 8, 9, 25])
def test_chunking_preserves_mlp_outputs_and_gradients(monkeypatch, tokens):
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP

    torch.manual_seed(42)
    model = Qwen3VLTextMLP(SimpleNamespace(hidden_size=8, intermediate_size=24, hidden_act="silu")).double()
    reference = copy.deepcopy(model)
    original = Qwen3VLTextMLP.forward
    monkeypatch.setattr(Qwen3VLTextMLP, "forward", original)
    inputs = torch.randn(tokens, 8, dtype=torch.float64)
    left = inputs.clone().requires_grad_()
    right = inputs.clone().requires_grad_()
    expected = reference(left)
    expected.square().sum().backward()
    install(8)
    actual = model(right)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(right.grad, left.grad, rtol=1e-13, atol=1e-13)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-13)


def test_chunking_calls_original_with_bounded_token_counts(monkeypatch):
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP

    original = Qwen3VLTextMLP.forward
    seen = []

    def observed(module, inputs):
        seen.append(inputs.shape[0])
        return original(module, inputs)

    monkeypatch.setattr(Qwen3VLTextMLP, "forward", observed)
    install(2)
    model = Qwen3VLTextMLP(SimpleNamespace(hidden_size=8, intermediate_size=24, hidden_act="silu"))
    model(torch.randn(5, 8))
    assert seen == [2, 2, 1]
