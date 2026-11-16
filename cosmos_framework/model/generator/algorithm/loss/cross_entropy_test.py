# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.model.generator.algorithm.loss.cross_entropy import (
    cross_entropy_loss,
    packed_cross_entropy_loss,
    weighted_cross_entropy_loss,
)
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX


def test_all_ignored_vlm_targets_skip_ce_with_finite_zero_gradient() -> None:
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.full((2, 4), IGNORE_INDEX, dtype=torch.long)
    for loss_fn in (cross_entropy_loss, lambda output, target: weighted_cross_entropy_loss(output, target, 0.5)):
        loss = loss_fn(logits, labels)
        assert loss.item() == 0.0
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_empty_packed_generation_targets_skip_ce() -> None:
    logits = torch.empty(0, 7, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)
    loss = packed_cross_entropy_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_packed_generation_ce_uses_only_explicit_targets() -> None:
    logits = torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1, 0])
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(packed_cross_entropy_loss(logits, labels), expected)
