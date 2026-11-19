# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offloaded reference clipping retains the native coefficient and None grads."""

import torch
from tp_pp_full_source_math import clip_offloaded_gradients


def test_offloaded_clipping_matches_native_formula_without_changing_weights():
    parameters = [torch.nn.Parameter(torch.ones(n)) for n in (2, 1, 3)]
    parameters[0].grad = torch.tensor([3.0, 4.0])
    parameters[1].grad = torch.tensor([12.0])
    expected = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    for original, copied in zip(parameters, expected, strict=True):
        copied.grad = original.grad.clone() if original.grad is not None else None
    expected_norm = torch.nn.utils.clip_grad_norm_(expected, 6.0)
    norm = clip_offloaded_gradients(parameters, 6.0, norm_device="cpu")
    torch.testing.assert_close(norm, expected_norm, rtol=0, atol=0)
    assert norm.item() == 13.0
    for actual, target in zip(parameters, expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        if target.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, target.grad, rtol=0, atol=0)
