# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tools.run_tp_pp_checkpointed_chunked_mlp_math import checkpointed_chunks


@pytest.mark.parametrize("outer_checkpoint", [False, True])
def test_chunk_checkpoint_preserves_output_and_gradients(outer_checkpoint):
    torch.manual_seed(12)
    reference = torch.nn.Sequential(torch.nn.Linear(7, 13), torch.nn.SiLU(), torch.nn.Linear(13, 5))
    candidate = copy.deepcopy(reference)
    original_input = torch.randn(19, 7, requires_grad=True)
    candidate_input = original_input.detach().clone().requires_grad_()
    expected = torch.cat([reference(part) for part in original_input.split(6)], dim=0)

    def wrapped(value):
        return checkpointed_chunks(candidate, value, 6)

    actual = checkpoint(wrapped, candidate_input, use_reentrant=False) if outer_checkpoint else wrapped(candidate_input)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate_input.grad, original_input.grad, rtol=1e-6, atol=1e-6)
    for actual_param, expected_param in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-6, atol=1e-6)


def test_chunk_checkpoint_saves_less_than_plain_chunking():
    torch.manual_seed(13)
    network = torch.nn.Sequential(torch.nn.Linear(7, 128), torch.nn.SiLU(), torch.nn.Linear(128, 7))
    inputs = torch.randn(31, 7, requires_grad=True)

    def count_saved(function):
        sizes = []

        def pack(tensor):
            sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            function()
        return sum(sizes)

    plain = count_saved(lambda: torch.cat([network(part) for part in inputs.split(8)]))
    recomputed = count_saved(lambda: checkpointed_chunks(network, inputs, 8))
    assert recomputed < plain


@pytest.mark.parametrize("shape,chunk", [((2, 3, 4), 2), ((4, 3), 0), ((4, 3), -1)])
def test_invalid_shape_or_chunk_rejected(shape, chunk):
    with pytest.raises(ValueError):
        checkpointed_chunks(lambda value: value, torch.ones(shape), chunk)
