# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest
import torch
from torch import nn

from tools.run_tp_pp_flat_checkpoint_math import flat_checkpoint


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_nested_inputs_are_visible_to_saved_tensor_offload(use_checkpoint):
    token = object()
    calls, packed = [], []
    model = nn.Linear(4, 4).double()
    inputs = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    masks = torch.randn(5, 4, dtype=torch.float64)

    def block(pack, position, *, metadata):
        calls.append(tuple(pack))
        assert tuple(pack) == ("video", "nested", "none")
        assert pack["none"] is None and metadata["token"] is token
        assert position == ("fixed", None)
        return model(pack["video"]) * pack["nested"][0]["mask"]

    def pack_tensor(tensor):
        packed.append((tuple(tensor.shape), tensor.detach().clone()))
        return tensor.detach().clone()

    with torch.autograd.graph.saved_tensors_hooks(pack_tensor, lambda tensor: tensor):
        pack = {"video": inputs, "nested": [{"mask": masks}], "none": None}
        output = (
            flat_checkpoint(block, pack, ("fixed", None), metadata={"token": token})
            if use_checkpoint
            else torch.utils.checkpoint.checkpoint(
                block,
                pack,
                ("fixed", None),
                metadata={"token": token},
                use_reentrant=False,
            )
        )
        output.square().sum().backward()
    # Native checkpoint captures a dict directly and skips the hooks for its
    # nested inputs. Flattening exposes both nested tensors before recompute.
    saved = [value for shape, value in packed if shape == (5, 4)]
    assert any(torch.equal(value, inputs.detach()) for value in saved) is use_checkpoint
    assert any(torch.equal(value, masks) for value in saved) is use_checkpoint
    assert len(calls) == 2


def test_flat_checkpoint_preserves_outputs_gradients_and_rng_with_nested_inputs():
    torch.manual_seed(314)
    native = nn.Sequential(
        nn.Linear(4, 7), nn.SiLU(), nn.Dropout(0.25), nn.Linear(7, 4)
    ).double()
    candidate = copy.deepcopy(native)
    inputs = torch.randn(5, 4, dtype=torch.float64)
    left, right = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
    context = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    left_context, right_context = (
        context.detach().clone().requires_grad_(),
        context.detach().clone().requires_grad_(),
    )
    opaque = object()

    def run(model, x, c, checkpointed):
        def block(pack, *, tags):
            assert tuple(pack) == ("video", "meta", "none")
            assert pack["none"] is None and pack["meta"][1] is opaque
            assert tags == ("alpha", None, 17)
            value = model(pack["video"]) + pack["meta"][0]["context"]
            return {"tokens": value, "marker": None}

        torch.manual_seed(9821)
        pack = {"video": x, "meta": [{"context": c}, opaque], "none": None}
        output = (
            flat_checkpoint(block, pack, tags=("alpha", None, 17))
            if checkpointed
            else block(pack, tags=("alpha", None, 17))
        )
        output["tokens"].square().sum().backward()
        return output, torch.random.get_rng_state().clone()

    expected, expected_rng = run(native, left, left_context, False)
    actual, actual_rng = run(candidate, right, right_context, True)
    assert actual["marker"] is None
    torch.testing.assert_close(actual["tokens"], expected["tokens"], rtol=0, atol=0)
    torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
    torch.testing.assert_close(right_context.grad, left_context.grad, rtol=0, atol=0)
    for actual_param, expected_param in zip(
        candidate.parameters(), native.parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_param.grad, expected_param.grad, rtol=0, atol=0
        )
    assert torch.equal(actual_rng, expected_rng)
