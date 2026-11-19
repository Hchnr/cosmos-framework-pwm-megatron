# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tools.tp_pp_fused_input_projection import (
    FusedInputMLP,
    FusionObservation,
    fused_local_linear,
    ineligibility_reasons,
)


def _compare(compiled, shape, connected, trainable, input_trainable):
    torch.manual_seed(7)
    inputs = torch.randn(shape, dtype=torch.float64, requires_grad=input_trainable)
    reference_inputs = inputs.detach().clone().requires_grad_(input_trainable)
    weights = [
        torch.randn(width, shape[-1], dtype=torch.float64, requires_grad=flag)
        for width, flag in zip((7, 3, 2), trainable, strict=True)
    ]
    reference_weights = [weight.detach().clone().requires_grad_(weight.requires_grad) for weight in weights]
    fused = fused_local_linear
    if compiled:
        fused = torch.compile(fused, backend="eager", fullgraph=True)
    actual = fused(inputs, *weights)
    expected = tuple(F.linear(reference_inputs, weight) for weight in reference_weights)
    assert all(torch.allclose(a, b, rtol=1e-12, atol=1e-12) for a, b in zip(actual, expected, strict=True))
    scales = [torch.randn_like(output) for output in actual]
    loss = sum((actual[i] * scales[i]).sum() for i in connected)
    reference_loss = sum((expected[i] * scales[i]).sum() for i in connected)
    loss.backward()
    reference_loss.backward()
    for actual_tensor, expected_tensor in zip((inputs, *weights), (reference_inputs, *reference_weights), strict=True):
        assert (actual_tensor.grad is None) == (expected_tensor.grad is None)
        if actual_tensor.grad is not None:
            torch.testing.assert_close(actual_tensor.grad, expected_tensor.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", [(8, 4), (2, 3, 4)])
@pytest.mark.parametrize(
    "connected,trainable,input_trainable",
    [
        ((0, 1, 2), (True, True, True), True),
        ((0, 2), (True, True, True), True),
        ((1,), (True, False, True), True),
        ((0, 2), (True, True, True), False),
        ((0, 1), (False, True, False), True),
    ],
)
def test_fused_local_matches_individual_linears_and_none_gradients(shape, connected, trainable, input_trainable):
    _compare(False, shape, connected, trainable, input_trainable)


def test_fused_local_compiles_with_disconnected_branch():
    _compare(True, (3, 4), (0, 2), (True, True, True), True)


def test_aot_compiled_graph_preserves_internally_disconnected_weight():
    def loss_fn(inputs, *weights):
        output = fused_local_linear(inputs, *weights)
        return output[0].sum() + output[2].sum()

    compiled = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    inputs = torch.randn(3, 4, requires_grad=True)
    weights = [torch.randn(width, 4, requires_grad=True) for width in (7, 3, 2)]
    compiled(inputs, *weights).backward()
    assert weights[0].grad is not None and weights[2].grad is not None
    assert weights[1].grad is None


def test_adamw_skips_missing_branch_with_prior_momentum_under_aot():
    def loss_fn(inputs, w0, w1, w2, include_middle):
        y = fused_local_linear(inputs, w0, w1, w2)
        return y[0].sum() + y[2].sum() + (y[1].sum() if include_middle else 0)

    compiled = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    inputs = torch.randn(3, 4)
    weights = [torch.nn.Parameter(torch.randn(n, 4)) for n in (7, 3, 2)]
    optimizer = torch.optim.AdamW(weights, lr=0.01, weight_decay=0.1)
    compiled(inputs, *weights, True).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_weight = weights[1].detach().clone()
    old_step = optimizer.state[weights[1]]["step"].clone()
    compiled(inputs, *weights, False).backward()
    assert weights[1].grad is None
    optimizer.step()
    assert torch.equal(weights[1], old_weight)
    assert torch.equal(optimizer.state[weights[1]]["step"], old_step)


def test_fused_mlp_preserves_parameter_objects_and_names():
    original = torch.nn.Module()
    original.gate_proj = torch.nn.Linear(4, 8, bias=False)
    original.up_proj = torch.nn.Linear(4, 8, bias=False)
    original.down_proj = torch.nn.Linear(8, 4, bias=False)
    original.act_fn = F.silu
    replacement = FusedInputMLP(original)
    assert dict(replacement.named_parameters()).keys() == dict(original.named_parameters()).keys()
    assert all(dict(replacement.named_parameters())[k] is v for k, v in original.named_parameters())


def test_ineligibility_reports_actual_tp_constraints():
    class Projection:
        def __init__(self, group):
            self.column = True
            self.tp_size = 2
            self._impl = SimpleNamespace(tp_group=group)
            self.bias = None
            self.weight = torch.randn(4, 3)

    group = object()
    a, b = Projection(group), Projection(group)
    assert ineligibility_reasons((a, b), projection_type=Projection) == []
    b.bias = torch.randn(4)
    b._impl.tp_group = object()
    b.weight = torch.randn(4, 5)
    assert set(ineligibility_reasons((a, b), projection_type=Projection)) == {
        "bias_present",
        "different_tp_group",
        "different_input_width",
    }
    assert ineligibility_reasons((), projection_type=Projection) == ["missing_projection_or_module"]


def test_fusion_report_requires_requested_modules_and_preserves_failure(tmp_path):
    observation = FusionObservation(tmp_path, 2, fuse_qkv=True, fuse_gate_up=True)
    stage = observation.start_stage()
    assert observation.record(stage, "layers.0.self_attn", "qkv", (), reasons=[])
    assert not observation.record(stage, "layers.0.mlp", "gate_up", (), reasons=["bias_present"])
    with pytest.raises(RuntimeError, match="gate_up"):
        observation.require_enabled()
    with pytest.raises(RuntimeError, match="gate_up"):
        observation.finish()
    report = json.loads((tmp_path / "fused_input_projection.rank2.json").read_text())
    assert not report["completed"] and "gate_up" in report["failure"]
    assert report["enabled_counts"] == {"qkv": 1, "gate_up": 0}
    assert report["ineligible_reason_counts"]["gate_up"] == {"bias_present": 1}
    assert report["stages"][0]["modules"][0]["module"] == "layers.0.self_attn"
    with pytest.raises(FileExistsError):
        FusionObservation(tmp_path, 2, fuse_qkv=True, fuse_gate_up=True)


def test_fusion_report_requested_subset_and_success(tmp_path):
    observation = FusionObservation(tmp_path, 1, fuse_qkv=True, fuse_gate_up=False)
    stage = observation.start_stage()
    assert observation.record(stage, "q", "qkv", (), reasons=[])
    assert not observation.record(stage, "g", "gate_up", (), reasons=[])
    observation.require_enabled()
    observation.finish()
    report = json.loads((tmp_path / "fused_input_projection.rank1.json").read_text())
    assert report["completed"] and report["failure"] is None
    assert report["enabled_counts"] == {"qkv": 1, "gate_up": 0}
