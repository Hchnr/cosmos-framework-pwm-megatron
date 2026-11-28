# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tracing preserves gradients/RNG and distinguishes full from sampled bytes."""

import copy
import hashlib
import json
import sys

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from tools import run_tp_pp_traced_fp32_math, trace_tp_pp_attention_isolation
from tools.trace_tp_pp_attention_isolation import Recorder, attach_recorders


def test_second_forward_capture_preserves_gradients_and_skips_recomputation(tmp_path):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj_moe_gen = nn.Linear(4, 4, bias=False)
            self.recorder = None

        def forward(self, inputs):
            query = self.self_attn.q_proj_moe_gen(inputs)

            def original(q, k, v, mask):
                return q + k + v

            return self.recorder.attend(original, query, query * 2, query * 3, None)

    class Network(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, inputs):
            return self.layer(inputs)

    torch.manual_seed(91)
    layer = Layer()
    reference_layer = copy.deepcopy(layer)
    recorder = Recorder(tmp_path, rank=2, tp_rank=0, tp_size=2, max_elements=16, rows=3)
    layer.recorder = recorder
    network = Network(layer)
    recorder.attach(network, [(30, layer)])
    input_value = torch.randn(15, 4)
    with torch.no_grad():
        network(input_value)
    rng = torch.get_rng_state().clone()
    actual_input = input_value.clone().requires_grad_()
    actual = network(actual_input)
    # A checkpoint recomputation calls the layer again without re-entering the
    # root forward. The duplicate must not overwrite or add a second record.
    layer(actual_input)
    actual.square().sum().backward()
    reference_input = input_value.clone().requires_grad_()
    reference = reference_layer.self_attn.q_proj_moe_gen(reference_input) * 6
    reference.square().sum().backward()
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        layer.self_attn.q_proj_moe_gen.weight.grad,
        reference_layer.self_attn.q_proj_moe_gen.weight.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(torch.get_rng_state(), rng)
    files = sorted(tmp_path.glob("*.pt"))
    assert len(files) == 12
    record = torch.load(tmp_path / "layer30.self_attn_q_proj_moe_gen.input.rank2.pt", weights_only=False)
    assert record["forward"] == 1
    assert record["shape"] == [15, 4]
    assert record["dtype"] == "torch.float32"
    assert record["scope"] == {"kind": "first_last_dim0", "rows_each_end": 3}
    expected = torch.cat((input_value[:3], input_value[-3:]))
    torch.testing.assert_close(record["value"], expected, rtol=0, atol=0)
    digest = hashlib.sha256(expected.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert record["stored_sha256"] == digest
    assert (tmp_path / "layer30.attention_call0.output_dgrad.rank2.pt").is_file()
    mask = json.loads((tmp_path / "layer30.attention_call0.mask.rank2.json").read_text())
    assert mask["mask_type"] is None and mask["full_mask_stored"]


def test_small_and_scalar_records_hash_all_values(tmp_path):
    recorder = Recorder(tmp_path, rank=0, tp_rank=0, tp_size=1, max_elements=16)
    recorder.save(torch.tensor(2.5), 31, "scalar", "output")
    data = torch.load(tmp_path / "layer31.scalar.output.rank0.pt", weights_only=False)
    assert data["scope"] == {"kind": "full"}
    assert data["shape"] == []
    assert (
        data["stored_sha256"]
        == hashlib.sha256(data["value"].reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    )


def test_attention_mask_and_dispatch_arguments_are_recorded(tmp_path):
    recorder = Recorder(tmp_path, rank=1, tp_rank=1, tp_size=4, max_elements=4, rows=1)
    recorder.active_layer = 32
    tensor = torch.randn(10, 2)
    mask = torch.eye(10, dtype=torch.bool)

    def original(q, k, v, mask, **kwargs):
        assert kwargs == {"frozen_und": True}
        return q + k + v

    output = recorder.attend(original, tensor, tensor, tensor, mask, frozen_und=True)
    torch.testing.assert_close(output, tensor * 3)
    metadata = json.loads((tmp_path / "layer32.attention_call0.mask.rank1.json").read_text())
    assert metadata["mask_type"] == "torch.Tensor"
    assert metadata["full_mask_stored"] is False
    assert metadata["kwargs"] == {"frozen_und": True}
    stored = torch.load(tmp_path / "layer32.attention_call0.mask.rank1.pt", weights_only=False)
    assert stored["dtype"] == "torch.bool"
    assert stored["scope"] == {"kind": "first_last_dim0", "rows_each_end": 1}


@pytest.mark.parametrize("overlap_forwards", [False, True])
def test_real_checkpoint_recomputation_keeps_target_cotangents(tmp_path, overlap_forwards):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj_moe_gen = nn.Linear(4, 4, bias=False)
            self.recorder = None

        def forward(self, value):
            q = self.self_attn.q_proj_moe_gen(value)
            return self.recorder.attend(lambda q, k, v, mask: q + k + v, q, q * 2, q * 3, None)

    class Network(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = checkpoint_wrapper(layer, preserve_rng_state=True)

        def forward(self, value):
            return self.layer(value)

    torch.manual_seed(8)
    layer = Layer()
    recorder = Recorder(tmp_path, rank=0, tp_rank=0, tp_size=1)
    layer.recorder = recorder
    network = Network(layer)
    recorder.attach(network, [(30, layer)])
    values = [torch.randn(3, 4, requires_grad=True) for _ in range(2)]
    first = network(values[0])
    if not overlap_forwards:
        first.square().sum().backward()
    second = network(values[1])
    if overlap_forwards:
        first.square().sum().backward()
    second.square().sum().backward()

    def captured(tag, kind):
        return torch.load(tmp_path / f"layer30.{tag}.{kind}.rank0.pt", weights_only=False)["value"]

    torch.testing.assert_close(captured("self_attn_q_proj_moe_gen", "input"), values[1])
    torch.testing.assert_close(captured("attention_call0", "output"), second)
    torch.testing.assert_close(captured("attention_call0", "output_dgrad"), second * 2)
    torch.testing.assert_close(captured("self_attn_q_proj_moe_gen", "input_dgrad"), values[1].grad)
    assert recorder.sequence == 1


def test_multiple_forwards_and_pipeline_stage_layer_coverage(tmp_path):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj_moe_gen = nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return self.self_attn.q_proj_moe_gen(value)

    class Stage(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = checkpoint_wrapper(layer, preserve_rng_state=True)

        def forward(self, value):
            return self.layer(value)

    for rank in range(4):
        index = 0 if rank < 2 else 30
        layer = Layer()
        stage = Stage(layer)
        recorders = attach_recorders(
            stage,
            [(index, layer)],
            tmp_path,
            "tp",
            rank=rank,
            tp_rank=rank % 2,
            tp_size=2,
            target_forwards=(0, 1),
            split_by_forward=True,
            max_elements=100,
            rows=2,
            selected_layers=(0, 30, 32),
        )
        values = [torch.full((2, 4), float(i + 1), requires_grad=True) for i in range(2)]
        outputs = [stage(value) for value in values]
        for output in outputs:
            output.square().sum().backward()
        for forward in range(2):
            directory = tmp_path / f"forward{forward:03d}" / "tp"
            saved = torch.load(
                directory / f"layer{index:02d}.self_attn_q_proj_moe_gen.input.rank{rank}.pt", weights_only=False
            )
            assert saved["forward"] == forward
            torch.testing.assert_close(saved["value"], values[forward])
            gradient = torch.load(
                directory / f"layer{index:02d}.self_attn_q_proj_moe_gen.input_dgrad.rank{rank}.pt", weights_only=False
            )
            torch.testing.assert_close(gradient["value"], values[forward].grad)
            assert not list(directory.glob(f"layer32.*.rank{rank}.pt"))
        assert all(recorder.sequence == 1 for recorder in recorders)


@pytest.mark.parametrize("layers", [(), (0, 0), (-1,), ("0",)])
def test_invalid_selected_layers_rejected(tmp_path, layers):
    with pytest.raises(ValueError, match="Layer indices"):
        Recorder(tmp_path, rank=0, tp_rank=0, tp_size=1, layers=layers)


@pytest.mark.parametrize(
    ("extra", "forwards", "split", "layers"),
    [
        ([], (1,), False, (30, 31, 32)),
        (["--forward-indices", "0,1", "--layers", "0,30,31,32"], (0, 1), True, (0, 30, 31, 32)),
    ],
)
def test_cli_preserves_single_forward_and_supports_two_forwards(monkeypatch, tmp_path, extra, forwards, split, layers):
    called = {}
    monkeypatch.setattr(trace_tp_pp_attention_isolation, "install", lambda path, **kwargs: called.update(kwargs))
    monkeypatch.setattr(run_tp_pp_traced_fp32_math, "main", lambda: called.update(downstream_argv=sys.argv[1:]))
    monkeypatch.setattr(
        sys, "argv", ["trace", "--entry", "standard", "--capture-dir", str(tmp_path), *extra, "--", "--mode", "native"]
    )
    trace_tp_pp_attention_isolation.main()
    assert called["target_forwards"] == forwards
    assert called["split_by_forward"] is split
    assert called["layers"] == layers
    assert called["downstream_argv"] == ["--mode", "native"]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_opt_in_branch_edges_preserve_gradients_and_close_residual(tmp_path, checkpointed):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.o_proj_moe_gen = nn.Linear(4, 4, bias=False)
            self.post_attention_layernorm_moe_gen = nn.LayerNorm(4)
            self.mlp_moe_gen = nn.Module()
            self.mlp_moe_gen.gate_proj = nn.Linear(4, 4, bias=False)
            self.mlp_moe_gen.up_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            residual = x + self.self_attn.o_proj_moe_gen(x)
            normalized = self.post_attention_layernorm_moe_gen(residual)
            gate = self.mlp_moe_gen.gate_proj(normalized)
            up = self.mlp_moe_gen.up_proj(normalized)
            return residual + gate * up

    class Network(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = checkpoint_wrapper(layer, preserve_rng_state=True) if checkpointed else layer

        def forward(self, x):
            return self.layer(x)

    torch.manual_seed(12)
    layer = Layer()
    reference = copy.deepcopy(layer)
    model = Network(layer)
    baseline = Network(reference)
    recorder = Recorder(
        tmp_path, rank=0, tp_rank=0, tp_size=1, target_forward=0, layers=(0,), capture_branch_gradients=True
    )
    recorder.attach(model, [(0, layer)])
    inputs = torch.randn(3, 4)
    x = inputs.clone().requires_grad_()
    xr = inputs.clone().requires_grad_()
    rng = torch.get_rng_state().clone()
    actual = model(x)
    expected = baseline(xr)
    dy = (
        torch.randn_like(actual, generator=None)
        if False
        else torch.arange(1, 13, dtype=actual.dtype).reshape_as(actual)
    )
    actual.backward(dy)
    expected.backward(dy)
    assert torch.equal(actual, expected)
    assert torch.equal(x.grad, xr.grad)
    for p, q in zip(layer.parameters(), reference.parameters(), strict=True):
        assert torch.equal(p.grad, q.grad)
    assert torch.equal(torch.get_rng_state(), rng)

    def captured(tag, kind):
        return torch.load(tmp_path / f"layer00.{tag}.{kind}.rank0.pt", weights_only=True)["value"]

    gate = captured("mlp_moe_gen_gate_proj", "branch_input_dgrad")
    up = captured("mlp_moe_gen_up_proj", "branch_input_dgrad")
    norm_out = captured("post_attention_layernorm_moe_gen", "output_dgrad")
    norm_branch = captured("post_attention_layernorm_moe_gen", "branch_input_dgrad")
    attention_out = captured("self_attn_o_proj_moe_gen", "output_dgrad")
    assert torch.equal(gate + up, norm_out)
    assert not torch.equal(gate, up)
    # Independent local VJPs must match each edge, not just their sum.
    residual = (inputs + reference.self_attn.o_proj_moe_gen(inputs)).detach().requires_grad_()
    normalized = reference.post_attention_layernorm_moe_gen(residual)
    gate_value = reference.mlp_moe_gen.gate_proj(normalized)
    up_value = reference.mlp_moe_gen.up_proj(normalized)
    gate_expected = torch.autograd.grad(gate_value, normalized, dy * up_value, retain_graph=True)[0]
    up_expected = torch.autograd.grad(up_value, normalized, dy * gate_value, retain_graph=True)[0]
    norm_expected = torch.autograd.grad(normalized, residual, gate_expected + up_expected)[0]
    assert torch.equal(gate, gate_expected)
    assert torch.equal(up, up_expected)
    assert torch.equal(norm_branch, norm_expected)
    assert torch.equal(norm_branch + dy, attention_out)


def test_branch_capture_interleaved_microbatches_and_missing_file_fails(tmp_path):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm_moe_gen = nn.LayerNorm(4)
            self.mlp_moe_gen = nn.Module()
            self.mlp_moe_gen.gate_proj = nn.Linear(4, 4, bias=False)
            self.mlp_moe_gen.up_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            normalized = self.post_attention_layernorm_moe_gen(x)
            return x + self.mlp_moe_gen.gate_proj(normalized) * self.mlp_moe_gen.up_proj(normalized)

    layer = Layer()
    model = nn.Sequential(checkpoint_wrapper(layer, preserve_rng_state=True))
    attach_recorders(
        model,
        [(0, layer)],
        tmp_path,
        "tp",
        rank=0,
        tp_rank=0,
        tp_size=1,
        target_forwards=(0, 1),
        split_by_forward=True,
        max_elements=1000,
        rows=4,
        selected_layers=(0,),
        capture_branch_gradients=True,
    )
    inputs = [torch.randn(3, 4, requires_grad=True) for _ in range(2)]
    outputs = [model(value) for value in inputs]
    for value in outputs:
        value.square().sum().backward()
    check = trace_tp_pp_attention_isolation.verify_branch_capture(
        tmp_path, kind="tp", forwards=(0, 1), layers=(0,), ranks=(0,)
    )
    assert check == {
        "coverage_passed": True,
        "expected_files": 6,
        "full_tensor_coverage": True,
        "sampled_files": [],
        "numerical_invariance_verified": False,
    }
    for index in range(2):
        folder = tmp_path / f"forward{index:03d}" / "tp"

        def value(tag, kind):
            return torch.load(folder / f"layer00.{tag}.{kind}.rank0.pt", weights_only=True)["value"]

        gate = value("mlp_moe_gen_gate_proj", "branch_input_dgrad")
        up = value("mlp_moe_gen_up_proj", "branch_input_dgrad")
        norm = value("post_attention_layernorm_moe_gen", "output_dgrad")
        assert torch.equal(gate + up, norm)
    corrupt = tmp_path / "forward000/tp/layer00.mlp_moe_gen_up_proj.branch_input_dgrad.rank0.pt"
    record = torch.load(corrupt, weights_only=True)
    record["value"][0, 0] += 1
    torch.save(record, corrupt)
    with pytest.raises(ValueError, match="Invalid branch capture payload"):
        trace_tp_pp_attention_isolation.verify_branch_capture(
            tmp_path, kind="tp", forwards=(0, 1), layers=(0,), ranks=(0,)
        )
    (tmp_path / "forward001/tp/layer00.mlp_moe_gen_up_proj.branch_input_dgrad.rank0.pt").unlink()
    with pytest.raises(ValueError, match="coverage"):
        trace_tp_pp_attention_isolation.verify_branch_capture(
            tmp_path, kind="tp", forwards=(0, 1), layers=(0,), ranks=(0,)
        )


def test_branch_coverage_reports_sampled_tensors(tmp_path):
    folder = tmp_path / "forward000/tp"
    recorder = Recorder(folder, rank=0, tp_rank=0, tp_size=1, target_forward=0, max_elements=4, rows=1)
    for tag in ("post_attention_layernorm_moe_gen", "mlp_moe_gen_gate_proj", "mlp_moe_gen_up_proj"):
        recorder.save(torch.ones(10, 4), 0, tag, "branch_input_dgrad")
    result = trace_tp_pp_attention_isolation.verify_branch_capture(
        tmp_path, kind="tp", forwards=(0,), layers=(0,), ranks=(0,)
    )
    assert result["coverage_passed"] is True
    assert result["full_tensor_coverage"] is False
    assert len(result["sampled_files"]) == 3


def test_single_forward_branch_cli_uses_split_layout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trace_tp_pp_attention_isolation, "install", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(run_tp_pp_traced_fp32_math, "main", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trace",
            "--entry",
            "standard",
            "--capture-dir",
            str(tmp_path),
            "--forward-index",
            "0",
            "--capture-branch-gradients",
        ],
    )
    trace_tp_pp_attention_isolation.main()
    assert calls[0]["split_by_forward"] is True
    assert calls[0]["target_forwards"] == (0,)
    assert calls[0]["capture_branch_gradients"] is True
