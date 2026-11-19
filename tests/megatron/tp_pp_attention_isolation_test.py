# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tracing preserves gradients/RNG and distinguishes full from sampled bytes."""

import copy
import hashlib
import json

import torch
from torch import nn

from tools.trace_tp_pp_attention_isolation import Recorder


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
