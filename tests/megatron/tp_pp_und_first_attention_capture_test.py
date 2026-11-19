# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from tools import trace_tp_pp_attention_isolation as trace
from tools.run_tp_pp_und_first_attention_capture import (
    FirstUndAttentionCaptured,
    add_modules,
    install_witness,
    validate_captured,
)


def test_capture_first_und_attention_and_runtime_weights(tmp_path, monkeypatch):
    for name in ("__init__", "attach", "attend"):
        monkeypatch.setattr(trace.Recorder, name, getattr(trace.Recorder, name))
    monkeypatch.setattr(trace, "MODULE_NAMES", trace.MODULE_NAMES)
    add_modules(trace)
    tower = SimpleNamespace(rotate=lambda x, position: x * position[0][:, None] + position[1][:, None])
    capture_dir = tmp_path / "capture"
    weight_dir = capture_dir / "weights" / "native"
    install_witness(trace, tower, weight_dir)

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(4)
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.self_attn.q_norm = torch.nn.LayerNorm(2)

        def forward(self, inputs):
            values = self.input_layernorm(inputs)
            query = self.self_attn.q_norm(self.self_attn.q_proj(values).view(-1, 2, 2))
            position = (torch.ones(len(inputs), 2), torch.zeros(len(inputs), 2))
            query = tower.rotate(query, position)
            return recorder.attend(lambda q, k, v, m, **kwargs: q, query, query, query, None)

    layer = FakeLayer()
    recorder = trace.Recorder(capture_dir / "native", rank=0, tp_rank=0, tp_size=1, target_forward=0, layers=(0,))
    recorder.attach(layer, [(0, layer)])
    with pytest.raises(FirstUndAttentionCaptured):
        layer(torch.ones(3, 4))
    assert validate_captured(capture_dir, weight_dir, "native", 0) == 17
    saved = torch.load(weight_dir / "self_attn_q_proj.rank0.pt", weights_only=True)
    assert torch.equal(saved["weight"], layer.self_attn.q_proj.weight.detach())
    assert saved["bias"] is None and saved["tp_size"] == 1
    record = torch.load(capture_dir / "native/layer00.und_rotate0.output.rank0.pt", weights_only=False)
    assert record["scope"] == {"kind": "full"}
    assert torch.equal(
        record["value"],
        torch.load(capture_dir / "native/layer00.attention_call0.query.rank0.pt", weights_only=False)["value"],
    )


def test_rejects_duplicate_module_registration(monkeypatch):
    monkeypatch.setattr(trace, "MODULE_NAMES", trace.MODULE_NAMES)
    add_modules(trace)
    with pytest.raises(ValueError):
        add_modules(trace)


def test_incomplete_capture_cannot_report_success(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_captured(tmp_path, tmp_path, "tp", 0)
