# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from tools.analyze_tp_pp_layer_traces import analyze, difference
from tools.run_tp_pp_traced_fp32_math import attach_trace


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, pack):
        return ({"full_only_seq": self.linear(pack["full_only_seq"]).tanh()},)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({str(index): checkpoint_wrapper(Block()) for index in range(2)})

    def forward(self, tensor):
        pack = {"full_only_seq": tensor}
        for block in self.layers.values():
            pack = block(pack)[0]
        return pack["full_only_seq"]


def test_layer_trace_keeps_gradients_rng_and_checkpoint_recomputation(tmp_path):
    torch.manual_seed(73)
    reference = Network()
    traced = copy.deepcopy(reference)
    attach_trace(traced, traced.layers.items(), tmp_path)
    for _ in range(6):
        inputs = torch.randn(4, 8)
        before = torch.get_rng_state().clone()
        expected, actual = reference(inputs), traced(inputs)
        assert torch.equal(expected, actual)
        expected.square().sum().backward()
        actual.square().sum().backward()
        assert torch.equal(before, torch.get_rng_state())
        for original, observed in zip(reference.parameters(), traced.parameters(), strict=True):
            assert torch.equal(original.grad, observed.grad)
    assert len(list(tmp_path.glob("*.outputs.pt"))) == 12
    assert len(list(tmp_path.glob("*.gradient.pt"))) == 12


def test_layer_report_traverses_each_boundary_in_its_direction(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    same = {"shape": [2], "sample": torch.tensor([1.0, 0.0]), "norm": 1.0}
    changed = {"shape": [2], "sample": torch.tensor([0.0, 1.0]), "norm": 1.0}
    stream = "full_only_seq"
    for capture in range(6):
        for layer in range(36):
            stem = f"forward{capture:03d}.layer{layer:02d}"
            diverges = capture == 0 and layer in (1, 7)
            output = changed if diverges else same
            torch.save({stream: same}, reference / f"{stem}.outputs.pt")
            torch.save({stream: output}, candidate / f"{stem}.outputs.pt")
            if diverges:
                torch.save(same, reference / f"{stem}.{stream}.gradient.pt")
                torch.save(output, candidate / f"{stem}.{stream}.gradient.pt")

    report = analyze(reference, candidate)
    first = report["first_observed_difference_by_capture_and_stream"]
    assert first["forward"][f"forward000.{stream}"]["layer"] == 1
    assert first["backward"][f"forward000.{stream}"]["layer"] == 7
    assert first["backward"][f"forward000.{stream}"]["sample_cosine_similarity"] == 0.0
    assert not report["mathematical_acceptance"]


def test_layer_report_uses_absolute_error_for_zero_reference():
    zero = {"shape": [1], "sample": torch.zeros(1), "norm": 0.0}
    nonzero = {"shape": [1], "sample": torch.ones(1), "norm": 1.0}
    result = difference(zero, nonzero)
    assert result["sample_relative_l2"] is None
    assert result["sample_cosine_similarity"] is None
    assert result["sample_maximum_absolute_error"] == 1.0
    assert result["full_norm_relative_difference"] is None
    assert result["full_norm_absolute_difference"] == 1.0
