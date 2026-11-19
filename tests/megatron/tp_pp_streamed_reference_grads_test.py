# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.build_tp_pp_streamed_reference_helper import transform
from tools.compare_tp_pp_native_reference_streamed import compare, tensor_exact
from tools.tp_pp_streamed_reference_grads import StreamedReferenceGradients


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Linear(4, 3)
        self.optional = nn.Linear(4, 3)
        self.frozen = nn.Linear(4, 3)
        self.frozen.requires_grad_(False)

    def forward(self, x, use_optional):
        result = self.main(x) + self.frozen(x)
        return result + self.optional(x) if use_optional else result


def test_streamed_accumulation_clipping_and_adam_match_across_four_updates():
    torch.manual_seed(71)
    reference = Model()
    streamed = copy.deepcopy(reference)
    reference_optimizer = torch.optim.AdamW(
        (p for p in reference.parameters() if p.requires_grad),
        lr=0.002,
        weight_decay=0.03,
        foreach=False,
    )
    streamed_optimizer = torch.optim.AdamW(
        (p for p in streamed.parameters() if p.requires_grad),
        lr=0.002,
        weight_decay=0.03,
        foreach=False,
    )
    for index, (left, right) in enumerate(
        zip(
            reference_optimizer.param_groups[0]["params"],
            streamed_optimizer.param_groups[0]["params"],
            strict=True,
        )
    ):
        generator = torch.Generator().manual_seed(1100 + index)
        state = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.randn(left.shape, generator=generator) * 0.01,
            "exp_avg_sq": torch.rand(left.shape, generator=generator) * 0.01 + 0.001,
        }
        reference_optimizer.state[left] = {
            name: value.clone() for name, value in state.items()
        }
        streamed_optimizer.state[right] = {
            name: value.clone() for name, value in state.items()
        }
    streamer = StreamedReferenceGradients(streamed.parameters())
    for update in range(4):
        reference_optimizer.zero_grad(set_to_none=True)
        streamed_optimizer.zero_grad(set_to_none=True)
        streamer.reset()
        for microbatch in range(2):
            torch.manual_seed(900 + update * 2 + microbatch)
            x = torch.randn(5, 4)
            used = update != 1 and (microbatch == 0 or update != 2)
            (reference(x, used).square().mean() / 2).backward()
            (streamed(x, used).square().mean() / 2).backward()
            for expected, actual in zip(
                reference.parameters(), streamed.parameters(), strict=True
            ):
                bank = streamer.raw(actual)
                if expected.grad is None:
                    assert bank is None
                else:
                    assert torch.equal(bank, expected.grad)
                assert actual.grad is None
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        actual_norm = streamer.norm(streamed.parameters())
        assert torch.equal(actual_norm, expected_norm)
        clipped = {}

        def step_one(optimizer):
            for name, parameter in streamed.named_parameters():
                if parameter.grad is not None:
                    clipped[name] = parameter.grad.clone()
            optimizer.step()

        streamer.clip_and_step(streamed_optimizer, 0.3, actual_norm, step_one)
        for name, parameter in reference.named_parameters():
            if parameter.grad is None:
                assert name not in clipped
            else:
                assert torch.equal(clipped[name], parameter.grad)
        reference_optimizer.step()
        for a, b in zip(reference.parameters(), streamed.parameters(), strict=True):
            assert torch.equal(a, b)
            assert (
                reference_optimizer.state[a].keys()
                == streamed_optimizer.state[b].keys()
            )
            for key in reference_optimizer.state[a]:
                assert torch.equal(
                    reference_optimizer.state[a][key], streamed_optimizer.state[b][key]
                )
    streamer.close()


def test_helper_generation_is_unique_compilable_and_marks_scope(tmp_path):
    source = Path(
        "tests/megatron/tp_pp_full_source_math_cuda_compare_correct_pp_report.py"
    ).read_text()
    generated = transform(source)
    assert '"gradient_streaming_candidate": streamer is not None' in generated
    assert '"streamed_cuda_norm_and_clip_candidate"' in generated
    assert "streamer.clip_and_step(" in generated
    with pytest.raises(ValueError, match="Unexpected helper source"):
        transform(generated)


def test_streamed_gradients_reject_nonfinite_norm():
    p = nn.Parameter(torch.ones(3))
    streamer = StreamedReferenceGradients([p])
    (p * float("inf")).sum().backward()
    with pytest.raises(RuntimeError, match="non-finite"):
        streamer.norm([p])
    streamer.close()


def test_strict_reference_comparator_checks_all_updates_and_provenance(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    manifest = {
        "passed": True,
        "names": ["a", "b"],
        "input_hashes": ["input0", "input1"],
        "runtime_constants_sha256": "constants",
        "objective": "native_rf",
        "retain_full_text": True,
        "clipping_norm_device": "cuda",
        "native_optimizer": "native_cuda",
        "gpu_leaf_reference": True,
        "preserve_saved_tensor_dense_layout": True,
        "deterministic_algorithms": True,
        "original_capacity": False,
        "attention_backend": "sdpa",
        "clipping_implementation": "native_grad_clip",
    }
    import json

    (baseline / "manifest.json").write_text(json.dumps(manifest))
    streamed_manifest = dict(
        manifest,
        gradient_streaming_candidate=True,
        clipping_implementation="streamed_cuda_norm_and_clip_candidate",
    )
    (candidate / "manifest.json").write_text(json.dumps(streamed_manifest))
    outputs = [
        {
            "norm": float(update + 1),
            "outputs": [{"value": torch.ones(2)} for _ in range(2)],
        }
        for update in range(3)
    ]
    for directory in (baseline, candidate):
        torch.save(outputs, directory / "outputs.pt")
        for update in range(3):
            for index in range(2):
                item = {
                    key: torch.tensor([float(update), float(index)])
                    for key in ("gradient", "master", "exp_avg", "exp_avg_sq", "step")
                }
                torch.save(item, directory / f"update{update}.parameter{index:04d}.pt")
    result = compare(baseline, candidate, expected_parameters=2, expected_updates=3)
    assert (
        result["passed"]
        and result["compared_parameter_files"] == 6
        and result["compared_tensor_fields"] == 30
    )
    item = torch.load(candidate / "update2.parameter0001.pt", weights_only=True)
    item["exp_avg"][0] += 1
    torch.save(item, candidate / "update2.parameter0001.pt")
    result = compare(baseline, candidate, expected_parameters=2, expected_updates=3)
    assert not result["passed"] and result["mismatches"] == [
        {"update": 2, "name": "b", "field": "exp_avg"}
    ]
    streamed_manifest["input_hashes"][0] = "other"
    (candidate / "manifest.json").write_text(json.dumps(streamed_manifest))
    with pytest.raises(ValueError, match="provenance"):
        compare(baseline, candidate, expected_parameters=2, expected_updates=3)


def test_tensor_comparison_is_bitwise_and_rejects_nonfinite():
    assert not tensor_exact(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not tensor_exact(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
