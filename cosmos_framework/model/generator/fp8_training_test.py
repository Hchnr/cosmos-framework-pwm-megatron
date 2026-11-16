# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.fp8_training import FP8TrainingConfig
from cosmos_framework.model.generator.fp8_training import (
    convert_fp8_training,
    validate_fp8_optimizer,
    validate_precision_resume,
)


def make_mlp_model(*, device="cpu", dtype=torch.bfloat16, hidden=128):
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP

    config = SimpleNamespace(hidden_size=hidden, intermediate_size=256, hidden_act="silu")
    with torch.device(device):
        net = nn.Module()
        net.language_model = nn.Module()
        net.language_model.model = nn.Module()
        layer = nn.Module()
        layer.mlp_moe_gen = Qwen3VLTextMLP(config)
        layer.mlp = Qwen3VLTextMLP(config)
        net.language_model.model.layers = nn.ModuleList([layer])
    return net.to(dtype=dtype)


def mlp(net):
    return net.language_model.model.layers[0].mlp_moe_gen


def test_disabled_does_not_touch_cuda_or_parameters():
    net = nn.Linear(3, 5)
    before = dict(net.named_parameters())
    with patch("torch.cuda.is_available", side_effect=AssertionError("disabled accessed CUDA")):
        assert convert_fp8_training(net, FP8TrainingConfig()) == {"version": 1, "enabled": False}
    assert dict(net.named_parameters()) == before


@pytest.mark.parametrize("include,match", [([], "nonempty"), (["["], "Invalid"), (["missing"], "No nn.Linear")])
def test_invalid_selection_rejected(include, match):
    with pytest.raises(ValueError, match=match):
        convert_fp8_training(make_mlp_model(), FP8TrainingConfig(enabled=True, include_regex=include))


def test_unsupported_layer_and_shape_rejected_before_mutation():
    with pytest.raises(ValueError, match="plain generation MLP"):
        convert_fp8_training(make_mlp_model(), FP8TrainingConfig(enabled=True, include_regex=[".*"]))
    with pytest.raises(ValueError, match="divisible by 16"):
        convert_fp8_training(make_mlp_model(hidden=127), FP8TrainingConfig(enabled=True))


def test_resume_contract_distinguishes_migration_from_resume():
    current = json.dumps({"enabled": True, "recipe": "rowwise"})
    other = json.dumps({"enabled": False})
    validate_precision_resume(current, None, warm_start=True)
    validate_precision_resume(current, current, warm_start=False)
    validate_precision_resume(current, other, warm_start=True)
    with pytest.raises(ValueError, match="same-job resume"):
        validate_precision_resume(current, other, warm_start=False)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
def test_gpu_meta_conversion_materialization_and_checkpoint(recipe):
    from torchao.float8.float8_linear import Float8Linear

    net = make_mlp_model(device="meta")
    params = dict(net.named_parameters())
    manifest = convert_fp8_training(net, FP8TrainingConfig(enabled=True, recipe=recipe))
    assert len(manifest["targets"]) == 3
    assert all(dict(net.named_parameters())[name] is parameter for name, parameter in params.items())
    net.to_empty(device="cuda")
    torch.manual_seed(123)
    for module in net.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    assert isinstance(mlp(net).gate_proj, Float8Linear)
    assert type(net.language_model.model.layers[0].mlp.gate_proj) is nn.Linear
    plain = make_mlp_model(device="cuda")
    torch.manual_seed(123)
    for module in plain.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, plain.state_dict()[name], rtol=0, atol=0)
    plain.load_state_dict(net.state_dict(), strict=True)
    net.load_state_dict(plain.state_dict(), strict=True)
    assert all(value.dtype == torch.bfloat16 for value in net.state_dict().values())
    assert not list(net.named_buffers())
    optimizer = torch.optim.AdamW(mlp(net).parameters(), lr=1e-4)
    validate_fp8_optimizer(net, optimizer, manifest)
    validate_fp8_optimizer(net, SimpleNamespace(optimizers=[optimizer]), manifest)
    mlp(net).gate_proj.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="not updated"):
        validate_fp8_optimizer(net, SimpleNamespace(optimizers=[optimizer]), manifest)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
def test_gpu_fp8_gemms_execute_in_forward_and_backward(recipe, tmp_path):
    net = make_mlp_model(device="cuda")
    convert_fp8_training(net, FP8TrainingConfig(enabled=True, recipe=recipe))
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Inspect actual dispatched operators, not just the replacement class.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as forward:
        output = mlp(net)(x)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as backward:
        output.float().square().mean().backward()
    counts = {}
    for phase, profile in [("forward", forward), ("backward", backward)]:
        events = {event.key: event.count for event in profile.key_averages()}
        counts[phase] = events
        assert events.get("aten::_scaled_mm", 0) >= 3, events
    (tmp_path / f"{recipe}_operators.json").write_text(json.dumps(counts, indent=2))


@pytest.mark.gpus(1)
@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
@pytest.mark.parametrize("compile_model", [False, True])
def test_gpu_forward_backward_ac_ga2_and_resume(recipe, compile_model, tmp_path):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(42)
    baseline = make_mlp_model(device="cuda")
    net = copy.deepcopy(baseline)
    convert_fp8_training(net, FP8TrainingConfig(enabled=True, recipe=recipe))
    fn = mlp(net)
    if compile_model:
        fn = torch.compile(fn, fullgraph=True, dynamic=False)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    x[-16:] = 0  # physical packing padding
    target = torch.randn_like(x)
    base_x = x.detach().clone().requires_grad_()
    fp8_x = x.detach().clone().requires_grad_()
    base_out = mlp(baseline)(base_x)
    out = checkpoint(fn, fp8_x, use_reentrant=False, preserve_rng_state=True)
    loss = (out.float() - target.float()).square().mean()
    base_loss = (base_out.float() - target.float()).square().mean()
    loss.backward()
    base_loss.backward()
    assert torch.isfinite(out).all()
    assert abs(loss.item() / base_loss.item() - 1) < 0.05
    metrics = {"loss": loss.item(), "bf16_loss": base_loss.item(), "gradients": {}}
    for name, actual, expected in [("input", fp8_x.grad, base_x.grad)] + [
        (name, a.grad, b.grad) for (name, a), b in zip(mlp(net).named_parameters(), mlp(baseline).parameters())
    ]:
        assert torch.isfinite(actual).all()
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
        assert cosine > 0.99
        delta = actual.float() - expected.float()
        metrics["gradients"][name] = {
            "cosine": cosine.item(),
            "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-12)).item(),
            "abs_p99": torch.quantile(delta.abs().flatten(), 0.99).item(),
        }
    (tmp_path / "numeric_metrics.json").write_text(json.dumps(metrics, indent=2))

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, foreach=False)

    def step(model_fn, opt):
        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            value = checkpoint(model_fn, x, use_reentrant=False, preserve_rng_state=True)
            ((value.float() - target.float()).square().mean() / 2).backward()
        opt.step()

    step(fn, optimizer)
    saved = copy.deepcopy({"model": net.state_dict(), "optimizer": optimizer.state_dict()})
    torch.save(saved, tmp_path / "resume.pt")
    step(fn, optimizer)
    restored = make_mlp_model(device="cuda")
    convert_fp8_training(restored, FP8TrainingConfig(enabled=True, recipe=recipe))
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False)
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    restored.load_state_dict(state["model"], strict=True)
    restored_opt.load_state_dict(state["optimizer"])
    restored_fn = torch.compile(mlp(restored), fullgraph=True, dynamic=False) if compile_model else mlp(restored)
    step(restored_fn, restored_opt)
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    for left, right in zip(optimizer.state.values(), restored_opt.state.values()):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
@pytest.mark.parametrize("scale", [0.0, 1e-8])
def test_gpu_zero_and_small_inputs(recipe, scale):
    net = make_mlp_model(device="cuda")
    convert_fp8_training(net, FP8TrainingConfig(enabled=True, recipe=recipe))
    x = (torch.randn(32, 128, device="cuda", dtype=torch.bfloat16) * scale).requires_grad_()
    output = mlp(net)(x)
    output.float().sum().backward()
    for value in [output, x.grad, *(parameter.grad for parameter in mlp(net).parameters())]:
        assert torch.isfinite(value).all()
        if scale == 0:
            assert torch.count_nonzero(value) == 0
