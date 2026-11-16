# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""A skipped GA window must preserve Adam's parameter and moment trajectory."""

import copy
from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.callbacks.skip_nan_step as skip_module
from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.callbacks.skip_nan_step import SkipNaNStep
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils.generator.fused_adam import FusedAdam


@pytest.mark.gpus(1)
@pytest.mark.parametrize("optimizer_class", [torch.optim.AdamW, FusedAdam], ids=["adamw", "fused"])
@pytest.mark.parametrize("scaler_enabled", [False, True], ids=["no_scaler", "scaler"])
@pytest.mark.parametrize("bad_microbatch", [0, 1, 2])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_all_bad_ga_window_preserves_optimizer_and_recovers(
    monkeypatch, optimizer_class, scaler_enabled, bad_microbatch, bad_value
):
    # The collective decision is isolated here; a separate real-rank probe
    # verifies agreement. GradScaler's unscale kernel requires FP32 grads here;
    # the no-scaler cases cover production BF16 weights and FP32 master copies.
    monkeypatch.setattr(skip_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(skip_module.dist, "get_world_size", lambda: 1)
    dtype = torch.float32 if scaler_enabled else torch.bfloat16
    model = torch.nn.Linear(2, 1, bias=False, device="cuda", dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(1)
    kwargs = {"master_weights": True, "capturable": True} if optimizer_class is FusedAdam else {}
    optimizer = optimizer_class(model.parameters(), lr=0.1, weight_decay=0.05, **kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 100)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled, growth_interval=4)
    trainer = object.__new__(ImaginaireTrainer)
    callback = SkipNaNStep()
    clip = GradClip()
    inputs = torch.tensor([[1.0, 0.5]], device="cuda", dtype=dtype)

    def accumulate(*, invalid=False):
        for microbatch in range(3):
            loss = model(inputs).float().square().sum()
            if invalid and microbatch == bad_microbatch:
                loss = loss * bad_value
            callback.on_before_backward(model, loss)
            scaler.scale(loss / 3).backward()

    def update(iteration):
        callback.on_before_optimizer_step(model, optimizer, scheduler, scaler, iteration=iteration)
        clip.on_before_optimizer_step(model, optimizer, scheduler, scaler, iteration=iteration)
        trainer._optimizer_step(model, optimizer, scheduler, scaler, iteration=iteration)
        optimizer.zero_grad(set_to_none=True)

    accumulate()
    update(0)
    parameters = [p.detach().clone() for p in model.parameters()]
    state = copy.deepcopy(optimizer.state_dict())
    scale_state = copy.deepcopy(scaler.state_dict())
    assert any(value["exp_avg"].count_nonzero() for value in optimizer.state.values())

    accumulate(invalid=True)
    update(1)
    for observed, expected in zip(model.parameters(), parameters, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    current = optimizer.state_dict()
    torch.testing.assert_close(current["state"], state["state"], rtol=0, atol=0)
    if optimizer_class is FusedAdam:
        assert current["param_groups"][0]["step"] == state["param_groups"][0]["step"]
    assert scaler.state_dict() == scale_state
    assert scheduler.last_epoch == 2
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(0.098)
    assert callback._consecutive_nan_count == 1

    accumulate()
    update(2)
    assert any(not torch.equal(p, expected) for p, expected in zip(model.parameters(), parameters, strict=True))
    assert callback._consecutive_nan_count == 0
    assert scheduler.last_epoch == 3


def test_partial_and_all_rank_nonfinite_counts_use_existing_collective(monkeypatch):
    callback = skip_module.SkipNaNStep()
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=2))
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    original_tensor = torch.tensor
    monkeypatch.setattr(skip_module.torch, "tensor", lambda value, **_: original_tensor(value))
    reduced = iter([2, 4])
    calls = []

    def all_reduce(tensor, **_):
        calls.append(1)
        tensor.fill_(next(reduced))

    monkeypatch.setattr(skip_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(skip_module.dist, "get_world_size", lambda: 4)
    warnings = []
    monkeypatch.setattr(skip_module.log, "warning", lambda message, **_: warnings.append(message))
    logged = []
    monkeypatch.setattr(skip_module.wandb, "run", object())
    monkeypatch.setattr(skip_module.wandb, "log", lambda values, **_: logged.append(values))

    callback.on_before_backward(model, original_tensor(float("nan")))
    callback.on_before_optimizer_step(model, None, None, None, iteration=1)
    assert model.weight.grad.item() == 1.0
    callback.on_before_backward(model, original_tensor(float("inf")))
    callback.on_before_optimizer_step(model, None, None, None, iteration=2)
    assert model.weight.grad.item() == 0.0
    callback.on_training_step_end(model, {}, {}, original_tensor(0), iteration=2)

    assert len(calls) == 2
    assert any("2/4 ranks affected" in message for message in warnings)
    assert any("Local rank observed" in message for message in warnings)
    assert logged[0]["loss_nonfinite/steps"] == 2
    assert logged[0]["loss_nonfinite/affected_rank_steps"] == 6
    assert logged[0]["loss_nonfinite/all_rank_steps"] == 1
    assert callback._loss_nonfinite_counts == {"steps": 0, "affected_rank_steps": 0, "all_rank_steps": 0}
