"""Strict benchmark gradients must be checked before the default sanitization."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks import grad_clip
from cosmos_framework.callbacks.grad_clip import GradClip


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_strict_grad_clip_rejects_invalid_grad_before_sanitizing(value):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[value, 1.0]])
    callback = GradClip(error_if_nonfinite=True)
    with pytest.raises(RuntimeError, match="non-finite"):
        callback.on_before_optimizer_step(model, None, None, None)
    assert not torch.isfinite(model.weight.grad[0, 0])


def test_default_sanitization_and_strict_finite_clipping_remain_available():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("nan"), 2.0]])
    GradClip().on_before_optimizer_step(model, None, None, None)
    assert torch.isfinite(model.weight.grad).all()
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    GradClip(error_if_nonfinite=True).on_before_optimizer_step(model, None, None, None)
    torch.testing.assert_close(model.weight.grad.norm(), torch.tensor(1.0))


def test_nonfinite_event_reports_local_entries_before_sanitize_and_resets(monkeypatch):
    messages = []
    logged = []
    monkeypatch.setattr(grad_clip.log, "warning", lambda message, **_: messages.append(message))
    monkeypatch.setattr(grad_clip.wandb, "run", object())
    monkeypatch.setattr(grad_clip.wandb, "log", lambda values, **_: logged.append(values))
    model = torch.nn.Linear(3, 1, bias=False)
    model.weight.grad = torch.tensor([[float("nan"), float("inf"), 2.0]])
    callback = GradClip()
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=1))

    callback.on_before_optimizer_step(model, None, None, None, iteration=7)
    callback.on_training_step_end(model, {}, {}, torch.tensor(0.0), iteration=7)

    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.0, 0.0, 1.0]]))
    assert "'local_nan_elements': 1" in messages[0]
    assert "'local_inf_elements': 1" in messages[0]
    assert "'parameter': 'weight'" in messages[0]
    assert logged[-1]["grad_nonfinite/norm_steps_rank0"] == 1
    assert logged[-1]["grad_nonfinite/sanitize_steps_rank0"] == 1
    assert logged[-1]["grad_nonfinite/local_nan_elements_rank0"] == 1

    model.weight.grad = torch.tensor([[0.1, 0.0, 0.0]])
    callback.on_before_optimizer_step(model, None, None, None, iteration=8)
    callback.on_training_step_end(model, {}, {}, torch.tensor(0.0), iteration=8)
    assert logged[-1]["grad_nonfinite/norm_steps_rank0"] == 0
    assert logged[-1]["grad_nonfinite/sanitize_steps_rank0"] == 0


def test_failed_nonfinite_inspection_does_not_block_existing_sanitization(monkeypatch):
    def fail_inspection(*_):
        raise RuntimeError("unsupported layout")

    messages = []
    monkeypatch.setattr(grad_clip, "_local_nonfinite_grad_summary", fail_inspection)
    monkeypatch.setattr(grad_clip.log, "warning", lambda message, **_: messages.append(message))
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("nan"), 2.0]])
    callback = GradClip()

    callback.on_before_optimizer_step(model, None, None, None, iteration=1)

    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.0, 1.0]]))
    assert callback._nonfinite_states[""]["inspection_failures"] == 1
    assert "inspection_error" in messages[0]
