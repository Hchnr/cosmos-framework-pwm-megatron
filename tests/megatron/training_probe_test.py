"""Ensure resumed measurements do not overwrite checkpoint RNG or iteration."""

from types import SimpleNamespace

import full_depth_gradient_audit
import pytest
import torch
from train_real_batch_smoke import StepMeasurement


def test_resumed_measurement_preserves_rng(monkeypatch):
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda value, **kwargs: tensor(value, dtype=kwargs.get("dtype")))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    measurement = StepMeasurement(reset_seed=False, expected_start_iteration=1200)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        restored_rng = torch.get_rng_state().clone()
        measurement.on_train_start(None, iteration=1200)
        assert torch.equal(torch.get_rng_state(), restored_rng)
    assert measurement.initial_iteration == 1200


def test_measurement_refuses_wrong_restored_iteration():
    measurement = StepMeasurement(reset_seed=False, expected_start_iteration=1200)
    with pytest.raises(ValueError, match="Restored iteration 0, expected 1200"):
        measurement.on_train_start(None, iteration=0)


def test_fp32_gradient_audit_rejects_tf32_before_capture(monkeypatch):
    monkeypatch.setenv("GRADIENT_AUDIT_PRECISION", "float32")
    monkeypatch.delenv("GRADIENT_AUDIT_INPUTS_ONLY", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(ValueError, match="TF32 to remain disabled"):
        full_depth_gradient_audit.capture_then_clip(None, None, None, None, None)


def test_fp32_gradient_audit_rejects_non_fp32_parameters(monkeypatch):
    monkeypatch.setenv("GRADIENT_AUDIT_PRECISION", "float32")
    monkeypatch.delenv("GRADIENT_AUDIT_INPUTS_ONLY", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = SimpleNamespace(net=torch.nn.Linear(2, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="non-FP32 parameter/gradient"):
        full_depth_gradient_audit.capture_then_clip(None, model, None, None, None)
