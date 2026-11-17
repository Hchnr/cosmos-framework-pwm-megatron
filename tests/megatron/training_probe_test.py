"""Ensure resumed measurements do not overwrite checkpoint RNG or iteration."""

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
