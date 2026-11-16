"""Numerical coverage for allocation-free activation statistics."""

import pytest
import torch

from cosmos_framework.callbacks.norm_monitor import NormMonitor


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_norm_statistics_with_strided_input(dtype):
    torch.manual_seed(42)
    value = torch.randn(32, 128).to(dtype).T
    stats = NormMonitor()._compute_l2_stats(value)
    torch.testing.assert_close(stats["sq_sum"], value.float().square().sum(), rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(stats["max"], value.abs().max(), rtol=0, atol=0)


def test_empty_and_nonfinite_statistics():
    monitor = NormMonitor()
    stats = monitor._compute_l2_stats(torch.empty(0))
    assert stats["sq_sum"].item() == stats["max"].item() == 0
    stats = monitor._compute_l2_stats(torch.tensor([1.0, -float("inf")]))
    assert torch.isinf(stats["sq_sum"]) and torch.isinf(stats["max"])
    stats = monitor._compute_l2_stats(torch.tensor([1.0, float("nan")]))
    assert torch.isnan(stats["sq_sum"]) and torch.isnan(stats["max"])
