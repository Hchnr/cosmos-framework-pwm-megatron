# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

import pytest
import torch

from tools.tp_pp_tensor_metrics import compare_tensor_metrics


def test_metrics_match_hand_calculation_and_chunking():
    ref = torch.tensor([3.0, 4.0, 0.0])
    candidate = torch.tensor([0.0, 4.0, 0.0])
    result = compare_tensor_metrics(ref, candidate, chunk_size=1)
    assert result["status"] == "ok"
    assert result["relative_l2"] == pytest.approx(3 / 5)
    assert result["max_abs"] == 3
    assert result["cosine"] == pytest.approx(4 / 5)
    assert result["bitwise_equal"] is False
    assert result == compare_tensor_metrics(ref, candidate, chunk_size=10)


def test_zero_norm_and_empty_tensors_are_explicit():
    zero = torch.zeros(4)
    both_zero = compare_tensor_metrics(zero, zero)
    assert both_zero["bitwise_equal"] is True
    assert both_zero["reference_norm_zero"] is True
    assert both_zero["candidate_norm_zero"] is True
    assert both_zero["relative_l2"] is None and both_zero["cosine"] is None
    assert both_zero["max_abs"] == 0
    nonzero_candidate = compare_tensor_metrics(zero, torch.ones(4))
    assert nonzero_candidate["relative_l2"] is None and nonzero_candidate["max_abs"] == 1
    nonzero_reference = compare_tensor_metrics(torch.ones(4), zero)
    assert nonzero_reference["relative_l2"] == pytest.approx(1)
    assert nonzero_reference["cosine"] is None
    empty = compare_tensor_metrics(torch.empty(0), torch.empty(0))
    assert empty["numel"] == 0 and empty["bitwise_equal"] is True and empty["max_abs"] == 0


def test_bitwise_comparison_observes_signed_zero_and_noncontiguous_logical_values():
    positive = torch.tensor([0.0, 1.0])
    negative = torch.tensor([-0.0, 1.0])
    result = compare_tensor_metrics(positive, negative)
    assert result["bitwise_equal"] is False
    assert result["relative_l2"] == 0 and result["max_abs"] == 0
    view = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not view.is_contiguous()
    with pytest.raises(ValueError, match="noncontiguous"):
        compare_tensor_metrics(view, view.contiguous())
    assert compare_tensor_metrics(view, view.contiguous(), allow_noncontiguous_copy=True)["bitwise_equal"] is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_scalar_and_float_dtypes(dtype):
    value = torch.tensor(2.0, dtype=dtype)
    result = compare_tensor_metrics(value, value.clone())
    assert result["bitwise_equal"] is True and result["relative_l2"] == 0
    assert result["max_abs"] == 0 and result["cosine"] == pytest.approx(1)


def test_nonfinite_values_fail_numerical_comparison_even_when_bytes_match():
    ref = torch.tensor([float("nan"), float("inf")])
    result = compare_tensor_metrics(ref, ref.clone())
    assert result["status"] == "nonfinite" and result["bitwise_equal"] is True
    assert result["relative_l2"] is None and result["max_abs"] is None and result["cosine"] is None
    other = compare_tensor_metrics(ref, torch.zeros(2))
    assert other["status"] == "nonfinite" and other["bitwise_equal"] is False


def test_rejects_shape_dtype_and_invalid_chunk():
    with pytest.raises(ValueError, match="shape and dtype"):
        compare_tensor_metrics(torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="shape and dtype"):
        compare_tensor_metrics(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating"):
        compare_tensor_metrics(torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="chunk_size"):
        compare_tensor_metrics(torch.zeros(2), torch.zeros(2), chunk_size=0)


def test_large_and_tiny_finite_values_do_not_overflow_squares():
    for value in (1e-300, 1e300):
        ref = torch.tensor([value, -value], dtype=torch.float64)
        other = torch.tensor([value / 2, -value / 2], dtype=torch.float64)
        result = compare_tensor_metrics(ref, other, chunk_size=1)
        assert result["status"] == "ok"
        assert result["relative_l2"] == pytest.approx(0.5)
        assert result["cosine"] == pytest.approx(1)
        assert math.isfinite(result["max_abs"])


def test_unrepresentable_absolute_difference_is_not_reported_as_finite():
    ref = torch.tensor([1e308], dtype=torch.float64)
    other = -ref
    result = compare_tensor_metrics(ref, other)
    assert result["status"] == "arithmetic_overflow"
    assert result["relative_l2"] == pytest.approx(2) and result["max_abs"] is None
    assert result["cosine"] == pytest.approx(-1)


def test_representable_extreme_ratio_avoids_intermediate_overflow():
    ref = torch.tensor([1e-308, 1e-308], dtype=torch.float64)
    other = torch.tensor([2.0, 0.0], dtype=torch.float64)
    result = compare_tensor_metrics(ref, other, chunk_size=1)
    assert result["status"] == "ok"
    assert result["relative_l2"] == pytest.approx(math.sqrt(2) * 1e308)


def test_unrepresentable_relative_error_is_not_reported_as_zero():
    ref = torch.tensor([1e300, 0.0], dtype=torch.float64)
    other = torch.tensor([1e300, 1e-300], dtype=torch.float64)
    result = compare_tensor_metrics(ref, other)
    assert result["status"] == "arithmetic_underflow"
    assert result["relative_l2"] is None and result["max_abs"] == 1e-300
