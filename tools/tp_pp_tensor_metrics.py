# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tensor metrics for differential tests, independent of acceptance policy.

The reference is the denominator for relative L2. Zero reference norm makes
relative L2 undefined; cosine is undefined if either norm is zero. Nonfinite
inputs invalidate numerical metrics. Bitwise equality compares logical element
bytes, including signed zero and NaN payloads, independently of numerical status.
"""

import math

import torch


def compare_tensor_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_size: int = 1 << 20,
    allow_noncontiguous_copy: bool = False,
) -> dict:
    """Compare same-shape, same-dtype CPU floating tensors without changing them.

    Arithmetic is FP64 and scaled before squaring to avoid overflow for normal
    FP32/BF16 model values. Undefined or unrepresentable metrics are ``None``;
    ``status`` distinguishes nonfinite inputs from arithmetic overflow. Exact
    byte comparison uses logical contiguous element order, not storage layout.
    ``chunk_size`` bounds tensor arithmetic workspaces; scalar partial sums
    grow with the number of chunks. Noncontiguous tensors are rejected by
    default; opting in may copy them in full to establish logical element order.
    """
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise TypeError("reference and candidate must be tensors")
    if reference.device.type != "cpu" or candidate.device.type != "cpu":
        raise ValueError("CPU tensors are required")
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("tensor shape and dtype must match")
    if not reference.is_floating_point() or reference.is_complex() or reference.is_quantized:
        raise TypeError("real floating tensors are required")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if type(allow_noncontiguous_copy) is not bool:
        raise TypeError("allow_noncontiguous_copy must be a boolean")
    if not allow_noncontiguous_copy and (not reference.is_contiguous() or not candidate.is_contiguous()):
        raise ValueError("noncontiguous input requires allow_noncontiguous_copy=True")

    a, b = reference.detach().reshape(-1), candidate.detach().reshape(-1)
    result = {
        "status": "ok",
        "numel": a.numel(),
        "bitwise_equal": True,
        "reference_norm_zero": None,
        "candidate_norm_zero": None,
        "relative_l2": None,
        "max_abs": None,
        "cosine": None,
    }
    ref_scale = cand_scale = error_scale = 0.0
    finite = True
    overflow = False
    for start in range(0, a.numel(), chunk_size):
        x = a[start : start + chunk_size].contiguous()
        y = b[start : start + chunk_size].contiguous()
        if not torch.equal(x.view(torch.uint8), y.view(torch.uint8)):
            result["bitwise_equal"] = False
        xd, yd = x.double(), y.double()
        if not bool(torch.isfinite(xd).all() and torch.isfinite(yd).all()):
            finite = False
            continue
        ref_scale = max(ref_scale, xd.abs().max().item())
        cand_scale = max(cand_scale, yd.abs().max().item())
        difference = xd - yd
        if not bool(torch.isfinite(difference).all()):
            overflow = True
        else:
            error_scale = max(error_scale, difference.abs().max().item())
    if not finite:
        result["status"] = "nonfinite"
        return result

    result["reference_norm_zero"] = ref_scale == 0.0
    result["candidate_norm_zero"] = cand_scale == 0.0
    result["max_abs"] = None if overflow else error_scale
    if overflow:
        result["status"] = "arithmetic_overflow"

    difference_scale = max(ref_scale, cand_scale) if overflow else error_scale
    ref_squares, cand_squares, dot_products, errors = [], [], [], []
    for start in range(0, a.numel(), chunk_size):
        x, y = a[start : start + chunk_size].double(), b[start : start + chunk_size].double()
        nx = x / ref_scale if ref_scale else torch.zeros_like(x)
        ny = y / cand_scale if cand_scale else torch.zeros_like(y)
        ref_squares.append(nx.square().sum().item())
        cand_squares.append(ny.square().sum().item())
        dot_products.append((nx * ny).sum().item())
        if difference_scale:
            scaled_difference = x / difference_scale - y / difference_scale if overflow else (x - y) / difference_scale
            errors.append(scaled_difference.square().sum().item())
    rr, cc, dot = math.fsum(ref_squares), math.fsum(cand_squares), math.fsum(dot_products)
    if rr and cc:
        result["cosine"] = max(-1.0, min(1.0, dot / math.sqrt(rr * cc)))
    if rr:
        if not difference_scale:
            result["relative_l2"] = 0.0
        else:
            error_mantissa, error_exponent = math.frexp(difference_scale)
            ref_mantissa, ref_exponent = math.frexp(ref_scale)
            factor = (error_mantissa / ref_mantissa) * math.sqrt(math.fsum(errors) / rr)
            mantissa, exponent = math.frexp(factor)
            try:
                relative = math.ldexp(mantissa, error_exponent - ref_exponent + exponent)
                if relative == 0:
                    result["status"] = "arithmetic_underflow"
                else:
                    result["relative_l2"] = relative
            except OverflowError:
                result["status"] = "arithmetic_overflow"
    return result
