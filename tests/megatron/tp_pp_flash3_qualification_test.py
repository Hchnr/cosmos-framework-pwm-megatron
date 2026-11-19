# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import torch

from tools.qualify_tp_pp_flash3_selective_ac import compare_tensors, forward_kernel_count


def test_profiler_forward_counter_requires_cuda_and_exact_kernel_family():
    events = [
        SimpleNamespace(device_type=SimpleNamespace(name="CUDA"), name="FlashAttnFwdSm90<bf16>"),
        SimpleNamespace(device_type=SimpleNamespace(name="CPU"), name="FlashAttnFwdSm90"),
        SimpleNamespace(device_type=SimpleNamespace(name="CUDA"), name="FlashAttnBwdSm90"),
    ]
    assert forward_kernel_count(events) == 1


def test_comparison_checks_shapes_finiteness_and_bf16_tolerance():
    value = torch.tensor([0.2, -0.5], dtype=torch.bfloat16)
    assert compare_tensors([value], [value.clone()])["passed"]
    assert not compare_tensors([value], [torch.tensor([0.2], dtype=torch.bfloat16)])["passed"]
    assert not compare_tensors([value], [torch.tensor([float("nan"), -0.5], dtype=torch.bfloat16)])["passed"]
    assert not compare_tensors([value], [torch.tensor([0.5, -0.5], dtype=torch.bfloat16)])["passed"]
