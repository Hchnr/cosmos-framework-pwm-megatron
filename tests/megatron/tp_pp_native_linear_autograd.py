# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Diagnostic native local Linear autograd under the existing TP communication.

This isolates MCore's local GEMM/autograd from the partitioning, pipeline,
distributed optimizer, and loss scaling. It never changes production defaults.
"""

import torch.nn.functional as functional


class NativeLinearAutograd:
    @staticmethod
    def apply(
        inputs,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        gtp_remat_size,
        output_dtype,
    ):
        if (
            gradient_accumulation_fusion
            or sequence_parallel
            or grad_output_buffer is not None
            or wgrad_deferral_limit is not None
            or gtp_remat_size != 1
            or output_dtype not in (None, inputs.dtype)
        ):
            raise ValueError("Native local-linear isolation supports only the existing eager diagnostic contract")
        if allreduce_dgrad:
            from cosmos_framework.model.generator.megatron_tp_pp.compile import CopyToTensorParallel

            inputs = CopyToTensorParallel.apply(inputs, tp_group.group_name)
        return functional.linear(inputs, weight, bias)
