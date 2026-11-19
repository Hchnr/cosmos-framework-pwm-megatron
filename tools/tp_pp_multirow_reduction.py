# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimental grouped SUM for independent TP row-parallel outputs.

Every rank must enter the same group, dtype and per-output shapes in the same
order. This prototype is isolated from training and makes no performance or
full-model correctness claim.
"""

import torch
import torch.nn.functional as F


class GroupedRowSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group_name, *values):
        if not values:
            raise ValueError("At least one output is required")
        first = values[0]
        if any(value.device != first.device or value.dtype != first.dtype for value in values):
            raise ValueError("Every output must use the same device and dtype")
        shapes = [tuple(value.shape) for value in values]
        lengths = [value.numel() for value in values]
        ctx.set_materialize_grads(False)
        if sum(lengths):
            packed = torch.cat([value.contiguous().view(-1) for value in values])
            reduced = torch.ops._c10d_functional.all_reduce(packed, "sum", group_name)
            reduced = torch.ops._c10d_functional.wait_tensor(reduced)
            result = tuple(piece.reshape(shape) for piece, shape in zip(reduced.split(lengths), shapes, strict=True))
        else:
            # Empty tensors still participate in autograd but need no payload
            # collective. All ranks must agree on lengths.
            result = tuple(value.clone() for value in values)
        frozen = [value for index, value in enumerate(result) if not ctx.needs_input_grad[index + 1]]
        if frozen:
            ctx.mark_non_differentiable(*frozen)
        return result

    @staticmethod
    def backward(ctx, *gradients):
        return (None, *gradients)


def grouped_row_sum(values, group):
    return GroupedRowSum.apply(group.group_name, *values)


def grouped_row_linears(inputs, weights, biases, group):
    """Use independent local GEMMs, one grouped SUM, and replicated biases.

    This uses F.linear for the isolated mathematical test. Production must
    retain MCore's local linear and register unchanged parameters.
    """
    if not (len(inputs) == len(weights) == len(biases)):
        raise ValueError("Input, weight and bias counts must match")
    partials = [F.linear(value, weight) for value, weight in zip(inputs, weights, strict=True)]
    reduced = grouped_row_sum(partials, group)
    return tuple(value if bias is None else value + bias for value, bias in zip(reduced, biases, strict=True))
