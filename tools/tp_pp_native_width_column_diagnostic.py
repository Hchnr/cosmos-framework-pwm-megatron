# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Eager FP32 native-width column-linear diagnostic; not a production layer.

Every rank must traverse the same forward AND backward branches in the same
order. No collective can detect a peer that skips an autograd node altogether.
This reconstruction is expensive and is not a numerical acceptance result.
"""

import torch
import torch.distributed as dist
from torch.nn import functional as F


def gather_rows(shard, group):
    parts = [torch.empty_like(shard) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, shard.contiguous(), group=group)
    return torch.cat(parts, dim=0)


def gather_output_gradient(local_gradient, group):
    parts = [torch.empty_like(local_gradient) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, local_gradient.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def native_width_vjp(inputs, full_weight, local_gradient, group, *, need_input=True, need_weight=True):
    """Reconstruct complete dY and use full-width matmuls before selecting dW."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if (
        inputs.ndim != 2
        or full_weight.ndim != 2
        or local_gradient.ndim != 2
        or inputs.shape[0] != local_gradient.shape[0]
        or inputs.shape[1] != full_weight.shape[1]
        or full_weight.shape[0] != local_gradient.shape[1] * world
    ):
        raise ValueError("Expected 2-D input, full weight and equal column-output shards")
    if any(t.dtype != torch.float32 or t.device != inputs.device for t in (inputs, full_weight, local_gradient)):
        raise ValueError("All VJP operands must be FP32 on the same device")
    full_gradient = gather_output_gradient(local_gradient, group)
    input_gradient = torch.mm(full_gradient, full_weight) if need_input else None
    weight_gradient = None
    if need_weight:
        full_weight_gradient = torch.mm(full_gradient.t(), inputs)
        width = local_gradient.shape[1]
        weight_gradient = full_weight_gradient.narrow(0, rank * width, width).contiguous()
    return input_gradient, weight_gradient


class _NativeWidthLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, local_weight, group):
        ctx.set_materialize_grads(False)
        if (
            inputs.ndim != 2
            or local_weight.ndim != 2
            or inputs.shape[1] != local_weight.shape[1]
            or inputs.dtype != torch.float32
            or local_weight.dtype != torch.float32
            or inputs.device != local_weight.device
        ):
            raise ValueError("Expected FP32 2-D activations and equal column-sharded weights")
        if inputs.is_cuda and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("TF32 must be disabled")
        if torch.compiler.is_compiling():
            raise ValueError("This diagnostic supports eager execution only")
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if size < 2:
            raise ValueError("TP group must contain at least two ranks")
        full_weight = gather_rows(local_weight.detach(), group)
        ctx.save_for_backward(inputs, full_weight)
        ctx.group = group
        width = local_weight.shape[0]
        return F.linear(inputs, full_weight).narrow(1, rank * width, width)

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            # Correct only when every rank omits this node. Rank-divergent use
            # cannot be repaired by an autograd callback that is never invoked.
            return None, None, None
        inputs, full_weight = ctx.saved_tensors
        dx, dw = native_width_vjp(
            inputs,
            full_weight,
            gradient,
            ctx.group,
            need_input=ctx.needs_input_grad[0],
            need_weight=ctx.needs_input_grad[1],
        )
        return dx, dw, None


def native_width_column_linear(inputs, local_weight, group=None):
    """Return this rank's output, with native-width forward and VJP matmuls."""
    return _NativeWidthLinear.apply(inputs, local_weight, group)
