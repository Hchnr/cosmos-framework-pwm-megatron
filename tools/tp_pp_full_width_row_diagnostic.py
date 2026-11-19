# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Isolated FP32 row-parallel full-width forward/VJP diagnostic.

All ranks must enter identical forward/backward branches in the same order.
Upstream dY must be identical on every rank as expected after a row reduction.
Not a production implementation; full tensors are intentionally reconstructed.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def gather_last(local, group):
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=-1)


class _FullRow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group):
        ctx.set_materialize_grads(False)
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError("Expected 2-D equal-width input and row-sharded weight")
        if x.dtype != torch.float32 or weight.dtype != torch.float32 or x.device != weight.device:
            raise ValueError("Expected same-device FP32 tensors")
        if x.is_cuda and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("Disable TF32 for this diagnostic")
        if torch.compiler.is_compiling():
            raise ValueError("Eager diagnostic only")
        if dist.get_world_size(group) < 2:
            raise ValueError("At least two TP ranks are required")
        full_x = gather_last(x, group)
        full_w = gather_last(weight.detach(), group)
        ctx.save_for_backward(full_x, full_w)
        ctx.group, ctx.width = group, x.shape[1]
        return F.linear(full_x, full_w)

    @staticmethod
    def backward(ctx, dy):
        if dy is None:
            return None, None, None
        full_x, full_w = ctx.saved_tensors
        group, width = ctx.group, ctx.width
        copies = [torch.empty_like(dy) for _ in range(dist.get_world_size(group))]
        dist.all_gather(copies, dy.contiguous(), group=group)
        if not all(torch.equal(copies[0], part) for part in copies[1:]):
            raise RuntimeError("Row output gradient differs across TP ranks")
        offset = dist.get_rank(group) * width
        dx = torch.mm(dy, full_w).narrow(1, offset, width).contiguous() if ctx.needs_input_grad[0] else None
        dw = torch.mm(dy.t(), full_x).narrow(1, offset, width).contiguous() if ctx.needs_input_grad[1] else None
        return dx, dw, None


def full_row(x, weight, group=None):
    return _FullRow.apply(x, weight, group)
