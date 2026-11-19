# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimental SP boundaries, not installed into production training.

Every rank must supply an identical packed input/loss and collective order.
Shard streams by token, preserve dummy context, and gather only for replicated
output heads. Adapter gradient synchronization is explicit and preserves None.
"""

import torch
import torch.distributed as dist


class ReplicatedOutputGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, group):
        ctx.rank, ctx.world, ctx.group = dist.get_rank(group), dist.get_world_size(group), group
        ctx.rows = shard.shape[0]
        ctx.set_materialize_grads(False)
        if shard.numel() == 0:
            return shard.clone()
        result = shard.new_empty((shard.shape[0] * ctx.world, *shard.shape[1:]))
        dist.all_gather_into_tensor(result, shard.contiguous(), group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None
        return grad.narrow(0, ctx.rank * ctx.rows, ctx.rows).contiguous(), None


class ReplicatedLossSumGather(torch.autograd.Function):
    """Deliberately wrong output boundary when loss is divided and heads replicate."""

    @staticmethod
    def forward(ctx, shard, group):
        ctx.world, ctx.group, ctx.rows = dist.get_world_size(group), group, shard.shape[0]
        ctx.set_materialize_grads(False)
        result = shard.new_empty((ctx.rows * ctx.world, *shard.shape[1:]))
        dist.all_gather_into_tensor(result, shard.contiguous(), group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None
        local = grad.new_empty((ctx.rows, *grad.shape[1:]))
        dist.reduce_scatter_tensor(local, grad.contiguous(), group=ctx.group)
        return local, None


class DivideBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, divisor):
        ctx.divisor = divisor
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad):
        return grad / ctx.divisor if grad is not None else None, None


def sharded_input(pack, group, *, scale_dummy=True):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    out = dict(pack)
    for name in ("causal_seq", "full_only_seq", "action_seq"):
        value = pack[name]
        if value.shape[0] % world:
            raise ValueError("Physical stream length must be divisible by TP")
        length = value.shape[0] // world
        out[name] = value.narrow(0, rank * length, length).contiguous()
    if scale_dummy:
        out["pipeline_context"] = DivideBackward.apply(pack["pipeline_context"], world)
    return out


def replicated_output(pack, group, *, incorrect_sum_backward=False):
    out = dict(pack)
    op = ReplicatedLossSumGather if incorrect_sum_backward else ReplicatedOutputGather
    for name in ("causal_seq", "full_only_seq", "action_seq"):
        out[name] = op.apply(pack[name], group)
    return out


def reduce_partial_input_gradients(module, group):
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        present = torch.tensor([parameter.grad is not None], dtype=torch.int64, device=parameter.device)
        gathered = [torch.empty_like(present) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, present, group=group)
        flags = [int(value.item()) for value in gathered]
        if len(set(flags)) != 1:
            raise ValueError(f"Inconsistent rank gradient presence: {name} {flags}")
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
