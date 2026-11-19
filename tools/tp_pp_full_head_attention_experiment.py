# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in full-head FP32 SDPA reconstruction, not a production backend.

All TP ranks must enter each forward and backward branch in identical order.
Reconstructing full q/k/v and cotangents is deliberately costly. The local
q/k/v VJP is sliced from the native full-head result and must NOT be TP-SUMed.
"""

from contextvars import ContextVar

import torch
import torch.distributed as dist


def gather_heads(value, group):
    parts = [torch.empty_like(value, memory_format=torch.contiguous_format) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, value.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def validate_collectively(q, k, v, mask, group):
    size = dist.get_world_size(group)
    tensors = (q, k, v, mask)
    metadata = [
        (tuple(x.shape), str(x.dtype), x.device.type, bool(x.requires_grad)) if isinstance(x, torch.Tensor) else None
        for x in tensors
    ]
    all_metadata = [None] * size
    dist.all_gather_object(all_metadata, metadata, group=group)
    if any(row != all_metadata[0] for row in all_metadata):
        raise ValueError("All TP ranks require identical attention shapes, dtypes and branch flags")
    if any(item is None for item in metadata):
        raise ValueError("Expected tensor q/k/v and dense bool mask")
    if (
        any(x.dtype != torch.float32 for x in (q, k, v))
        or mask.dtype != torch.bool
        or any(x.device != q.device for x in (k, v, mask))
        or any(x.ndim != 3 for x in (q, k, v))
        or mask.ndim != 2
        or k.shape != v.shape
        or q.shape[2] != k.shape[2]
        or q.shape[1] % k.shape[1]
        or tuple(mask.shape) != (q.shape[0], k.shape[0])
    ):
        raise ValueError("Unsupported FP32 dense GQA attention layout")
    if q.shape[0] and not k.shape[0]:
        raise ValueError("Nonempty queries require keys")
    if mask.numel():
        reference = mask.contiguous().clone()
        source = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast(reference, src=source, group=group)
        matching = torch.tensor(int(torch.equal(reference, mask)), dtype=torch.int32, device=q.device)
        dist.all_reduce(matching, op=dist.ReduceOp.MIN, group=group)
        if not matching.item():
            raise ValueError("Attention masks differ across TP ranks")


class _FullHeadAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, group, original, frozen_und, audit):
        validate_collectively(q, k, v, mask, group)
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        if size < 2:
            raise ValueError("Full-head diagnostic requires TP > 1")
        if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("TF32 must be disabled")
        ctx.rank, ctx.group, ctx.original, ctx.frozen_und = (
            rank,
            group,
            original,
            frozen_und,
        )
        ctx.local_heads = (q.shape[1], k.shape[1], v.shape[1])
        ctx.audit = audit
        ctx.empty_query = q.shape[0] == 0
        if ctx.empty_query:
            ctx.save_for_backward(q, k, v, mask)
            return original(q, k, v, mask, backend="sdpa", frozen_und=frozen_und)
        full_q, full_k, full_v = (gather_heads(tensor, group) for tensor in (q, k, v))
        output = original(full_q, full_k, full_v, mask, backend="sdpa", frozen_und=frozen_und)
        width = q.shape[1] * q.shape[2]
        if output.ndim != 2 or output.shape != (q.shape[0], width * size):
            raise ValueError("Unexpected full-head attention output")
        ctx.save_for_backward(full_q, full_k, full_v, mask)
        return output.narrow(1, rank * width, width).contiguous()

    @staticmethod
    def backward(ctx, dy):
        if dy is None:
            return (None,) * 8
        if ctx.audit is not None:
            ctx.audit["backward_calls"] += 1
        full_q, full_k, full_v, mask = ctx.saved_tensors
        if ctx.empty_query:
            with torch.enable_grad():
                inputs = [value.detach().requires_grad_() for value in (full_q, full_k, full_v)]
                output = ctx.original(*inputs, mask, backend="sdpa", frozen_und=ctx.frozen_und)
                gradients = torch.autograd.grad(output, inputs, grad_outputs=dy)
            return (
                *(grad if needed else None for grad, needed in zip(gradients, ctx.needs_input_grad[:3], strict=True)),
                None,
                None,
                None,
                None,
                None,
            )
        parts = [
            torch.empty_like(dy, memory_format=torch.contiguous_format) for _ in range(dist.get_world_size(ctx.group))
        ]
        dist.all_gather(parts, dy.contiguous(), group=ctx.group)
        full_dy = torch.cat(parts, dim=1)
        with torch.enable_grad():
            inputs = [value.detach().requires_grad_() for value in (full_q, full_k, full_v)]
            output = ctx.original(*inputs, mask, backend="sdpa", frozen_und=ctx.frozen_und)
            gradients = torch.autograd.grad(output, inputs, grad_outputs=full_dy)
        local = [
            gradient.narrow(1, ctx.rank * count, count).contiguous() if needed else None
            for gradient, count, needed in zip(gradients, ctx.local_heads, ctx.needs_input_grad[:3], strict=True)
        ]
        # Each selected KV head's full native VJP already combines its GQA
        # query groups. No TP SUM is permitted here.
        return *local, None, None, None, None, None


def full_head_attention(q, k, v, mask, group, original, *, frozen_und=False, audit=None):
    return _FullHeadAttention.apply(q, k, v, mask, group, original, frozen_und, audit)


def install(audit=None):
    from megatron.core import parallel_state as ps

    from cosmos_framework.model.generator.megatron_tp_pp.layers import (
        TensorParallelLinear,
    )
    from cosmos_framework.model.generator.mot import small_action_tower

    original = small_action_tower.attend
    original_layer = small_action_tower.small_layer_forward
    in_tp_layer = ContextVar("full_head_attention_tp_layer", default=False)
    if getattr(original, "_full_head_attention_diagnostic", False):
        raise RuntimeError("Full-head attention is already installed")

    def attend(q, k, v, mask, backend="sdpa", *, frozen_und=False):
        if not in_tp_layer.get():
            return original(q, k, v, mask, backend=backend, frozen_und=frozen_und)
        if backend != "sdpa" or torch.compiler.is_compiling():
            raise ValueError("Full-head diagnostic supports eager dense SDPA only")
        if audit is not None:
            audit["forward_calls"] += 1
        return full_head_attention(
            q,
            k,
            v,
            mask,
            ps.get_tensor_model_parallel_group(),
            original,
            frozen_und=frozen_und,
            audit=audit,
        )

    def small_layer_forward(layer, *args, **kwargs):
        token = in_tp_layer.set(isinstance(layer.self_attn.q_proj, TensorParallelLinear))
        try:
            return original_layer(layer, *args, **kwargs)
        finally:
            in_tp_layer.reset(token)

    attend._full_head_attention_diagnostic = True
    small_action_tower.attend = attend
    small_action_tower.small_layer_forward = small_layer_forward
