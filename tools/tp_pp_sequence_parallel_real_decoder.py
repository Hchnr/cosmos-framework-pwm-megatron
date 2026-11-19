# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Isolated sequence parallel for the actual small-action decoder layer.

Not a training integration: CPU/NCCL eager qualification only. The native
small_action_tower attention/rotation/normalization and Megatron local GEMM are
used; residuals are sequence-sharded. Input and post norms are sequence-partial;
all trainable replicated grads must be TP-summed after backward.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cosmos_framework.model.generator.megatron_tp_pp.compile import NonePreservingLinear
from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear
from cosmos_framework.model.generator.mot.small_action_tower import attend, rotate

TRACE = []


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group, ctx.world = group, dist.get_world_size(group)
        TRACE.append(("gather", "forward", tuple(value.shape)))
        if not value.numel():
            return value.clone()
        result = value.new_empty((value.shape[0] * ctx.world, value.shape[1]))
        dist.all_gather_into_tensor(result, value.contiguous(), group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        TRACE.append(("gather", "backward_reduce_scatter", tuple(gradient.shape)))
        if not gradient.numel():
            return gradient.clone(), None
        result = gradient.new_empty((gradient.shape[0] // ctx.world, gradient.shape[1]))
        dist.reduce_scatter_tensor(result, gradient.contiguous(), group=ctx.group)
        return result, None


class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group, ctx.world = group, dist.get_world_size(group)
        if value.shape[0] % ctx.world:
            raise ValueError("Physical sequence must divide TP")
        TRACE.append(("reduce_scatter", "forward", tuple(value.shape)))
        if not value.numel():
            return value.clone()
        result = value.new_empty((value.shape[0] // ctx.world, value.shape[1]))
        dist.reduce_scatter_tensor(result, value.contiguous(), group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        TRACE.append(("reduce_scatter", "backward_gather", tuple(gradient.shape)))
        if not gradient.numel():
            return gradient.clone(), None
        result = gradient.new_empty((gradient.shape[0] * ctx.world, gradient.shape[1]))
        dist.all_gather_into_tensor(result, gradient.contiguous(), group=ctx.group)
        return result, None


def gather(value, group):
    return Gather.apply(value, group)


def reduce_scatter(value, group):
    return ReduceScatter.apply(value, group)


def local_linear(module, value, group, *, row=False):
    if not isinstance(module, TensorParallelLinear):
        raise TypeError("All production projections must use the MCore TP adapter")
    if (module.axis == 1) != row:
        raise ValueError("Unexpected projection sharding axis")
    # The SP gather's adjoint performs the SUM reduce-scatter for input grads;
    # enabling the original column dgrad all-reduce would count it twice.
    output = NonePreservingLinear.apply(
        value, module.weight, None if row else module.bias,
        False, False, False, None, None, group, 1, None,
    )
    if row:
        output = reduce_scatter(output, group)
        if module.bias is not None:
            output = output + module.bias
    return output


def qkv(module, x, group, suffix=""):
    full = gather(x, group)
    return tuple(local_linear(getattr(module, name + suffix), full, group) for name in ("q_proj", "k_proj", "v_proj"))


def mlp(module, local, group):
    full = gather(local, group)
    gate = local_linear(module.gate_proj, full, group)
    up = local_linear(module.up_proj, full, group)
    activation = getattr(module, "act_fn", F.silu)
    return local_linear(module.down_proj, activation(gate) * up, group, row=True)


def forward(layer, pack, group):
    attn = layer.self_attn
    u, v, a = pack["causal_seq"], pack["full_only_seq"], pack["action_seq"]
    block = getattr(layer, "action_block_moe_gen_action", None)
    active = block is not None
    qs, ks, vs = [], [], []
    for suffix, x in (("", layer.input_layernorm(u)), ("_moe_gen", layer.input_layernorm_moe_gen(v))):
        q, k, value = qkv(attn, x, group, suffix)
        qs.append(getattr(attn, "q_norm" + suffix)(q.view(-1, attn.num_attention_heads, attn.head_dim)))
        ks.append(getattr(attn, "k_norm" + suffix)(k.view(-1, attn.num_key_value_heads, attn.head_dim)))
        vs.append(value.view(-1, attn.num_key_value_heads, attn.head_dim))
    if active:
        an = block.input_layernorm(a)
        s = block.shape
        q, k, value = qkv(block, an, group)
        qs.append(block.q_norm(q.view(-1, s.q_heads, s.head_dim)))
        ks.append(block.k_norm(k.view(-1, s.kv_heads, s.head_dim)))
        vs.append(value.view(-1, s.kv_heads, s.head_dim))
    rope, masks = pack["_small_rope"], pack["_small_masks"]
    yu = attend(rotate(qs[0], rope[0]), rotate(ks[0], rope[0]), vs[0], masks[(False, 0)], frozen_und=True)
    vk = [rotate(ks[i], rope[i]) for i in (0, 1)]
    vv = vs[:2]
    if active:
        vk.append(rotate(block.adapt(ks[2], "action_to_video", "k"), rope[2]))
        vv.append(block.adapt(vs[2], "action_to_video", "v"))
    yv = attend(rotate(qs[1], rope[1]), torch.cat(vk), torch.cat(vv), masks[(active, 1)])
    if active:
        ar = pack["_small_action_rope"]
        ak = [rotate(block.adapt(ks[i], direction, "k"), ar[i]) for i, direction in enumerate(("und_to_action", "video_to_action"))]
        av = [block.adapt(vs[i], direction, "v") for i, direction in enumerate(("und_to_action", "video_to_action"))]
        ak.append(rotate(ks[2], ar[2]))
        av.append(vs[2])
        ya = attend(rotate(qs[2], ar[2]), torch.cat(ak), torch.cat(av), masks[(True, 2)])
        a = a + local_linear(block.o_proj, ya, group, row=True)
        a = a + mlp(block, block.post_attention_layernorm(a), group)
    u = u + local_linear(attn.o_proj, yu, group, row=True)
    u = u + mlp(layer.mlp, layer.post_attention_layernorm(u), group)
    v = v + local_linear(attn.o_proj_moe_gen, yv, group, row=True)
    v = v + mlp(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(v), group)
    out = dict(pack)
    out.update(causal_seq=u, full_only_seq=v, action_seq=a)
    return out, {}, None


class SPBlock(torch.nn.Module):
    def __init__(self, layer, group):
        super().__init__()
        self.layer = layer
        self.__dict__["group"] = group

    def forward(self, pack):
        return forward(self.layer, pack, self.group)[0]
