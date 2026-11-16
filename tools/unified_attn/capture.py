# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Diagnostic-only, compile-safe capture of real attention inputs and dO.

Explicit clone identities keep file I/O outside compiler tracing. Never enabled
in a performance/profile run. The first microbatch is captured at layers 0/17/35.
"""

import os
from pathlib import Path

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)

_seen = set()


def _save(tensors, layer, stream, tag):
    key = (layer, stream, tag)
    if key in _seen:
        return
    _seen.add(key)
    root = Path(os.environ["UNIFIED_ATTN_RUN"]) / "activations" / f"rank{os.environ['RANK']}"
    root.mkdir(parents=True, exist_ok=True)
    torch.save([t.detach().cpu() for t in tensors], root / f"layer{layer}_{stream}_{tag}.pt")


@torch.library.custom_op("cosmos_audit::capture", mutates_args=())
def capture(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: int, stream: int, tag: int) -> torch.Tensor:
    _save((q, k, v) if tag == 0 else (q,), layer, stream, "qkv" if tag == 0 else "out")
    return q.contiguous().clone()


@capture.register_fake
def _fake(q, k, v, layer, stream, tag):
    return q.new_empty(q.shape)


@torch.library.custom_op("cosmos_audit::capture_grad", mutates_args=())
def capture_grad(grad: torch.Tensor, layer: int, stream: int, tag: int) -> torch.Tensor:
    _save((grad,), layer, stream, "dq" if tag == 0 else "dout")
    return grad.contiguous().clone()


@capture_grad.register_fake
def _fake_grad(grad, layer, stream, tag):
    return grad.new_empty(grad.shape)


def _setup(ctx, inputs, output):
    ctx.location = inputs[3:]


def _backward(ctx, grad):
    return capture_grad(grad, *ctx.location), None, None, None, None, None


capture.register_autograd(_backward, setup_context=_setup)


def install(model):
    def wrap(original, index):
        def dispatch(q, k, v, *args, **kwargs):
            qu = capture(get_und_seq(q), get_und_seq(k), get_und_seq(v), index, 0, 0)
            qg = capture(get_gen_seq(q), get_gen_seq(k), get_gen_seq(v), index, 1, 0)
            result, kv = original(from_und_gen_splits(qu, qg, q), k, v, *args, **kwargs)
            yu, yg = get_und_seq(result), get_gen_seq(result)
            yu = capture(yu, yu[:0], yu[:0], index, 0, 1)
            yg = capture(yg, yg[:0], yg[:0], index, 1, 1)
            return from_und_gen_splits(yu, yg, result), kv

        return dispatch

    for index in (0, 17, 35):
        attn = model.net.language_model.model.layers[index].self_attn
        attn.dispatch_attention_fn = wrap(attn.dispatch_attention_fn, index)
