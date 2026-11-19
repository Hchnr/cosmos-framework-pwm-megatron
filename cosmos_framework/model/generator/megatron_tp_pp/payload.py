# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Differentiable, unequal-width pipeline boundary; no hidden-width padding."""

import math
from dataclasses import dataclass

import torch

STREAMS = ("causal_seq", "full_only_seq", "action_seq")


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shapes, *values):
        ctx.shapes = shapes
        data = [value.reshape(-1, 1, 1) for value in values]
        # Backward-only structural usage flags preserve grad=None across a
        # single-tensor P2P boundary. Ordinary cat/split materializes zeros for
        # unused streams, causing Adam to update disconnected parameters.
        return torch.cat([*data, values[0].new_zeros(len(values), 1, 1)], dim=0)

    @staticmethod
    def backward(ctx, gradient):
        counts = [math.prod(shape) for shape in ctx.shapes]
        parts = gradient.view(-1).split([*counts, len(counts)])
        active = parts[-1].bool().cpu().tolist()
        return (
            None,
            *(
                part.view(shape) if used else None
                for part, shape, used in zip(parts[:-1], ctx.shapes, active, strict=True)
            ),
        )


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, shapes):
        ctx.set_materialize_grads(False)
        ctx.shapes, ctx.device, ctx.dtype = shapes, tensor.device, tensor.dtype
        counts = [math.prod(shape) for shape in shapes]
        parts = tensor.view(-1).split([*counts, len(counts)])
        return tuple(part.view(shape) for part, shape in zip(parts[:-1], shapes, strict=True))

    @staticmethod
    def backward(ctx, *gradients):
        parts = [
            gradient.reshape(-1, 1, 1)
            if gradient is not None
            else torch.zeros(math.prod(shape), 1, 1, device=ctx.device, dtype=ctx.dtype)
            for gradient, shape in zip(gradients, ctx.shapes, strict=True)
        ]
        flags = torch.tensor([gradient is not None for gradient in gradients], device=ctx.device, dtype=ctx.dtype)
        return torch.cat([*parts, flags.view(-1, 1, 1)], dim=0), None


@dataclass(frozen=True)
class PayloadLayout:
    shapes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    context_width: int = 0

    def __post_init__(self):
        if len(self.shapes) != 3 or any(len(s) != 2 or s[0] < 0 or s[1] < 1 for s in self.shapes):
            raise ValueError("Payload requires three (length, width) shapes")
        if self.context_width < 0:
            raise ValueError("Context width must be nonnegative")

    @property
    def keys(self):
        return STREAMS + (("pipeline_context",) if self.context_width else ())

    @property
    def transport_shapes(self):
        return self.shapes + (((1, self.context_width),) if self.context_width else ())

    @property
    def numel(self):
        return sum(math.prod(s) for s in self.transport_shapes) + len(self.keys)

    @property
    def tensor_shape(self):
        # Standard MCore P2P supports three dimensions, including dynamic shapes.
        return (self.numel, 1, 1)

    def pack(self, streams):
        values = [streams[key] for key in self.keys]
        if any(tuple(v.shape) != s for v, s in zip(values, self.transport_shapes, strict=True)):
            raise ValueError("Activation shapes do not match the pipeline payload layout")
        # Concatenate in transport shape so the result owns storage. MCore's
        # pseudo-deallocation rejects views of a separate flat cat allocation.
        return _Pack.apply(self.transport_shapes, *values)

    def unpack(self, tensor, *, frozen_und=False):
        if tuple(tensor.shape) != self.tensor_shape:
            raise ValueError(f"Unexpected pipeline shape: {tensor.shape}, expected {self.tensor_shape}")
        pieces = _Unpack.apply(tensor, self.transport_shapes)
        result = dict(zip(self.keys, pieces, strict=True))
        if frozen_und:
            result["causal_seq"] = result["causal_seq"].detach()
        return result
