# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in CPU gradient bank for the native FP32 full-capacity oracle.

Per-parameter accumulation, norms, clipping and optimizer arithmetic remain on
the parameter device. This candidate requires exact GPU qualification against
the original native oracle before its reference values can be trusted.
"""

import torch


class StreamedReferenceGradients:
    def __init__(self, parameters):
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        if not self.parameters or len({id(p) for p in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("Expected unique trainable leaf parameters")
        if any(not p.is_leaf or p.dtype != torch.float32 for p in self.parameters):
            raise ValueError("Streamed reference requires FP32 leaf parameters")
        self.bank = {}
        self.handles = [
            p.register_post_accumulate_grad_hook(self._offload) for p in self.parameters
        ]

    @torch.no_grad()
    def _offload(self, parameter):
        grad = parameter.grad
        if grad is None or grad.layout != torch.strided or grad.dtype != torch.float32:
            raise ValueError("Only dense FP32 parameter gradients are supported")
        if id(parameter) in self.bank:
            # Preserve native leaf-accumulation order on the original device.
            previous = self.bank[id(parameter)].to(parameter.device, copy=True)
            previous.add_(grad)
            grad = previous
        self.bank[id(parameter)] = grad.detach().to(
            "cpu", copy=True, memory_format=torch.preserve_format
        )
        parameter.grad = None

    def reset(self):
        if any(p.grad is not None for p in self.parameters):
            raise ValueError(
                "A device gradient is unexpectedly live at the update boundary"
            )
        self.bank.clear()

    def raw(self, parameter):
        return self.bank.get(id(parameter))

    @torch.no_grad()
    def norm(self, parameters):
        norms = []
        for parameter in parameters:
            host = self.raw(parameter)
            if host is not None:
                grad = host.to(parameter.device, copy=True)
                norms.append(torch._foreach_norm([grad], 2)[0])
                del grad
        if not norms:
            raise ValueError("Native oracle requires at least one used gradient")
        total = torch.linalg.vector_norm(torch.stack(norms), 2)
        if not torch.isfinite(total):
            raise RuntimeError(
                "The total norm of streamed reference gradients is non-finite"
            )
        return total

    @torch.no_grad()
    def clip_and_step(self, optimizer, max_norm, norm, step_gpu_leaves):
        ordered = [p for group in optimizer.param_groups for p in group["params"]]
        if len(ordered) != len(self.parameters) or {id(p) for p in ordered} != {
            id(p) for p in self.parameters
        }:
            raise ValueError(
                "Optimizer parameter coverage differs from registered leaves"
            )
        if len({id(p) for p in ordered}) != len(ordered) or any(
            p.grad is not None for p in ordered
        ):
            raise ValueError(
                "Optimizer leaves must be unique and have no live device gradients"
            )
        for parameter in ordered:
            host = self.raw(parameter)
            if host is None:
                continue
            try:
                parameter.grad = host.to(parameter.device, copy=True)
                torch.nn.utils.clip_grads_with_norm_([parameter], max_norm, norm)
                step_gpu_leaves(optimizer)
            finally:
                parameter.grad = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
