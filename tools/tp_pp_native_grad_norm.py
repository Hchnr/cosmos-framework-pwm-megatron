# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Candidate native per-parameter FP32 norm arithmetic for DP1 TP/PP.

The order argument contains parameter names only. No reference gradient/norm,
parameter value, optimizer state, or teacher-forced state is read here.
"""

import types

import torch
import torch.distributed as dist


@torch.no_grad()
def native_gradient_norm(driver, order):
    from megatron.core import parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import stage_parameter_manifest

    if driver.topology.data_parallel_size(dist.get_world_size()) != 1:
        raise ValueError("The native norm candidate currently requires physical DP1")
    if driver.stage.config.params_dtype != torch.float32:
        raise ValueError("The native norm candidate currently requires FP32 compute/gradients")
    driver._native_norm_call_count += 1
    indices = {name: index for index, name in enumerate(order)}
    if len(indices) != len(order):
        raise ValueError("Duplicate native norm parameter identity")
    manifest = stage_parameter_manifest(driver.stage)
    tp_group = ps.get_tensor_model_parallel_group()
    owner = ps.get_tensor_model_parallel_rank() == 0
    norms = torch.zeros(len(order), dtype=torch.float32, device="cuda")
    used = torch.zeros(len(order), dtype=torch.int32, device="cuda")
    for name, parameter in driver.stage.named_parameters():
        if not parameter.requires_grad:
            continue
        spec = manifest[name]
        index = indices[spec.canonical_name]
        gradient = parameter.main_grad.detach()
        if gradient.dtype != torch.float32:
            raise ValueError("Native norm requires FP32 gradients")
        if spec.axis is not None and ps.get_tensor_model_parallel_world_size() > 1:
            pieces = [torch.empty_like(gradient) for _ in range(ps.get_tensor_model_parallel_world_size())]
            dist.all_gather(pieces, gradient.contiguous(), group=tp_group)
            gradient = torch.cat(pieces, dim=spec.axis)
        if owner:
            norms[index] = torch._foreach_norm([gradient], 2)[0]
        used[index] = int(parameter in driver._globally_used)
    group = ps.get_model_parallel_group()
    dist.all_reduce(norms, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(used, op=dist.ReduceOp.MAX, group=group)
    # Exactly the native get_total_norm decomposition: per-parameter FP32
    # foreach norms, then an FP32 vector norm in registration order.
    return torch.linalg.vector_norm(norms[used.bool()], 2).item()


def install_native_gradient_norm(order):
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    original = ModelParallelDriver.__init__
    original_step = ModelParallelDriver.step

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._native_norm_call_count = 0
        self.optimizer.get_grad_norm = types.MethodType(
            lambda optimizer: native_gradient_norm(self, order), self.optimizer
        )

    def step(self, *args, **kwargs):
        before = self._native_norm_call_count
        result = original_step(self, *args, **kwargs)
        if self._native_norm_call_count <= before:
            raise RuntimeError("MCore optimizer did not consume the candidate native norm")
        return result

    ModelParallelDriver.__init__ = initialize
    ModelParallelDriver.step = step
