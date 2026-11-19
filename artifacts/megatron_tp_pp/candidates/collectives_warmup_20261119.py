# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Initialize optimizer gradient-statistics collectives before activation peaks."""

import time

import torch
import torch.distributed as dist


def warm_optimizer_collectives(optimizer):
    """Warm the actual MCore groups with the FP32 shape used by gradient norm.

    Only independent zero scalars are reduced. Parameters, gradients, optimizer
    states, and random generators are not modified. No CUDA cache is cleared.
    """
    if not dist.is_initialized():
        raise RuntimeError("Optimizer collective warmup requires initialized process groups")
    groups = []
    for wrapper in getattr(optimizer, "chained_optimizers", [optimizer]):
        group = wrapper.get_grad_stats_parallel_group()
        group = dist.group.WORLD if group is None else group
        if group is dist.GroupMember.NON_GROUP_MEMBER or dist.get_world_size(group) < 1:
            raise RuntimeError("Optimizer gradient-statistics group excludes this rank")
        if not any(group is existing for existing in groups):
            groups.append(group)
    reports = []
    for group in groups:
        free_before, _ = torch.cuda.mem_get_info()
        started = time.monotonic()
        value = torch.zeros(1, dtype=torch.float32, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
        torch.cuda.synchronize()
        if value.item() != 0:
            raise RuntimeError("Optimizer collective warmup produced an invalid zero reduction")
        free_after, _ = torch.cuda.mem_get_info()
        reports.append(
            {
                "group_size": dist.get_world_size(group),
                "group_rank": dist.get_rank(group),
                "backend": str(dist.get_backend(group)),
                "dtype": "float32",
                "shape": [1],
                "operations": ["sum", "max"],
                "host_seconds": time.monotonic() - started,
                "free_cuda_bytes_before": free_before,
                "free_cuda_bytes_after": free_after,
                "allocated_cuda_bytes_after": torch.cuda.memory_allocated(),
                "reserved_cuda_bytes_after": torch.cuda.memory_reserved(),
            }
        )
    return reports
