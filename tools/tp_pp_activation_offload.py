# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimental saved-tensor CPU offload preserving dense tensor layouts."""

import torch


def saved_tensor_cpu_offload():
    """Keep transposed dense saved weights transposed after their CUDA restore.

    Unlike save_on_cpu(pin_memory=True), empty_like with preserve_format keeps
    non-overlapping dense strides. Non-dense views retain values but may still
    change layout; this candidate requires separate full-model qualification.
    """

    def pack(tensor):
        if tensor.device.type == "cpu":
            return tensor.device, tensor
        if tensor.layout != torch.strided:
            raise ValueError("Diagnostic activation offload requires strided tensors")
        packed = torch.empty_like(tensor, device="cpu", pin_memory=True, memory_format=torch.preserve_format)
        packed.copy_(tensor)
        return tensor.device, packed

    def unpack(saved):
        device, tensor = saved
        return tensor.to(device, non_blocking=tensor.is_pinned())

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
