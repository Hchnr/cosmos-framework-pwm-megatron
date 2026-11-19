# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in eager FP32 diagnostic: full-width K/V GEMM with local TP outputs.

Every TP rank gathers detached weight shards, reattaches only its own original
Parameter, executes the native-width GEMM, and keeps its output shard. The
input-gradient identity performs the same TP SUM as a column-parallel linear.
This is expensive and is neither a production optimization nor a proven fix.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F


class InputGradientSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, group):
        ctx.group = group
        ctx.set_materialize_grads(False)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return None, None
        result = gradient.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=ctx.group)
        return result, None


def reassemble(gathered, local, rank, size):
    if size < 2 or not 0 <= rank < size or gathered.shape[0] != size * local.shape[0]:
        raise ValueError("Inconsistent TP gather shape or rank")
    if tuple(gathered.shape[1:]) != tuple(local.shape[1:]):
        raise ValueError("Gathered and local parameter dimensions differ")
    pieces = list(gathered.detach().split(local.shape[0], dim=0))
    pieces[rank] = local
    return torch.cat(pieces, dim=0)


def gather_local_parameter(local, group, rank, size):
    gathered = torch.empty((size * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(gathered, local.detach().contiguous(), group=group)
    return reassemble(gathered, local, rank, size)


def reconstructed_linear(inputs, weight, bias, group):
    if torch.compiler.is_compiling():
        raise ValueError("Reconstructed K/V diagnostic requires eager execution")
    if (
        inputs.dtype != torch.float32
        or weight.dtype != torch.float32
        or (bias is not None and bias.dtype != torch.float32)
    ):
        raise ValueError("Reconstructed K/V diagnostic requires FP32 inputs and parameters")
    if inputs.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable TF32 for the FP32 reconstructed K/V diagnostic")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size < 2 or weight.ndim != 2 or inputs.shape[-1] != weight.shape[1]:
        raise ValueError("Expected a column-sharded rank-two weight and TP size > 1")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("Bias must have the same local output width as the weight")
    full_weight = gather_local_parameter(weight, group, rank, size)
    full_bias = gather_local_parameter(bias, group, rank, size) if bias is not None else None
    inputs = InputGradientSum.apply(inputs, group)
    full = F.linear(inputs, full_weight, full_bias)
    return full.narrow(-1, rank * weight.shape[0], weight.shape[0])


class Observation:
    def __init__(self, directory, rank):
        if not str(rank).isdecimal():
            raise ValueError("Numeric distributed rank required")
        self.path = Path(directory) / f"fp32_reconstructed_kv.rank{rank}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "rank": int(rank),
            "scope": "experimental eager FP32 K/V reconstruction; not a proven correctness fix",
            "modules": {},
            "completed": False,
            "failure": None,
        }
        with self.path.open("x") as handle:
            json.dump(self.data, handle, indent=2)
            handle.write("\n")

    def save(self):
        temp = self.path.with_name(self.path.name + f".{os.getpid()}.partial")
        temp.write_text(json.dumps(self.data, indent=2) + "\n")
        os.replace(temp, self.path)

    def finish(self, error=None):
        if error is None and not any(module["forward_calls"] for module in self.data["modules"].values()):
            error = "No reconstructed K/V forward was observed"
        self.data["failure"] = error
        self.data["completed"] = error is None
        self.save()
        if error == "No reconstructed K/V forward was observed":
            raise RuntimeError(error)


def install(observation):
    """Mark K/V modules after construction without replacing registered objects."""
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

    if getattr(TensorParallelLinear, "_fp32_reconstructed_kv_installed", False):
        raise RuntimeError("FP32 reconstructed K/V diagnostic already installed")
    original_stage = PipelineStage.__init__
    original_forward = TensorParallelLinear.forward

    def initialize_stage(self, *args, **kwargs):
        original_stage(self, *args, **kwargs)
        count = 0
        for layer_name, layer in self.layers.items():
            for module, suffix, path in (
                (layer.self_attn, "", "self_attn"),
                (layer.self_attn, "_moe_gen", "self_attn"),
                (getattr(layer, "action_block_moe_gen_action", None), "", "action_block_moe_gen_action"),
            ):
                if module is None:
                    continue
                for name in ("k_proj", "v_proj"):
                    projection = getattr(module, name + suffix)
                    if (
                        not isinstance(projection, TensorParallelLinear)
                        or not projection.column
                        or projection.tp_size <= 1
                    ):
                        raise ValueError("Every selected K/V projection must be column-parallel with TP > 1")
                    projection._fp32_reconstructed_kv = True
                    projection._fp32_reconstructed_kv_name = f"layers.{layer_name}.{path}.{name}{suffix}"
                    observation.data["modules"][projection._fp32_reconstructed_kv_name] = {
                        "local_weight_shape": list(projection.weight.shape),
                        "trainable": projection.weight.requires_grad,
                        "forward_calls": 0,
                    }
                    count += 1
        if not count:
            raise ValueError("No eligible local K/V projections were found")
        observation.save()

    def forward(self, inputs):
        if not getattr(self, "_fp32_reconstructed_kv", False):
            return original_forward(self, inputs)
        if self.weight.shape[0] * self.tp_size != self.out_features:
            raise ValueError("Local K/V shard shape differs from native output width")
        observation.data["modules"][self._fp32_reconstructed_kv_name]["forward_calls"] += 1
        return reconstructed_linear(inputs, self.weight, self.bias, self._impl.tp_group)

    PipelineStage.__init__ = initialize_stage
    TensorParallelLinear.forward = forward
    TensorParallelLinear._fp32_reconstructed_kv_installed = True
