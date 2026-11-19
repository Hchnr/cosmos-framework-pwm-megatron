# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in local TP projection fusion, pending true-model validation.

Each existing weight remains a separate Parameter; only temporary rank-local
weights/gradients are concatenated. Disconnected outputs retain None gradients.
Install before stage construction; no production default is changed.
"""

import json
import os
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class _ProjectionGradient(torch.autograd.Function):
    """Attach an independent autograd edge to one fused-forward slice."""

    @staticmethod
    def forward(ctx, inputs, weight, output):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(inputs, weight)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return None, None, None
        inputs, weight = ctx.saved_tensors
        grad = gradient.reshape(-1, gradient.shape[-1])
        input_gradient = None
        if ctx.needs_input_grad[0]:
            input_gradient = (grad @ weight).reshape(inputs.shape)
        weight_gradient = (
            grad.transpose(0, 1) @ inputs.reshape(-1, inputs.shape[-1]) if ctx.needs_input_grad[1] else None
        )
        return input_gradient, weight_gradient, None


def fused_local_linear(inputs, *weights):
    """Fuse the forward only, preserving independent gradient graph edges.

    The compiler can materialize unused outputs of a single multi-output
    autograd Function as zero gradients. Separate edges let unused projections
    remain unused when the branch is omitted within the compiled graph.
    """
    widths = tuple(weight.shape[0] for weight in weights)
    fused_weight = torch.cat([weight.detach() for weight in weights], dim=0)
    output = F.linear(inputs.detach(), fused_weight)
    slices = output.split(widths, dim=-1)
    return tuple(
        _ProjectionGradient.apply(inputs, weight, value) for weight, value in zip(weights, slices, strict=True)
    )


def ineligibility_reasons(projections, *, projection_type=None):
    if not projections or any(projection is None for projection in projections):
        return ["missing_projection_or_module"]
    if projection_type is None:
        from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear

        projection_type = TensorParallelLinear
    if not all(isinstance(projection, projection_type) for projection in projections):
        return ["not_tensor_parallel_linear"]
    first = projections[0]
    reasons = []
    if any(not projection.column for projection in projections):
        reasons.append("not_column_parallel")
    if any(projection.tp_size != first.tp_size for projection in projections):
        reasons.append("different_tp_size")
    if first.tp_size <= 1:
        reasons.append("tp_size_not_greater_than_one")
    if any(projection._impl.tp_group is not first._impl.tp_group for projection in projections):
        reasons.append("different_tp_group")
    if any(projection.bias is not None for projection in projections):
        reasons.append("bias_present")
    if any(projection.weight.ndim != 2 for projection in projections):
        reasons.append("weight_not_matrix")
    elif any(projection.weight.shape[1] != first.weight.shape[1] for projection in projections):
        reasons.append("different_input_width")
    if any(projection.weight.dtype != first.weight.dtype for projection in projections):
        reasons.append("different_weight_dtype")
    if any(projection.weight.device != first.weight.device for projection in projections):
        reasons.append("different_weight_device")
    return reasons


class FusionObservation:
    """Record installed stage modules; this does not claim runtime kernel fusion."""

    def __init__(self, report_dir, rank, *, fuse_qkv, fuse_gate_up):
        if not str(rank).isdecimal():
            raise ValueError("A numeric distributed rank is required for the fusion report")
        self.target = Path(report_dir) / f"fused_input_projection.rank{rank}.json"
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "scope": "stage-local module installation; execution and GPU kernel fusion must be checked separately",
            "rank": int(rank),
            "requested": {"qkv": fuse_qkv, "gate_up": fuse_gate_up},
            "stages": [],
            "enabled_counts": {"qkv": 0, "gate_up": 0},
            "ineligible_reason_counts": {"qkv": {}, "gate_up": {}},
            "completed": False,
            "failure": None,
        }
        with self.target.open("x") as handle:
            json.dump(self.data, handle, indent=2)
            handle.write("\n")

    def save(self):
        temp = self.target.with_name(self.target.name + f".{os.getpid()}.partial")
        temp.write_text(json.dumps(self.data, indent=2) + "\n")
        os.replace(temp, self.target)

    def start_stage(self):
        stage = {"stage_index": len(self.data["stages"]), "modules": []}
        self.data["stages"].append(stage)
        return stage

    def record(self, stage, module, kind, projections, *, reasons=None):
        reasons = ineligibility_reasons(projections) if reasons is None else list(reasons)
        requested = self.data["requested"][kind]
        enabled = requested and not reasons
        stage["modules"].append({"module": module, "kind": kind, "enabled": enabled, "reasons": reasons})
        self.data["enabled_counts"][kind] += int(enabled)
        counts = Counter(self.data["ineligible_reason_counts"][kind])
        counts.update(reasons)
        self.data["ineligible_reason_counts"][kind] = dict(counts)
        return enabled

    def require_enabled(self):
        missing = [
            key
            for key, requested in self.data["requested"].items()
            if requested and not self.data["enabled_counts"][key]
        ]
        if missing:
            raise RuntimeError("No eligible TP stage modules enabled for: " + ", ".join(missing))

    def finish(self, *, error=None):
        missing = [
            key
            for key, requested in self.data["requested"].items()
            if requested and not self.data["enabled_counts"][key]
        ]
        if error is None and missing:
            error = "No eligible TP stage modules enabled for: " + ", ".join(missing)
        self.data["failure"] = error
        self.data["completed"] = error is None
        self.save()
        if missing and error.startswith("No eligible TP stage modules"):
            raise RuntimeError(error)


def fused_projection(projections, inputs):
    from tools.tp_pp_shared_input_reduction import shared_input

    inputs = shared_input(projections, inputs)
    return fused_local_linear(inputs, *(projection.weight for projection in projections))


class FusedInputMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn

    def forward(self, inputs):
        gate, up = fused_projection((self.gate_proj, self.up_proj), inputs)
        return self.down_proj(self.act_fn(gate) * up)


def install(*, fuse_qkv=True, fuse_gate_up=True, observation):
    """Install the shared-reduction candidate plus optional fused local GEMMs."""
    if not fuse_qkv and not fuse_gate_up:
        raise ValueError("At least one fused projection must be enabled")
    if observation.data["requested"] != {"qkv": fuse_qkv, "gate_up": fuse_gate_up}:
        raise ValueError("Fusion report does not match requested configuration")
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.model.generator.mot import small_action_tower
    from tools.tp_pp_shared_input_reduction import install as install_shared

    install_shared()
    original_stage = PipelineStage.__init__
    original_qkv = small_action_tower.project_qkv
    original_action_mlp = small_action_tower.ActionBlock.mlp

    def initialize_stage(self, *args, **kwargs):
        original_stage(self, *args, **kwargs)
        stage = observation.start_stage()
        for layer_name, layer in self.layers.items():
            attention = layer.self_attn
            for suffix in ("", "_moe_gen"):
                projections = tuple(getattr(attention, name + suffix) for name in ("q_proj", "k_proj", "v_proj"))
                enabled = observation.record(stage, f"layers.{layer_name}.self_attn{suffix}", "qkv", projections)
                setattr(attention, "_fused_local_qkv" + suffix, enabled)
            for name in ("mlp", "mlp_moe_gen"):
                mlp = getattr(layer, name, None)
                projections = (mlp.gate_proj, mlp.up_proj) if mlp is not None else ()
                enabled = observation.record(stage, f"layers.{layer_name}.{name}", "gate_up", projections)
                if enabled:
                    setattr(layer, name, FusedInputMLP(mlp))
            action = getattr(layer, "action_block_moe_gen_action", None)
            projections = tuple(getattr(action, name) for name in ("q_proj", "k_proj", "v_proj")) if action else ()
            action_qkv = observation.record(stage, f"layers.{layer_name}.action", "qkv", projections)
            projections = (action.gate_proj, action.up_proj) if action else ()
            action_gate_up = observation.record(stage, f"layers.{layer_name}.action", "gate_up", projections)
            if action is not None:
                action._fused_local_qkv = action_qkv
                action._fused_local_gate_up = action_gate_up
        observation.save()
        observation.require_enabled()

    def project_qkv(module, inputs, suffix=""):
        if not getattr(module, "_fused_local_qkv" + suffix, False):
            return original_qkv(module, inputs, suffix)
        projections = tuple(getattr(module, name + suffix) for name in ("q_proj", "k_proj", "v_proj"))
        return fused_projection(projections, inputs)

    def action_mlp(self, inputs):
        if not getattr(self, "_fused_local_gate_up", False):
            return original_action_mlp(self, inputs)
        gate, up = fused_projection((self.gate_proj, self.up_proj), inputs)
        return self.down_proj(F.silu(gate) * up)

    PipelineStage.__init__ = initialize_stage
    small_action_tower.project_qkv = project_qkv
    small_action_tower.ActionBlock.mlp = action_mlp
