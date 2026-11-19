# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Candidate: sum shared Q/K/V or gate/up dgrad before one TP reduction.

Weights, individual MCore GEMMs and checkpoint identities stay separate. The
shared input identity performs one SUM after autograd accumulates its local
branches, instead of reducing each branch separately. Unused edges remain None.
This is an opt-in candidate until full-model validation and timing complete.
"""

import torch
from torch import nn
from torch.nn import functional


def shared_input(projections, inputs):
    from cosmos_framework.model.generator.megatron_tp_pp.compile import CopyToTensorParallel

    first = projections[0]
    if first.tp_size > 1:
        return CopyToTensorParallel.apply(inputs, first._impl.tp_group.group_name)
    return inputs


def eligible(projections):
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear

    return len(projections) > 0 and all(
        isinstance(projection, TensorParallelLinear)
        and projection.column
        and projection.tp_size == projections[0].tp_size
        and projection._impl.tp_group is projections[0]._impl.tp_group
        for projection in projections
    )


class SharedInputMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn

    def forward(self, inputs):
        inputs = shared_input((self.gate_proj, self.up_proj), inputs)
        gate = self.gate_proj._shared_local_linear(inputs, self.gate_proj.weight, self.gate_proj.bias)
        up = self.up_proj._shared_local_linear(inputs, self.up_proj.weight, self.up_proj.bias)
        return self.down_proj(self.act_fn(gate) * up)


def install():
    from cosmos_framework.model.generator.megatron_tp_pp.compile import NonePreservingLinear
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.model.generator.mot import small_action_tower

    if getattr(TensorParallelLinear, "_shared_input_candidate_installed", False):
        raise RuntimeError("Shared-input reduction candidate is already installed")
    original_linear = TensorParallelLinear.__init__
    original_stage = PipelineStage.__init__
    original_qkv = small_action_tower.project_qkv
    original_action_mlp = small_action_tower.ActionBlock.mlp

    def initialize_linear(self, *args, **kwargs):
        original_linear(self, *args, **kwargs)
        group = self._impl.tp_group

        def local_linear(inputs, weight, bias):
            # The ProcessGroup stays in the closure, matching the existing
            # traceable MCore boundary. SharedInput supplies the only dgrad SUM.
            return NonePreservingLinear.apply(inputs, weight, bias, False, False, False, None, None, group, 1, None)

        self.__dict__["_shared_local_linear"] = torch.compiler.allow_in_graph(local_linear)

    def initialize_stage(self, *args, **kwargs):
        original_stage(self, *args, **kwargs)
        for layer in self.layers.values():
            attention = layer.self_attn
            for suffix in ("", "_moe_gen"):
                projections = tuple(getattr(attention, name + suffix) for name in ("q_proj", "k_proj", "v_proj"))
                setattr(attention, "_shared_input_qkv" + suffix, eligible(projections))
            for name in ("mlp", "mlp_moe_gen"):
                mlp = getattr(layer, name, None)
                if mlp is not None and eligible((mlp.gate_proj, mlp.up_proj)):
                    setattr(layer, name, SharedInputMLP(mlp))
            action = getattr(layer, "action_block_moe_gen_action", None)
            if action is not None:
                action._shared_input_qkv = eligible(
                    tuple(getattr(action, name) for name in ("q_proj", "k_proj", "v_proj"))
                )
                action._shared_input_gate_up = eligible((action.gate_proj, action.up_proj))

    def project_qkv(module, inputs, suffix=""):
        projections = tuple(getattr(module, name + suffix) for name in ("q_proj", "k_proj", "v_proj"))
        if not getattr(module, "_shared_input_qkv" + suffix, False):
            return original_qkv(module, inputs, suffix)
        inputs = shared_input(projections, inputs)
        return tuple(
            projection._shared_local_linear(inputs, projection.weight, projection.bias) for projection in projections
        )

    def action_mlp(self, inputs):
        projections = (self.gate_proj, self.up_proj)
        if not getattr(self, "_shared_input_gate_up", False):
            return original_action_mlp(self, inputs)
        inputs = shared_input(projections, inputs)
        gate, up = (
            projection._shared_local_linear(inputs, projection.weight, projection.bias) for projection in projections
        )
        return self.down_proj(functional.silu(gate) * up)

    TensorParallelLinear.__init__ = initialize_linear
    TensorParallelLinear._shared_input_candidate_installed = True
    PipelineStage.__init__ = initialize_stage
    small_action_tower.project_qkv = project_qkv
    small_action_tower.ActionBlock.mlp = action_mlp
