# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Extend the opt-in K/V projection diagnostic to Q on all three towers."""


def mark_q_projections(observation):
    from cosmos_framework.model.generator.megatron_tp_pp.layers import (
        TensorParallelLinear,
    )
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

    original_stage = PipelineStage.__init__

    def initialize(stage, *args, **kwargs):
        original_stage(stage, *args, **kwargs)
        count = 0
        for layer_name, layer in stage.layers.items():
            for module, suffix, path in (
                (layer.self_attn, "", "self_attn"),
                (layer.self_attn, "_moe_gen", "self_attn"),
                (
                    getattr(layer, "action_block_moe_gen_action", None),
                    "",
                    "action_block_moe_gen_action",
                ),
            ):
                if module is None:
                    continue
                projection = getattr(module, "q_proj" + suffix)
                if not isinstance(projection, TensorParallelLinear) or not projection.column or projection.tp_size <= 1:
                    raise ValueError("All selected Q projections must be column-parallel with TP > 1")
                name = f"layers.{layer_name}.{path}.q_proj{suffix}"
                projection._fp32_reconstructed_kv = True
                projection._fp32_reconstructed_kv_name = name
                observation.data["modules"][name] = {
                    "local_weight_shape": list(projection.weight.shape),
                    "trainable": projection.weight.requires_grad,
                    "forward_calls": 0,
                }
                count += 1
        if not count:
            raise ValueError("No eligible Q projection was found")
        observation.save()

    PipelineStage.__init__ = initialize
