"""Construct source-compatible blocks directly on meta, one local depth at a time."""

import torch

from cosmos_framework.model.generator.mot.small_action_tower import ActionBlock, action_shape
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer


def nano_layer_factory(text_config, *, action_preset="width_only_1b"):
    shape = action_shape(action_preset)
    if action_preset != "width_only_1b" or (
        text_config.hidden_size,
        text_config.intermediate_size,
        text_config.num_hidden_layers,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    ) != (4096, 12288, 36, 32, 8, 128):
        raise ValueError("TP/PP production factory requires the planned dense Nano / width_only_1b layout")
    layer_types = LayerTypes("qwen3_vl_dense")

    def create(global_index):
        with torch.device("meta"):
            layer = MoTDecoderLayer(
                text_config,
                layer_idx=global_index,
                layer_types=layer_types,
                qk_norm_for_text=True,
                qk_norm_for_diffusion=True,
            )
            layer.action_block_moe_gen_action = ActionBlock(
                shape,
                layer_types.rms_norm,
                text_config.rms_norm_eps,
            )
            layer.small_action_enabled = True
        # Keep the source recipe's UND freeze and generator parameter identities.
        for name, parameter in layer.named_parameters():
            parameter.requires_grad_("_moe_gen" in name)
        return layer

    return create
