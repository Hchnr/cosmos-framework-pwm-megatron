# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage-local production construction using the native model configuration."""

import torch

from cosmos_framework.configs.base.defaults.compile import inductor_compile_options
from cosmos_framework.model.generator.megatron_tp_pp.factory import nano_layer_factory
from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, _prepare_full_model_instance
from cosmos_framework.utils.lazy_config import instantiate

from .runtime import transformer_config


def build_stage(model_config, topology):
    """Do not instantiate the full HF model, decoder or VAE on any rank."""
    instance, _, _ = _prepare_full_model_instance(
        model_config.vlm_config.model_instance, load_vlm_visual=False, include_generation_tower=True
    )
    language_config = instantiate(instance["config"])
    text = getattr(language_config, "text_config", language_config)
    view = OmniMoTModel.__new__(OmniMoTModel)
    view.__dict__.update(config=model_config, tokenizer_vision_gen=None)
    network = view.build_network_config(language_config)
    dtype = getattr(torch, model_config.precision)
    mcore = transformer_config(
        topology,
        num_layers=text.num_hidden_layers,
        hidden_size=text.hidden_size,
        num_attention_heads=text.num_attention_heads,
        dtype=dtype,
        variable_seq_lengths=True,
    )
    fixed = model_config.fixed_sequence_length
    layout = PayloadLayout(
        ((fixed.und_tokens, 4096), (fixed.gen_tokens - fixed.action_tokens, 4096), (fixed.action_tokens, 1280))
        if fixed.enabled
        else ((0, 4096), (0, 4096), (0, 1280)),
        context_width=5376,
    )
    stage = (
        PipelineStage(
            mcore,
            nano_layer_factory(text),
            layout=layout,
            input_factory=lambda: FirstStageInputs(network, language_config, mcore),
            output_factory=lambda: LastStageOutputs(network, language_config, mcore),
        )
        .to(dtype=dtype)
        .to_empty(device="cuda")
    )
    if stage.input_adapter is not None:
        stage.input_adapter._warm_full_und = (
            model_config.compile.enabled
            and not model_config.compile.compile_dynamic
            and fixed.enabled
            and model_config.action_tower_und_short_tokens > 0
        )
    return stage, network


def optimize_stage(stage, model_config):
    """Keep native per-block AC and Inductor options, outside I/O/P2P."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    ac, compile_config = model_config.activation_checkpointing, model_config.compile
    if ac.mode not in ("none", "full"):
        raise ValueError("TP/PP currently supports full AC or none")
    if compile_config.enabled and (compile_config.compiled_region != "language" or compile_config.use_cuda_graphs):
        raise ValueError("TP/PP training supports per-language-block compile without CUDA graphs")
    options = inductor_compile_options(compile_config)
    for name, block in list(stage.layers.items()):
        if ac.mode == "full":
            block = checkpoint_wrapper(
                block, preserve_rng_state=ac.preserve_rng_state, determinism_check=ac.determinism_check
            )
        if compile_config.enabled:
            block = torch.compile(
                block, fullgraph=True, dynamic=compile_config.compile_dynamic, options=options or None
            )
        stage.layers[name] = block
