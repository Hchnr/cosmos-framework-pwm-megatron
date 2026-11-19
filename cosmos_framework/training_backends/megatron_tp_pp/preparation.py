# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native VAE/noising and loss helpers without constructing a full decoder."""

import torch
from torch import nn

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from .batch import native_sample_level_loss


class GenerationPreparation(OmniMoTModel):
    """Construct tokenizers only on the first-stage TP data owner.

    The trainable input adapter remains owned by PipelineStage. Preparation
    calls the original model's non-decoder path and shares its normalization
    buffers; it never invokes the adapter's trainable forward.
    """

    def __init__(self, config, input_adapter, process_group=None):
        object.__setattr__(self, "_input_adapter", input_adapter)
        object.__setattr__(self, "_tokenizer_process_group", process_group)
        super().__init__(config)

    def set_up_parallelism(self):
        self.parallel_dims = None

    def set_up_model(self):
        self.net = self._input_adapter

    @torch.no_grad()
    def prepare(self, raw_batch, iteration):
        prepared = self.prepare_generation_training_inputs(raw_batch, iteration)
        memory = self.build_memory_state(prepared["packed_sequence"], prepared["memory_info"])
        if memory is not None or any(prepared["vlm_inputs"]):
            raise ValueError("The TP/PP action recipe requires no memory or visual reasoner inputs")
        clean = prepared["gen_data_clean"]
        return {
            "packed_sequence": prepared["packed_sequence"],
            "loss_inputs": {
                "gen_data_noised": prepared["gen_data_noised"],
                "timesteps": prepared["timesteps_vision"],
                "timesteps_action": prepared["timesteps_action"],
                "timesteps_sound": prepared["timesteps_sound"],
                "is_image_batch": clean.is_image_batch,
            },
            "sample_count": clean.batch_size,
        }


class GenerationLoss(OmniMoTModel):
    """Native loss/scheduler math on the last stage, without a VAE or blocks."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config, self.parallel_dims = config, None
        self.net = nn.Identity()
        self.set_precision()
        self.set_up_scheduler_and_sampler()

    def forward(self, outputs, batch):
        weight = batch.get("weight")
        if native_sample_level_loss(self.config) and weight is None:
            raise ValueError("Prepare logical micro-step sample weights before the pipeline schedule")
        if weight is not None:
            weight = torch.as_tensor(weight, **self.tensor_kwargs_fp32)
        value, metrics = self._compute_losses(
            out_net=outputs,
            data_batch_packed=batch["packed_sequence"],
            **batch["loss_inputs"],
            sample_level_scale_override=weight,
        )
        # MCore scales the returned loss in place by 1 / num_microbatches.
        # detach() alone aliases that storage and silently scales the logged
        # metric too. Preserve the original loss without changing backward.
        return value, {
            "loss": value.detach().clone(),
            **{key: tensor.detach().mean() for key, tensor in metrics.items()},
        }
