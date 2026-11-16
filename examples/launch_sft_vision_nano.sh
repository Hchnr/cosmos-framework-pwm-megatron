#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_nano (T2V / I2V / V2V vision-only
# SFT on Qwen3-VL-8B, 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_nano.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          visual-generation Parquet catalog file or directory
#   BASE_CHECKPOINT_PATH  default: /share/project/liyue/model_ckpts/Cosmos3-Nano-DCP
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_vision_nano.sh

TOML_FILE="examples/toml/sft_config/vision_sft_nano.toml"
: "${DATASET_PATH:?Set DATASET_PATH to a visual-generation Parquet catalog}"
: "${BASE_CHECKPOINT_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP}"
: "${HF_MODEL_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano}"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
    "model.config.vlm_config.tokenizer.pretrained_model_name=$HF_MODEL_PATH"
)
if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    TAIL_OVERRIDES+=(
        "job.name=vision_sft_nano_smoke"
        "trainer.max_iter=2"
        "trainer.grad_accum_iter=1"
        "checkpoint.save_iter=2"
        "model.config.compile.enabled=false"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
