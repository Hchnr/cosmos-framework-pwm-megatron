#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_edge (T2V / I2V / V2V vision-only
# SFT on Nemotron-2B-Dense-VL / Cosmos3-Edge, 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_edge.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          visual-generation Parquet catalog file or directory
#   BASE_CHECKPOINT_PATH  default: /share/project/liyue/model_ckpts/Cosmos3-Edge-DCP
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              not needed for nvidia/Cosmos3-Edge (the repo is
#                         ungated); set only if another download requires it
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_vision_edge.sh

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
: "${DATASET_PATH:?Set DATASET_PATH to a visual-generation Parquet catalog}"
: "${BASE_CHECKPOINT_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Edge-DCP}"
: "${HF_MODEL_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Edge}"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
    "model.config.vlm_config.tokenizer.repository=null"
    "model.config.vlm_config.tokenizer.revision=null"
    "+model.config.vlm_config.tokenizer.tokenizer_type=$HF_MODEL_PATH"
)
if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    TAIL_OVERRIDES+=(
        "job.name=vision_sft_edge_smoke"
        "trainer.max_iter=2"
        "trainer.grad_accum_iter=1"
        "checkpoint.save_iter=2"
        "model.config.compile.enabled=false"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
