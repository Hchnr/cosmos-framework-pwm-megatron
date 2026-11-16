#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Qwen3-VL-4B 4+4 Joint-MoT pure-GEN SFT on a visual-generation Parquet
# catalog. The model starts from the local Qwen3-VL-4B Hugging Face snapshot;
# no base DCP is loaded.
#
# Required:
#   DATASET_PATH          visual-generation Parquet catalog file or directory
#
# Optional:
#   QWEN3_VL_4B_PATH     local Qwen3-VL-4B snapshot
#   WAN_VAE_PATH          local Wan2.2 VAE checkpoint
#   OUTPUT_ROOT           training output root
#   SMOKE_RUN=1           two-step, one-sample-per-microbatch smoke

TOML_FILE="examples/toml/sft_config/vision_sft_qwen4b_gen.toml"
: "${DATASET_PATH:?Set DATASET_PATH to a visual-generation Parquet catalog}"
: "${QWEN3_VL_4B_PATH:=/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct}"

# Reuse the common launcher's local-HF validation and export the variable used
# directly by the Qwen experiment config.
HF_MODEL_PATH="$QWEN3_VL_4B_PATH"
export QWEN3_VL_4B_PATH

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
    "model.config.resolution='256'"
)
if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    TAIL_OVERRIDES+=(
        "job.name=vision_sft_qwen4b_gen_smoke"
        "trainer.max_iter=2"
        "trainer.grad_accum_iter=1"
        "checkpoint.save_iter=2"
        "model.config.compile.enabled=false"
        "dataloader_train.max_samples_per_batch=1"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
