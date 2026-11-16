#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Eight-GPU launcher for the frozen V2 20-way Understanding mixture on
# Cosmos3-Nano (Qwen3-VL-8B tier). Existing /share inputs remain read-only;
# OUTPUT_ROOT must be outside /share.

TOML_FILE="examples/toml/sft_config/qwen3vl_v2_0a_understanding_nano8b.toml"
: "${BASE_CHECKPOINT_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP}"
: "${HF_MODEL_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano}"
export BASE_CHECKPOINT_PATH HF_MODEL_PATH

PARQUET_ROOT="/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1"
[[ -d "$BASE_CHECKPOINT_PATH" ]] || {
    echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2
    exit 1
}
[[ -d "$HF_MODEL_PATH" ]] || {
    echo "ERROR: HF_MODEL_PATH not found: $HF_MODEL_PATH" >&2
    exit 1
}
[[ -d "$PARQUET_ROOT" ]] || {
    echo "ERROR: V2 Parquet root not found: $PARQUET_ROOT" >&2
    exit 1
}
[[ "${OUTPUT_ROOT:-}" != /share && "${OUTPUT_ROOT:-}" != /share/* ]] || {
    echo "ERROR: OUTPUT_ROOT must not be under read-only /share: $OUTPUT_ROOT" >&2
    exit 1
}

REQUIRE_WAN_VAE=0
PREFER_PROJECT_CUDA_LIBS=1
TAIL_OVERRIDES=(
    # Qwen's model-instance config is OmegaConf-structured. Predeclare the
    # family-native visual-tower switch before OmniMoT updates it at build time.
    "+model.config.vlm_config.model_instance.config.include_visual=true"
    "model.config.vlm_config.tokenizer.pretrained_model_name=$HF_MODEL_PATH"
    # Pure Understanding does not use inherited generation sampling callbacks.
    "~trainer.callbacks.every_n_sample_reg=null"
    "~trainer.callbacks.every_n_sample_ema=null"
    ${EXTRA_TAIL_OVERRIDES:-}
)

if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    : "${SMOKE_MAX_ITER:=2}"
    : "${SMOKE_CHECKPOINT_SAVE_ITER:=$SMOKE_MAX_ITER}"
    : "${SMOKE_GRAD_ACCUM_ITER:=1}"
    [[ "$SMOKE_MAX_ITER" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: SMOKE_MAX_ITER must be a positive integer: $SMOKE_MAX_ITER" >&2
        exit 2
    }
    [[ "$SMOKE_CHECKPOINT_SAVE_ITER" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: SMOKE_CHECKPOINT_SAVE_ITER must be a positive integer: $SMOKE_CHECKPOINT_SAVE_ITER" >&2
        exit 2
    }
    [[ "$SMOKE_GRAD_ACCUM_ITER" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: SMOKE_GRAD_ACCUM_ITER must be a positive integer: $SMOKE_GRAD_ACCUM_ITER" >&2
        exit 2
    }
    TAIL_OVERRIDES+=(
        "job.name=qwen3vl_v2_0a_understanding_nano8b_smoke"
        "trainer.max_iter=$SMOKE_MAX_ITER"
        "trainer.grad_accum_iter=$SMOKE_GRAD_ACCUM_ITER"
        "trainer.logging_iter=1"
        "checkpoint.save_iter=$SMOKE_CHECKPOINT_SAVE_ITER"
        "model.config.compile.enabled=false"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
