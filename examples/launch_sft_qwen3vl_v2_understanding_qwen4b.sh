#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Eight-GPU launcher for the frozen V2 20-way Understanding mixture.
# Existing /share inputs are read-only; OUTPUT_ROOT defaults under this /home checkout.
#
# Usage:
#   SMOKE_RUN=1 bash examples/launch_sft_qwen3vl_v2_understanding_qwen4b.sh
#
# Optional:
#   QWEN3_VL_4B_PATH  local Qwen3-VL-4B HF snapshot
#   OUTPUT_ROOT        logs/checkpoints root (must remain outside /share)
#   MASTER_PORT        torchrun rendezvous port
#   NPROC_PER_NODE     defaults to 8

TOML_FILE="examples/toml/sft_config/qwen3vl_v2_0a_understanding_qwen4b.toml"
: "${QWEN3_VL_4B_PATH:=/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct}"
export QWEN3_VL_4B_PATH

PARQUET_ROOT="/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1"
[[ -d "$QWEN3_VL_4B_PATH" ]] || {
    echo "ERROR: QWEN3_VL_4B_PATH not found: $QWEN3_VL_4B_PATH" >&2
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
    # Qwen's model-instance config is OmegaConf-structured. Predeclare this
    # family-native visual-tower switch so OmniMoT can update it during build.
    "+model.config.vlm_config.model_instance.config.include_visual=true"
    # Keep the instantiated Qwen/MoT config aligned with the catalog packer's
    # production token budget selected by the 8-GPU packing sweep.
    "+model.config.vlm_config.model_instance.config.max_num_tokens_after_packing=49152"
    # Pure Understanding has no generation sampling path. Remove inherited
    # generation visualization callbacks instead of attempting S3 writes.
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
        "job.name=qwen3vl_v2_0a_understanding_qwen4b_smoke"
        "trainer.max_iter=$SMOKE_MAX_ITER"
        "trainer.grad_accum_iter=$SMOKE_GRAD_ACCUM_ITER"
        "trainer.logging_iter=1"
        "checkpoint.save_iter=$SMOKE_CHECKPOINT_SAVE_ITER"
        "model.config.compile.enabled=false"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
