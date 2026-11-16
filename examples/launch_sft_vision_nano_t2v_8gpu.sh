#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Single-node 8-GPU Nano pure-T2V training, global packed-sequence batch size 8.
# Run from the repository root. Preserve an already-active environment; otherwise
# use the standard developer or container virtual environment when available.

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f /root/.venv/bin/activate ]]; then
        source /root/.venv/bin/activate
    elif [[ -f /opt/venv/bin/activate ]]; then
        source /opt/venv/bin/activate
    fi
fi
export WANDB_API_KEY="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"
export WANDB_ENTITY="${WANDB_ENTITY:-ybainlp}"
# Override SFT_TOML_FILE to exercise a production-compatible variant, such as
# vision_sft_nano_t2v_8gpu_fixed_static.toml, without duplicating launcher plumbing.
TOML_FILE="${SFT_TOML_FILE:-examples/toml/sft_config/vision_sft_nano_t2v_8gpu.toml}"
: "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH to the shared Cosmos3-Nano DCP directory}"
: "${HF_MODEL_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano}"
: "${ACTION100M_DATASET_PATH:?Set ACTION100M_DATASET_PATH to the shared Action100M catalog directory}"
: "${KOALA_DATASET_PATH:?Set KOALA_DATASET_PATH to the shared Koala catalog directory}"
: "${NPROC_PER_NODE:=8}"
: "${TRAIN_RESOLUTION:=256}"
T2V_VAE_MODE="${T2V_VAE_MODE:-online}"
case "$T2V_VAE_MODE" in
    online)
        REQUIRE_WAN_VAE=1
        ;;
    offline)
        REQUIRE_WAN_VAE=0
        : "${ACTION100M_VAE_MANIFEST_PATH:?Set ACTION100M_VAE_MANIFEST_PATH for offline VAE training}"
        : "${KOALA_VAE_MANIFEST_PATH:?Set KOALA_VAE_MANIFEST_PATH for offline VAE training}"
        ;;
    *)
        echo "ERROR: T2V_VAE_MODE must be online or offline, got: $T2V_VAE_MODE" >&2
        exit 1
        ;;
esac
export ACTION100M_DATASET_PATH KOALA_DATASET_PATH ACTION100M_VAE_MANIFEST_PATH KOALA_VAE_MANIFEST_PATH

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
    "dataloader_train.visual_generation.vae_mode=$T2V_VAE_MODE"
    "model.config.vlm_config.tokenizer.pretrained_model_name=$HF_MODEL_PATH"
    "model.config.resolution=$TRAIN_RESOLUTION"
    "trainer.callbacks.compile_tokenizer.warmup_resolutions=[$TRAIN_RESOLUTION]"
)

if [[ "$T2V_VAE_MODE" == "offline" ]]; then
    EXTRA_DATASET_CHECK='[[ -f "$ACTION100M_VAE_MANIFEST_PATH" ]] || { echo "ERROR: ACTION100M_VAE_MANIFEST_PATH not found: $ACTION100M_VAE_MANIFEST_PATH" >&2; exit 1; }; [[ -f "$KOALA_VAE_MANIFEST_PATH" ]] || { echo "ERROR: KOALA_VAE_MANIFEST_PATH not found: $KOALA_VAE_MANIFEST_PATH" >&2; exit 1; }'
fi

# Optional hcr detailed tracing. For example:
#   DETAILED_NVTX_GRANULARITY=module NVTX_DETAIL_TARGET_RANKS='[0]' bash ...
# Valid granularities are coarse, phase, module, and operator.
if [[ -n "${DETAILED_NVTX_GRANULARITY:-}" ]]; then
    TAIL_OVERRIDES+=(
        "+trainer.callbacks.nvtx._target_=cosmos_framework.utils.callback.NVTXCallback"
        "+trainer.callbacks.nvtx.synchronize=false"
        "+trainer.callbacks.nvtx.granularity=${DETAILED_NVTX_GRANULARITY}"
        "+trainer.callbacks.nvtx.detail_target_ranks=${NVTX_DETAIL_TARGET_RANKS:-[0]}"
    )
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
