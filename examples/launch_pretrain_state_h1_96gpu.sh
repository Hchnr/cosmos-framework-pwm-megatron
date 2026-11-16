#!/usr/bin/env bash
# Latest-main-based State pretraining; run on each of 12 eight-GPU nodes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash examples/_pretrain_state_paths.sh
export LD_LIBRARY_PATH=''
export SFT_TOML_FILE="${SFT_TOML_FILE:-$PWD/examples/toml/sft_config/action128_wam_nano_state_h1.toml}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/share-mixed/project/liyue/model_ckpts/mid_train_backup/pretrain23src_iter17200_eef_state32_from0}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/share-mixed/project/eai_pwm/.secrets/cosmos_wandb_api_key}"
export NNODES="${NNODES:-12}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}" DP_SHARD="${DP_SHARD:-16}"
export MASTER_PORT="${MASTER_PORT:-50012}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/share-mixed/projset/liyue/model_ckpts/pretrain24src_state_h1_nano_96gpu_ga3_from17200_run01}"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
exec bash examples/launch_sft_action128_wam_nano.sh "$@"
