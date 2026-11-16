#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -e /share/mixed ] && [ ! -L /share/mixed ]; then
    ln -s /share-mixed /share/mixed
fi
export PATH="/share-mixed/project/liyue/cosmos-framework/.venv/bin:$PATH"
export VIRTUAL_ENV=/share-mixed/project/liyue/cosmos-framework/.venv
export LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SFT_TOML_FILE="${SFT_TOML_FILE:-examples/toml/sft_config/action128_pretrain_state_h1_52src_8gpu.toml}"
export HF_MODEL_PATH=/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM
export WAN_VAE_PATH=/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/share-mixed/project/liyue/model_ckpts/mid_train_backup/pretrain23src_iter17200_eef_state32_from0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/share-mixed/projset/liyue/model_ckpts/pretrain_state_h1_52src_8gpu_20261029}"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export NNODES=1 NPROC_PER_NODE=8 DP_SHARD=8 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29654}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_API_KEY_FILE=/share-mixed/project/eai_pwm/.secrets/cosmos_wandb_api_key
ulimit -Sn 65536
exec bash examples/launch_sft_action128_wam_nano.sh "$@"
