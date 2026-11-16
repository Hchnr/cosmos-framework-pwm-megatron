#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash examples/_pretrain_state_paths.sh
# Preserve an explicitly activated environment; otherwise use the shared training venv.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${PRETRAIN_STATE_VENV:-/share-mixed/project/liyue/cosmos-framework/.venv}/bin/activate"
fi
export LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SFT_TOML_FILE=examples/toml/sft_config/action128_pretrain_state_h1_smoke.toml
export HF_MODEL_PATH=/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM
export WAN_VAE_PATH=/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/share-mixed/project/liyue/model_ckpts/mid_train_backup/pretrain23src_iter17200_eef_state32_from0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/share-mixed/projset/liyue/model_ckpts/pretrain_state_h1_main_6src_state32_compiled_smoke_20261028}"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export NNODES=1 NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export DP_SHARD="${DP_SHARD:-$NPROC_PER_NODE}" NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29653}"
export WANDB_MODE=disabled
export WANDB_API_KEY_FILE=/share-mixed/project/eai_pwm/.secrets/cosmos_wandb_api_key
ulimit -Sn 65536
exec bash examples/launch_sft_action128_wam_nano.sh "$@"
