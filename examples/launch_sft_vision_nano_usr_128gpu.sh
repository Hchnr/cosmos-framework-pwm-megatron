#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Launch once on each platform node; defaults to 16 x 8 GPUs, GA2 with token-budget packing.
# Single-node smoke: NNODES=1 MAX_ITER=23002 bash <this-script> \
#   model.config.parallelism.data_parallel_shard_degree=8
# Additional Hydra overrides are accepted as individually quoted arguments.
set -euo pipefail
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f /root/.venv/bin/activate ]]; then
        source /root/.venv/bin/activate
    elif [[ -f /opt/venv/bin/activate ]]; then
        source /opt/venv/bin/activate
    fi
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_OVERRIDE:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOML_FILE="${SFT_TOML_FILE:-$REPO_ROOT/examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml}"
# Model, data and checkpoint defaults live in TOML. Only explicit environment
# values override them; the shared launcher checks explicit path overrides.
export NNODES="${NNODES:-16}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-50012}"
if (( NNODES > 1 )); then
    : "${RANK:?The platform must provide node RANK}"
    : "${MASTER_ADDR:?The platform must provide MASTER_ADDR}"
    if [[ -z "${JOB_ID:-}" && -f /etc/downwardapi/annotations ]]; then
        JOB_ID="$(sed -n 's/.*scheduling.k8s.io\/group-name="\([^"]*\)".*/\1/p' /etc/downwardapi/annotations | head -1)"
    fi
    : "${JOB_ID:?Set JOB_ID to the same rendezvous ID on every node}"
    export RDZV_BACKEND=c10d RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT" RDZV_ID="$JOB_ID"
    export RDZV_CONF="${RDZV_CONF:-join_timeout=7200,last_call_timeout=60}"
else
    export NODE_RANK=0 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WANDB_API_KEY="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"
export WANDB_ENTITY="${WANDB_ENTITY:-ybainlp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/usr_v21}"
export LOG_FILENAME="${LOG_FILENAME:-nano_usr_node${RANK:-0}.log}"
mkdir -p "$OUTPUT_ROOT/logs"
_BOOTSTRAP_LOG="$OUTPUT_ROOT/logs/nano_usr_bootstrap_node${RANK:-0}.log"
exec > >(tee -a "$_BOOTSTRAP_LOG") 2>&1
REQUIRE_WAN_VAE=0
# DATASET_PATH invokes the shared launcher's Parquet existence check.
DATASET_PATH="${USR_DATASET_ROOT:-}"
EXTRA_OVERRIDES=()
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_TAIL_OVERRIDES:-}"
TAIL_OVERRIDES=()
if [[ -n "${TRAIN_RESOLUTION:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.resolution=$TRAIN_RESOLUTION")
fi
if [[ -n "${MAX_ITER:-}" ]]; then
    TAIL_OVERRIDES+=("trainer.max_iter=$MAX_ITER")
fi
# Explicit generic overrides take precedence over convenience environment flags.
TAIL_OVERRIDES+=("${EXTRA_OVERRIDES[@]}" "$@")
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

echo ">>> nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE (training defaults from $TOML_FILE)"
source "$SCRIPT_DIR/_sft_launcher_common.sh"
