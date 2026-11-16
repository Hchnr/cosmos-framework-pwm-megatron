#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Multi-node Nano pure-T2V training on 128 GPUs.
# Default topology: 16 nodes x 8 GPUs, grad accumulation 2, global packed-
# sequence batch size 256. The platform launches this script once on every node;
# all nodes join the same c10d rendezvous using the Kubernetes scheduling job ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export WANDB_API_KEY="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"
export WANDB_ENTITY="${WANDB_ENTITY:-ybainlp}"

export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export SFT_TOML_FILE="${SFT_TOML_FILE:-examples/toml/sft_config/vision_sft_nano_t2v_8gpu_fixed_static.toml}"
export TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-480}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

# These are provided by the training platform. RANK identifies the node for
# logs; c10d assigns the actual distributed ranks after rendezvous.
: "${RANK:?The platform must provide the node RANK}"
: "${MASTER_ADDR:?The platform must provide MASTER_ADDR}"
: "${MASTER_PORT:=50012}"
if [[ -z "${JOB_ID:-}" ]]; then
    JOB_ID="$(grep -oP 'scheduling.k8s.io/group-name="\K[^"]+' /etc/downwardapi/annotations | head -1)"
fi
: "${JOB_ID:?Unable to read the scheduling group name from /etc/downwardapi/annotations}"

mkdir -p "$OUTPUT_ROOT/logs"
_BOOTSTRAP_LOG="$OUTPUT_ROOT/logs/vision_sft_nano_t2v_128gpu_bootstrap_node${RANK}.log"
exec > >(tee -a "$_BOOTSTRAP_LOG") 2>&1
echo ">>> $(date '+%H:%M:%S') topology: platform_rank=$RANK nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE rendezvous=$MASTER_ADDR:$MASTER_PORT id=$JOB_ID"


# Every node runs the launcher. Keep their tee output separate when OUTPUT_ROOT
# is a shared filesystem; W&B itself is initialized only by global rank 0.
export LOG_FILENAME="${LOG_FILENAME:-vision_sft_nano_t2v_128gpu_node${RANK}.log}"
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT"
export RDZV_ID="$JOB_ID"
export RDZV_CONF="${RDZV_CONF:-join_timeout=7200,last_call_timeout=60}"

exec bash "$SCRIPT_DIR/launch_sft_vision_nano_t2v_8gpu.sh"
