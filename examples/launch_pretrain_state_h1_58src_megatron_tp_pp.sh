#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${COSMOS_MEGATRON_PYTHON:-$PWD/.venv/bin/python}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/outputs/pretrain52src_state_h1_nano_192gpu_ga3_from17200_run01/cosmos3_action/action128/action128_wam_nano_52src_state_h1/checkpoints/iter_000001200}"
export MEGATRON_TP_PP_IMPORT_PATH="${MEGATRON_TP_PP_IMPORT_PATH:-/opt/cosmos-tp-pp-canonical-iter1200}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/megatron_tp_pp}"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export ACTION128_EPISODE_CACHE_DIR="${ACTION128_EPISODE_CACHE_DIR-/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1}"
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export MIMALLOC_ALLOW_LARGE_OS_PAGES="${MIMALLOC_ALLOW_LARGE_OS_PAGES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
if [[ "$(ulimit -Sn)" != unlimited ]] && (( $(ulimit -Sn) < 65536 )); then
    ulimit -Sn 65536
fi
export LD_LIBRARY_PATH="${COSMOS_FFMPEG_LIBRARY_PATH:-}"
if [[ -z "$LD_LIBRARY_PATH" && -d /opt/cosmos-megatron-ffmpeg8 ]]; then
    export LD_LIBRARY_PATH=/opt/cosmos-megatron-ffmpeg8
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Original packed capacities need contiguous large allocations during 1F1B
# recomputation. Keep explicit user allocator settings authoritative.
if [[ -z "${PYTORCH_ALLOC_CONF+x}" && -z "${PYTORCH_CUDA_ALLOC_CONF+x}" ]]; then
    export PYTORCH_ALLOC_CONF=expandable_segments:True
fi
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC="${COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC:-7200}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
launch=(--standalone)
if [[ "$NNODES" != 1 ]]; then
    launch=(--nnodes "$NNODES" --node-rank "${NODE_RANK:?Set NODE_RANK}" --rdzv-backend c10d
            --rdzv-endpoint "${MASTER_ADDR:?Set MASTER_ADDR}:${MASTER_PORT:-29501}"
            --rdzv-id "${RDZV_ID:-cosmos_tp_pp}")
fi
exec "$PYTHON" -m torch.distributed.run "${launch[@]}" --nproc-per-node "$NPROC_PER_NODE" \
    -m cosmos_framework.scripts.train \
    --sft-toml examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml \
    -- trainer.seed=42 "$@"
