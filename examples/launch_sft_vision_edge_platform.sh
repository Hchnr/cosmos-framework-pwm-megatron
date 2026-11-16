#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Platform launcher for Cosmos3-Edge vision SFT on one 8-GPU node.
#
# Full training:
#   bash examples/launch_sft_vision_edge_platform.sh
#
# Short end-to-end validation (separate output job, two optimizer steps):
#   SMOKE_RUN=1 bash examples/launch_sft_vision_edge_platform.sh
#
# Every default below can be overridden by exporting the variable first.
# The training environment defaults to /root/.venv; override with VENV_PATH.

set -uo pipefail

# NGC/PyTorch containers can otherwise pick up incompatible host torch libraries.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_OVERRIDE:-}"

# Platform networking. Keep these overridable for clusters with different NIC names.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-50012}"

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
: "${DATASET_PATH:?Set DATASET_PATH to a visual-generation Parquet catalog}"
: "${BASE_CHECKPOINT_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Edge-DCP}"
: "${WAN_VAE_PATH:=/share/project/liyue/model_ckpts/vae/wan22_vae/Wan2.2_VAE.pth}"
: "${HF_MODEL_PATH:=/share/project/liyue/model_ckpts/Cosmos3-Edge}"
: "${OUTPUT_ROOT:=/share/project/liyue/training_log}"

# Edge's model config names the Hub repository as its processor source. Point it
# at the already-downloaded HF directory so every training rank stays offline.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$HF_MODEL_PATH" = /* ]] || HF_MODEL_PATH="$REPO_ROOT/$HF_MODEL_PATH"
VENV_PATH="${VENV_PATH:-/root/.venv}"
[[ -x "$VENV_PATH/bin/torchrun" ]] || {
    echo "ERROR: torchrun not found in $VENV_PATH; run /share/project/liyue/setup_cosmos_root_venv.sh first" >&2
    exit 1
}
source "$VENV_PATH/bin/activate"


TAIL_OVERRIDES=(
    "model.config.vlm_config.tokenizer.repository=null"
    "model.config.vlm_config.tokenizer.revision=null"
    "+model.config.vlm_config.tokenizer.tokenizer_type=$HF_MODEL_PATH"
    "model.config.load_vlm_visual=true"
)
if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    # Use a separate job name so the smoke test cannot resume or overwrite a full run.
    # Disabling torch.compile avoids paying compilation cost in a two-step validation.
    TAIL_OVERRIDES+=(
        "job.name=vision_sft_edge_smoke"
        "trainer.max_iter=2"
        "trainer.grad_accum_iter=1"
        "checkpoint.save_iter=2"
        "model.config.compile.enabled=false"
    )
    LOG_FILENAME="vision_sft_edge_smoke.log"
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
