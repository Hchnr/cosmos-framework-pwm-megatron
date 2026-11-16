#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Nano + 1B ACTION tower WAM: first-frame conditioning, video + Action128 supervision.
# Seven published R2 catalogs and their ratios live in the paired TOML.
# Local topology override: NNODES=1 DP_SHARD=8 MAX_ITER=7 WANDB_MODE=offline bash <this-script>.
# Production default: bash <this-script> on every node (16 x 8 GPUs, shard16).
# Platform supplies RANK/NODE_RANK and MASTER_ADDR. Packing uses the Action-1B production token budgets.
# Optional Hydra overrides: bash examples/launch_sft_action128_wam_nano.sh trainer.max_iter=500
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" && -f /root/.venv/bin/activate ]]; then
    source /root/.venv/bin/activate
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
TOML_FILE="${SFT_TOML_FILE:-examples/toml/sft_config/action128_wam_nano.toml}"
: "${BASE_CHECKPOINT_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano-action-1b-v022/width_only_1b}"
: "${HF_MODEL_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano-VLM}"
: "${WAN_VAE_PATH:=/share/project/liyue/model_ckpts/vae/wan22_vae/Wan2.2_VAE.pth}"
: "${NPROC_PER_NODE:=8}"
export BASE_CHECKPOINT_PATH HF_MODEL_PATH WAN_VAE_PATH NPROC_PER_NODE
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_OVERRIDE:-}"
: "${NNODES:=16}"
: "${DP_SHARD:=16}"
: "${MASTER_PORT:=29633}"
for name in NNODES NPROC_PER_NODE DP_SHARD; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $name must be positive." >&2; exit 1; }
done
ACTION128_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
ACTION128_DP_SHARD=$DP_SHARD
if (( NNODES == 1 )); then
    : "${NODE_RANK:=0}"
fi
: "${OUTPUT_ROOT:=$REPO_ROOT/outputs/action128_wam_optimized_${ACTION128_WORLD_SIZE}gpu}"
: "${LOG_FILENAME:=action128_node${NODE_RANK:-${RANK:-$(hostname)}}.log}"
mkdir -p "$OUTPUT_ROOT/logs"
ACTION128_BOOTSTRAP_LOG="$OUTPUT_ROOT/logs/action128_bootstrap_node${NODE_RANK:-${RANK:-$(hostname)}}.log"
exec > >(tee -a "$ACTION128_BOOTSTRAP_LOG") 2>&1
echo ">>> $(date -Is) Action128 launch on $(hostname); TOML=$TOML_FILE"
if (( NNODES == 1 )); then
    : "${NODE_RANK:=0}"
    : "${MASTER_ADDR:=127.0.0.1}"
    # A single-node check must not join an inherited platform rendezvous.
    unset RDZV_ENDPOINT RDZV_ID
else
    NODE_RANK="${NODE_RANK:-${RANK:-}}"
    if [[ "${ACTION128_PLATFORM_RENDEZVOUS:-0}" == "1" ]]; then
        : "${NODE_RANK:?Platform must provide RANK or NODE_RANK}"
        : "${MASTER_ADDR:?Platform must provide MASTER_ADDR}"
        if [[ -z "${JOB_ID:-}" && -f /etc/downwardapi/annotations ]]; then
            JOB_ID="$(sed -n 's/.*scheduling.k8s.io\/group-name="\([^"]*\)".*/\1/p' /etc/downwardapi/annotations | head -1)"
        fi
        : "${JOB_ID:?Set JOB_ID to the same unique job ID on every node (platform annotations unavailable)}"
        [[ "$JOB_ID" != "none" ]] || { echo "ERROR: JOB_ID must identify this job, not 'none'." >&2; exit 1; }
        # Match the working USR platform launcher, not inherited RDZV_* values
        # that can point at worker-0:23456 with the placeholder ID 'none'.
        export RDZV_BACKEND=c10d RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT" RDZV_ID="$JOB_ID"
        export RDZV_CONF="${ACTION128_RDZV_CONF:-join_timeout=7200,last_call_timeout=60}"
    fi
    if [[ -n "${RDZV_ENDPOINT:-}" ]]; then
        : "${RDZV_ID:?Set the same RDZV_ID on all nodes}"
    else
        : "${MASTER_ADDR:?Platform must provide MASTER_ADDR}"
        NODE_RANK="${NODE_RANK:-${RANK:-}}"
        : "${NODE_RANK:?Platform must provide RANK or NODE_RANK}"
    fi
fi
if (( ACTION128_WORLD_SIZE % ACTION128_DP_SHARD != 0 )); then
    echo "ERROR: world size $ACTION128_WORLD_SIZE must be divisible by shard $ACTION128_DP_SHARD." >&2
    exit 1
fi
if [[ -n "${NODE_RANK:-}" ]] && { ! [[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); }; then
    echo "ERROR: NODE_RANK must be in [0, NNODES)." >&2
    exit 1
fi
if [[ -n "${MAX_ITER:-}" ]] && ! [[ "$MAX_ITER" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_ITER must be positive." >&2
    exit 1
fi
: "${OMP_NUM_THREADS:=2}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${TORCHINDUCTOR_COMPILE_THREADS:=1}"
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT OUTPUT_ROOT LOG_FILENAME
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS TORCHINDUCTOR_COMPILE_THREADS

# The platform image need not contain a wandb login. Reuse the shared-cluster
# credential without embedding it in this script or printing it to bootstrap logs.
# An explicitly supplied WANDB_API_KEY always wins; otherwise SDK netrc login
# remains available when the optional file is absent.
: "${WANDB_API_KEY_FILE:=/share/project/eai_pwm/.secrets/cosmos_wandb_api_key}"
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_API_KEY_FILE" ]]; then
    [[ -r "$WANDB_API_KEY_FILE" ]] || { echo "ERROR: WANDB_API_KEY_FILE is not readable." >&2; exit 1; }
    WANDB_API_KEY="$(< "$WANDB_API_KEY_FILE")"
    [[ -n "$WANDB_API_KEY" ]] || { echo "ERROR: WANDB_API_KEY_FILE is empty." >&2; exit 1; }
    export WANDB_API_KEY
    echo ">>> Loaded W&B credential from WANDB_API_KEY_FILE (value hidden)."
fi
export WANDB_ENTITY="${WANDB_ENTITY:-ybainlp}"

# Fail before distributed launch if a catalog is missing. Do not duplicate the
# dataset list in the shell: the TOML is the sole source of paths and ratios.
python - "$TOML_FILE" <<'PY'
import sys
import tomllib
from pathlib import Path

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
for dataset in config["dataloader_train"]["datasets"]:
    path = Path(dataset["path"])
    if not path.is_file():
        raise SystemExit(f"ERROR: Action128 catalog not found: {path}")
    print(f"Action128 catalog: {path} (ratio={dataset['ratio']})", flush=True)
PY
TAIL_OVERRIDES=(
    "model.config.parallelism.data_parallel_shard_degree=$ACTION128_DP_SHARD"
    "model.config.parallelism.data_parallel_replicate_degree=-1"
    "trainer.seed=42"
    "job.wandb_mode=${WANDB_MODE:-online}"
)
if [[ -n "${MAX_ITER:-}" ]]; then
    TAIL_OVERRIDES+=("trainer.max_iter=$MAX_ITER")
fi
EXTRA_OVERRIDES=()
# Same whitespace-separated environment interface as the USR launcher; no eval.
# Pass overrides containing spaces as individually quoted command-line arguments.
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_TAIL_OVERRIDES:-}"
TAIL_OVERRIDES+=("${EXTRA_OVERRIDES[@]}" "$@")
echo ">>> Action128: ${NNODES}x${NPROC_PER_NODE} GPUs, shard=$ACTION128_DP_SHARD, replicas=$((ACTION128_WORLD_SIZE / ACTION128_DP_SHARD)); online VAE, fixed packing; compile settings from TOML and overrides"
echo ">>> node_rank=${NODE_RANK:-auto} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=$MASTER_PORT"
echo ">>> RDZV_ENDPOINT=${RDZV_ENDPOINT:-static} RDZV_ID=${RDZV_ID:-unset} RDZV_CONF=${RDZV_CONF:-unset}"
printf '>>> Effective tail overrides:'
printf ' %q' "${TAIL_OVERRIDES[@]}"
printf '\n'
source "$REPO_ROOT/examples/_sft_launcher_common.sh"
