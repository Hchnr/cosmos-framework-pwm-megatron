# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Shared launch plumbing for examples/launch_sft_<recipe>.sh — the
# structured-TOML / pydantic-schema flow that drives cosmos_framework.scripts.train.
#
# Caller MUST set before sourcing:
#   TOML_FILE            recipe TOML, e.g. "examples/toml/sft_config/<recipe>.toml".
#                        Absolute or repo-root-relative.
#
# Caller MAY set before sourcing (presence drives which existence checks fire):
#   DATASET_PATH         Parquet catalog file or directory.
#   BASE_CHECKPOINT_PATH recipe-local base DCP dir, e.g. "examples/checkpoints/<name>".
#   WAN_VAE_PATH         override the default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth.
#   HF_MODEL_PATH        local HF snapshot used for tokenizer / multimodal processor files.
#   REQUIRE_WAN_VAE      1 (default) for Generation/Action; 0 for pure Understanding.
#                        Offline-VAE Generation launchers also set this to 0.
#   EXTRA_DATASET_CHECK  bash snippet (string) eval'd after the default checks.
#   TAIL_OVERRIDES       bash array of Hydra CLI overrides appended after `--`
#                        (e.g. data_setting.max_tokens=16000 for VLM smokes).
#   MASTER_PORT          torchrun --master_port; default 50012.
#   SFT_TRAINING_MODULE  Optional diagnostic entry; defaults to the production train module.
#   NPROC_PER_NODE       torchrun --nproc_per_node; default 8.
#   NNODES               torchrun --nnodes; multi-node only (unset = single-node).
#   NODE_RANK            torchrun --node_rank; this worker's 0-based index.
#   MASTER_ADDR          torchrun --master_addr; rank-0 host (multi-node only — it
#                        has no torchrun env fallback, so it must be passed here).
#   RDZV_ENDPOINT        when set, use elastic c10d rendezvous instead of the
#                        static MASTER_ADDR/NODE_RANK launch mode.
#   RDZV_BACKEND/ID/CONF optional elastic rendezvous settings.
#   WANDB_API_KEY        W&B API key consumed by the SDK. Persistent CLI login is
#                        opt-in through WANDB_PERSIST_LOGIN=1.
#   LOG_FILENAME         override $LOG_DIR/${LOG_FILENAME}
#                        (default <toml-stem>_sft.log).
#   SFT_DIAGNOSTICS_DIR   Optional per-node memory/exit-status JSONL directory.
#
# Absolute paths are passed through; relative paths are anchored to the repo
# root (the parent of this examples/ directory). Paths set in the caller's
# shell via `export DATASET_PATH=...` etc. win over the launcher's defaults
# (use the `: "${VAR:=default}"` idiom in the launcher to preserve this).

set -uo pipefail

: "${TOML_FILE:?TOML_FILE must be set before sourcing _sft_launcher_common.sh}"

# Repo root = parent of the wrapper's directory (examples/).
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"

# Some hosts expose a forward-compatible CUDA driver stack while also
# inheriting system cuBLAS paths. Recipes that opt in must prefer the matching
# cuBLAS/cuBLASLt pair installed in this checkout's venv.
if [[ "${PREFER_PROJECT_CUDA_LIBS:-0}" == "1" ]]; then
    for PROJECT_CUDA_LIB_DIR in "$WORKDIR"/.venv/lib/python*/site-packages/nvidia/cu13/lib; do
        if [[ -d "$PROJECT_CUDA_LIB_DIR" ]]; then
            export LD_LIBRARY_PATH="$PROJECT_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            break
        fi
    done
fi

# Anchor relative paths to $WORKDIR.
[[ "$TOML_FILE" = /* ]] || TOML_FILE="$WORKDIR/$TOML_FILE"

if [[ -n "${DATASET_PATH:-}" ]]; then
    [[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$WORKDIR/$DATASET_PATH"
    export DATASET_PATH
fi

if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    [[ "$BASE_CHECKPOINT_PATH" = /* ]] || BASE_CHECKPOINT_PATH="$WORKDIR/$BASE_CHECKPOINT_PATH"
    export BASE_CHECKPOINT_PATH
fi

if [[ -n "${HF_MODEL_PATH:-}" ]]; then
    [[ "$HF_MODEL_PATH" = /* ]] || HF_MODEL_PATH="$WORKDIR/$HF_MODEL_PATH"
    export HF_MODEL_PATH
fi

REQUIRE_WAN_VAE="${REQUIRE_WAN_VAE:-1}"
if [[ "$REQUIRE_WAN_VAE" == "1" ]]; then
    WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
    [[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"
    export WAN_VAE_PATH
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/train}"
LOG_DIR="$OUTPUT_ROOT/logs"
TOML_STEM="$(basename "$TOML_FILE" .toml)"
LOG_FILE="$LOG_DIR/${LOG_FILENAME:-${TOML_STEM}_sft.log}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -f "$TOML_FILE" ]] || { echo "ERROR: TOML not found: $TOML_FILE" >&2; exit 1; }
if [[ -n "${DATASET_PATH:-}" ]]; then
    [[ -f "$DATASET_PATH" || -d "$DATASET_PATH" ]] || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
    if [[ -f "$DATASET_PATH" ]]; then
        [[ "$DATASET_PATH" == *.parquet ]] || { echo "ERROR: DATASET_PATH file must end in .parquet: $DATASET_PATH" >&2; exit 1; }
    else
        [[ -n "$(find "$DATASET_PATH" -name '*.parquet' -print -quit)" ]] || { echo "ERROR: DATASET_PATH directory contains no Parquet files: $DATASET_PATH" >&2; exit 1; }
    fi
fi
if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    [[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH (run Step 2 of docs/training.md, or export BASE_CHECKPOINT_PATH=<path>)" >&2; exit 1; }
fi
if [[ -n "${HF_MODEL_PATH:-}" ]]; then
    [[ -d "$HF_MODEL_PATH" ]] || { echo "ERROR: HF_MODEL_PATH not found: $HF_MODEL_PATH" >&2; exit 1; }
fi
if [[ "$REQUIRE_WAN_VAE" == "1" ]]; then
    [[ -f "$WAN_VAE_PATH" ]]         || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH (run Step 1 of docs/training.md, or export WAN_VAE_PATH=<path>)" >&2; exit 1; }
fi
if [[ -n "${EXTRA_DATASET_CHECK:-}" ]]; then eval "$EXTRA_DATASET_CHECK"; fi

# The SDK reads WANDB_API_KEY directly. Avoid an eager network login on every
# node: it runs before the configured W&B mode is applied and can otherwise
# prevent an offline torchrun from starting on clusters without external DNS. Set
# WANDB_PERSIST_LOGIN=1 only when a persistent ~/.netrc entry is explicitly
# desired.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    python -c 'import wandb' >/dev/null 2>&1 || {
        echo "ERROR: WANDB_API_KEY is set, but wandb is not available in the active interpreter." >&2
        exit 1
    }
    if [[ "${WANDB_PERSIST_LOGIN:-0}" == "1" ]]; then
        echo ">>> $(date '+%H:%M:%S') Persisting W&B credentials..."
        python -m wandb login --relogin "$WANDB_API_KEY" >/dev/null || {
            echo "ERROR: wandb login failed." >&2
            exit 1
        }
    else
        echo ">>> $(date '+%H:%M:%S') Using WANDB_API_KEY from the job environment (no eager network login)."
    fi
fi

# W&B 0.25.0 can receive a null viewer `flags` field from the service and crash
# while formatting the optional "logged in as" message. Authentication and run
# logging do not require that message; silent mode bypasses only console chatter.
export WANDB_SILENT="${WANDB_SILENT:-true}"

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
[[ -n "${DATASET_PATH:-}" ]]         && echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
[[ -n "${BASE_CHECKPOINT_PATH:-}" ]] && echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

# Default empty if caller didn't set; safe under set -u.
[[ ${TAIL_OVERRIDES+x} ]] || TAIL_OVERRIDES=()

TRAILING_ARGS=()
if (( ${#TAIL_OVERRIDES[@]} > 0 )); then
    TRAILING_ARGS=(-- "${TAIL_OVERRIDES[@]}")
fi

# torchrun topology. Elastic rendezvous is preferred by the multi-node platform
# launcher because it does not require translating platform RANK into NODE_RANK.
TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}")
if [[ -n "${RDZV_ENDPOINT:-}" ]]; then
    TORCHRUN_ARGS+=(
        --nnodes="${NNODES:?NNODES is required with RDZV_ENDPOINT}"
        --rdzv-backend="${RDZV_BACKEND:-c10d}"
        --rdzv-endpoint="$RDZV_ENDPOINT"
        --rdzv-id="${RDZV_ID:?RDZV_ID is required with RDZV_ENDPOINT}"
        --rdzv-conf="${RDZV_CONF:-join_timeout=7200,last_call_timeout=60}"
    )
else
    TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
    [[ -n "${NNODES:-}" ]]      && TORCHRUN_ARGS+=(--nnodes="$NNODES")
    [[ -n "${NODE_RANK:-}" ]]   && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
    [[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")
fi

# Use the active interpreter: dependency-sharing venvs may not own a torchrun
# console script, and an inherited script can silently select another PyTorch.
TRAIN_COMMAND=(python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" -m "${SFT_TRAINING_MODULE:-cosmos_framework.scripts.train}"
    --sft-toml="$TOML_FILE" "${TRAILING_ARGS[@]}")
if [[ -n "${SFT_DIAGNOSTICS_DIR:-}" ]]; then
    TRAIN_COMMAND=(python -m cosmos_framework.tools.launch_diagnostics
        --output-dir="$SFT_DIAGNOSTICS_DIR" -- "${TRAIN_COMMAND[@]}")
fi
# Callers use set -e: a failed pipeline must reach PIPESTATUS capture and the
# final exit message instead of terminating the shell before either is written.
if IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" PYTHONPATH=. \
    "${TRAIN_COMMAND[@]}" 2>&1 | tee "$LOG_FILE"; then
    EXIT_CODE=0
else
    EXIT_CODE=${PIPESTATUS[0]}
fi
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
