#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/share/project/liyue/model_ckpts/Cosmos3-Edge}"
VAE_PATH="${VAE_PATH:-/share/project/liyue/model_ckpts/vae/wan22_vae/Wan2.2_VAE.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/inference_edge}"
SEED="${SEED:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH=""

source /root/.venv/bin/activate
cd "${REPO_ROOT}"

python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i \
  inputs/omni/t2i.json \
  inputs/omni/t2v.json \
  inputs/omni/i2v.json \
  inputs/omni/v2v.json \
  -o "${OUTPUT_DIR}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --experiment-overrides \
  "model.config.tokenizer.bucket_name=" \
  "model.config.tokenizer.vae_path=${VAE_PATH}" \
  --seed="${SEED}" \
  --no-guardrails \
  --keep-going \
  "$@"
