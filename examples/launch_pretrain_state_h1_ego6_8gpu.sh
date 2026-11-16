#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SFT_TOML_FILE="${SFT_TOML_FILE:-examples/toml/sft_config/action128_pretrain_state_h1_ego6_8gpu.toml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/share-mixed/projset/liyue/model_ckpts/pretrain_state_h1_ego6_8gpu_20261030}"
export MASTER_PORT="${MASTER_PORT:-29656}"
exec bash examples/launch_pretrain_state_h1_52src_8gpu.sh "$@"
