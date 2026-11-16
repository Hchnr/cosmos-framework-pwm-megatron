#!/usr/bin/env bash
# Original Nano two-tower baseline, aligned with the online three-tower recipe.
# Reuse its topology, THP policy, credential handling and shared AOT setup.
# Set NNODES identically for both experiments (e.g. NNODES=18 for 144 GPUs).
set -euo pipefail

ACTION128_TWO_TOWER_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SFT_TOML_FILE:=$ACTION128_TWO_TOWER_LAUNCH_DIR/toml/sft_config/action128_wam_nano_2tower_160gpu.toml}"
# DCP root, not model/: the checkpoint loader appends model/ itself.
: "${BASE_CHECKPOINT_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano-DCP}"
: "${OUTPUT_ROOT:=$ACTION128_TWO_TOWER_LAUNCH_DIR/../outputs/action128_wam_2tower_160gpu_static_vae_aot_no_robomind2_run01}"
: "${ACTION128_COMPILE_CACHE_ROOT:=/tmp/cosmos_compile_cache/action128_wam_2tower_160gpu}"
: "${MASTER_PORT:=50013}"
export SFT_TOML_FILE BASE_CHECKPOINT_PATH OUTPUT_ROOT ACTION128_COMPILE_CACHE_ROOT MASTER_PORT

exec bash "$ACTION128_TWO_TOWER_LAUNCH_DIR/launch_sft_action128_wam_nano_160gpu.sh" "$@"
