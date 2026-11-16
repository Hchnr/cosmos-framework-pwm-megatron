#!/usr/bin/env bash
# 12 x 8 H100: 23-source Action128 WAM, 1B ACTION tower, shard16/replica6.
# Run on every node with the same MASTER_ADDR and distinct NODE_RANK=0..11
# (platform node RANK is also accepted). JOB_ID or platform annotations identify
# the rendezvous; inherited platform RDZV_ENDPOINT/RDZV_ID are replaced.
# Use a distinct SHARED OUTPUT_ROOT for each independent experiment. VAE AOT
# packages are built across ranks and loaded by every node from this directory.
# Load trained Action128 model weights into a new run with fresh training state.
# !!! TEMPORARY: default recipe pins video30; this does NOT fix the source timebase.
set -euo pipefail

ACTION128_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NNODES:=12}"
: "${NPROC_PER_NODE:=8}"
: "${DP_SHARD:=16}"
: "${SFT_TOML_FILE:=$ACTION128_LAUNCH_DIR/toml/sft_config/action128_wam_nano_160gpu.toml}"
: "${OUTPUT_ROOT:=$ACTION128_LAUNCH_DIR/../outputs/action128_wam_23src_20261015_warm_start_from22src_i8800_run01}"
# warm_start requires trained Action128 weights and their source contract metadata.
# Completed source checkpoint: latest_checkpoint.txt read on 2026-10-15.
# Its weights came from the 12-source e2aux345 run at iter_000006200.
: "${BASE_CHECKPOINT_PATH:=$ACTION128_LAUNCH_DIR/../outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008800}"
export BASE_CHECKPOINT_PATH
: "${MASTER_PORT:=50012}"
export ACTION128_PLATFORM_RENDEZVOUS=1
export MASTER_PORT
export NNODES NPROC_PER_NODE DP_SHARD SFT_TOML_FILE OUTPUT_ROOT
# Keep diagnostics outside training workers so a stalled or killed trainer
# leaves node memory counters and the torchrun exit status on shared storage.
export SFT_DIAGNOSTICS_DIR="${SFT_DIAGNOSTICS_DIR:-$OUTPUT_ROOT/diagnostics}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# Large video/metadata allocations across data workers can stall in
# synchronous transparent-hugepage compaction. The Linux process flag survives
# fork/exec, covering torchrun, data workers and decoder helpers without changing
# host settings. Set ACTION128_DISABLE_THP=0 to retain the host's THP policy.
if [[ "${ACTION128_DISABLE_THP:-1}" == "1" ]]; then
    exec python - "$ACTION128_LAUNCH_DIR/launch_sft_action128_wam_nano.sh" "$@" <<'PY'
import ctypes
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
print(
    ">>> Disabled transparent hugepages for this Action128 job (ACTION128_DISABLE_THP=1)",
    flush=True,
)
os.execv("/bin/bash", ["bash", *sys.argv[1:]])
PY
fi

exec bash "$ACTION128_LAUNCH_DIR/launch_sft_action128_wam_nano.sh" "$@"
