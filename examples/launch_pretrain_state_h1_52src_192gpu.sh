#!/usr/bin/env bash
# 52-source State128 pretraining on 24 eight-GPU nodes.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
# Packed batches and dataloader IPC can exceed the container's 1024-fd default.
if [[ "$(ulimit -Sn)" != "unlimited" ]] && (( $(ulimit -Sn) < 65536 )); then
    ulimit -Sn 65536
fi
echo ">>> Open-file soft limit: $(ulimit -Sn)"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SFT_TOML_FILE="${SFT_TOML_FILE:-$PWD/examples/toml/sft_config/action128_pretrain_state_h1_52src_192gpu.toml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/pretrain52src_state_h1_nano_192gpu_ga3_from17200_run01}"
export SFT_DIAGNOSTICS_DIR="${SFT_DIAGNOSTICS_DIR:-$OUTPUT_ROOT/logs/diagnostics}"
export NNODES="${NNODES:-24}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}" DP_SHARD="${DP_SHARD:-16}"

SHARED_REPO="$PWD"
LOCAL_REPO="$(bash "$SHARED_REPO/examples/_stage_training_code.sh" "$SHARED_REPO")"
echo ">>> Node-local training code: $LOCAL_REPO; shared source: $SHARED_REPO"
echo ">>> Staged dataloader source fingerprints:"
sha256sum \
    "$LOCAL_REPO/cosmos_framework/data/generator/dataflow/shared_prefetch.py" \
    "$LOCAL_REPO/cosmos_framework/data/generator/action/unified_action128.py" \
    "$LOCAL_REPO/cosmos_framework/data/generator/media_decode.py" \
    "$LOCAL_REPO/cosmos_framework/checkpoint/dcp.py"
cd "$LOCAL_REPO"
# Keep per-rank sample identities and phase timings for straggler diagnosis.
# Buffer ten microbatches per append; CUDA events are queried without adding a
# training-time synchronize. Explicit caller overrides remain last.
exec bash examples/launch_pretrain_state_h1_96gpu.sh \
    trainer.callbacks.load_balance_trace.enabled=true \
    trainer.callbacks.load_balance_trace.flush_every_n_batches=10 \
    trainer.callbacks.load_balance_trace.max_samples_per_record=92 \
    "$@"
