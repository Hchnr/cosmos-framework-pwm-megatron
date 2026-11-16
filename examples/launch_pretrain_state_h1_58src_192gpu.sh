#!/usr/bin/env bash
# 58-source continuation using the existing 24-node production launch chain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SFT_TOML_FILE="${SFT_TOML_FILE:-$PWD/examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/pretrain58src_state_h1_nano_192gpu_ga3_run01}"
# Large NumPy video buffers can synchronously compact fragmented NUMA memory.
# Change allocation advice for this process tree, leaving kernel THP policy alone.
export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
# Arrow's mimalloc pool also requests huge pages independently of NumPy.
# Keep the allocator, but disable its large-page advice for future launches.
export MIMALLOC_ALLOW_LARGE_OS_PAGES="${MIMALLOC_ALLOW_LARGE_OS_PAGES:-0}"

# Shared across nodes, restarts and job/output names. Empty explicitly disables caching.
export ACTION128_EPISODE_CACHE_DIR="${ACTION128_EPISODE_CACHE_DIR-/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1}"

SOURCE_RUN="${ACTION128_SOURCE_RUN:-$PWD/outputs/pretrain52src_state_h1_nano_192gpu_ga3_from17200_run01/cosmos3_action/action128/action128_wam_nano_52src_state_h1}"
if [[ -z "${BASE_CHECKPOINT_PATH:-}" ]]; then
    LATEST_FILE="$SOURCE_RUN/checkpoints/latest_checkpoint.txt"
    [[ -r "$LATEST_FILE" ]] || { echo "ERROR: source checkpoint pointer missing: $LATEST_FILE" >&2; exit 1; }
    LATEST_CHECKPOINT="$(< "$LATEST_FILE")"
    [[ "$LATEST_CHECKPOINT" =~ ^iter_[0-9]{9}$ ]] || { echo "ERROR: invalid source checkpoint pointer" >&2; exit 1; }
    export BASE_CHECKPOINT_PATH="$SOURCE_RUN/checkpoints/$LATEST_CHECKPOINT"
fi
[[ -f "$BASE_CHECKPOINT_PATH/.save_complete" ]] || {
    echo "ERROR: continuation requires a completed checkpoint: $BASE_CHECKPOINT_PATH" >&2
    exit 1
}
for component in model optim scheduler trainer; do
    [[ -f "$BASE_CHECKPOINT_PATH/$component/.metadata" ]] || {
        echo "ERROR: source checkpoint missing $component metadata" >&2
        exit 1
    }
done
[[ -f "$BASE_CHECKPOINT_PATH/dataloader/rank_0.pkl" ]] || {
    echo "ERROR: source checkpoint missing dataloader state" >&2
    exit 1
}
export LD_LIBRARY_PATH=''
python -c 'import mcap, mcap_protobuf, zstandard, google_crc32c; print("Native Ego media dependencies OK")'
echo ">>> 58-source full-state continuation from: $BASE_CHECKPOINT_PATH"
echo ">>> NumPy MADV_HUGEPAGE: $NUMPY_MADVISE_HUGEPAGE"
echo ">>> mimalloc large OS pages: $MIMALLOC_ALLOW_LARGE_OS_PAGES"
# Run outside torchrun and before local staging. All nodes share the manager
# lock; an existing tools-node manager is reused without touching training.
if [[ "${SWANLAB_SIDECAR_ENABLE:-1}" == "1" ]]; then
    SWANLAB_PYTHON="${SWANLAB_PYTHON:-/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-swanlab-bridge/bin/python}"
    if ! "$SWANLAB_PYTHON" "$PWD/tools/swanlab_sidecar.py" --output-root "$OUTPUT_ROOT" --background; then
        echo "WARNING: SwanLab sidecar startup failed; see docs/swanlab_sidecar.md" >&2
    fi
fi
# Use the production recipe with local staging, rendezvous and batch traces.
exec bash examples/launch_pretrain_state_h1_52src_192gpu.sh "$@"
