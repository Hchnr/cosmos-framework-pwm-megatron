#!/usr/bin/env bash
# Attribute the four-H100 production-capacity FSDP2/Megatron-ZeRO3 comparison.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${BASE_CHECKPOINT_PATH:?Set the original iter_000001200 checkpoint}"
: "${NSYS_ROOT:?Set a new persistent capture directory}"
export LD_LIBRARY_PATH='' PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false
export NUMPY_MADVISE_HUGEPAGE=0 MIMALLOC_ALLOW_LARGE_OS_PAGES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="${COSMOS_MEGATRON_PYTHON:-$PWD/.venv/bin/python}"
export NSYS_CAPTURE="${NSYS_CAPTURE:-1}"
case "$NSYS_CAPTURE:${PROFILE_PARAM_PREFETCH:-1}" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "ERROR: NSYS_CAPTURE and PROFILE_PARAM_PREFETCH must be 0 or 1" >&2; exit 1 ;;
esac
for backend in ${PROFILE_BACKENDS:-pytorch megatron}; do
    case "$backend" in
        pytorch|megatron) ;;
        *) echo "ERROR: unsupported backend: $backend" >&2; exit 1 ;;
    esac
    export NSYS_RUN_DIR="$NSYS_ROOT/$backend"
    [[ ! -e "$NSYS_RUN_DIR/launch.log" ]] || {
        echo "ERROR: refusing to overwrite an existing run: $NSYS_RUN_DIR" >&2
        exit 1
    }
    mkdir -p "$NSYS_RUN_DIR"
    profile_args=()
    if [[ "$NSYS_CAPTURE" == "1" ]]; then
        profile_args+=(--profile-start-iteration 1203 --profile-updates 2)
    fi
    if [[ "${PROFILE_PARAM_PREFETCH:-1}" == "0" ]]; then
        profile_args+=(--disable-param-prefetch)
    fi
    if [[ "${PROFILE_PYTHON_GC:-0}" == "1" ]]; then
        profile_args+=(--profile-python-gc)
    fi
    if [[ "${PROFILE_DISABLE_AUTO_GC:-0}" == "1" ]]; then
        profile_args+=(--disable-auto-gc-after-warmup)
    fi
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
        tools/profile_megatron_fsdp_rank.py \
        --backend "$backend" --sharding-strategy optim_grads_params \
        --batch outputs/megatron_validation/real_batch.pt --checkpoint "$BASE_CHECKPOINT_PATH" \
        --output-root "$NSYS_RUN_DIR/training" --steps 1205 --warmup-steps 3 \
        --live-data --production-capacities --production-loader \
        --resume-training --checkpoint-load-mode extend --preserve-source-normalization \
        --state-normalization-paths examples/normalization/state_h1_source_paths.json \
        --expected-start-iteration 1200 --benchmark-json "$NSYS_RUN_DIR/measurement.json" \
        --trace-json "$NSYS_RUN_DIR/inputs.json" "${profile_args[@]}" \
        > "$NSYS_RUN_DIR/launch.log" 2>&1
    echo "Completed: $NSYS_RUN_DIR"
done
