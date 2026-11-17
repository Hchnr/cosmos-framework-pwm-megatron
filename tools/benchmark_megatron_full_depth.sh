#!/usr/bin/env bash
# Paired full-model benchmark; opt-in live loading, excludes checkpoint save I/O.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${BASE_CHECKPOINT_PATH:?Set the original trained checkpoint path}"
BENCHMARK_LIVE_DATA="${BENCHMARK_LIVE_DATA:-0}"
case "$BENCHMARK_LIVE_DATA" in
    0) : "${REAL_BATCH_PATH:?Set the materialized real batch path}" ;;
    1) REAL_BATCH_PATH="${REAL_BATCH_PATH:-$PWD/outputs/megatron_validation/real_batch.pt}" ;;
    *) echo "ERROR: BENCHMARK_LIVE_DATA must be 0 or 1" >&2; exit 1 ;;
esac
export LD_LIBRARY_PATH=''
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false NUMPY_MADVISE_HUGEPAGE=0 MIMALLOC_ALLOW_LARGE_OS_PAGES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${COSMOS_MEGATRON_PYTHON:-$PWD/.venv/bin/python}"
BENCHMARK_TAG="${BENCHMARK_TAG:-full_depth_replay}"
RESULT_DIR="${RESULT_DIR:-$PWD/artifacts/megatron}"
mkdir -p "$RESULT_DIR"
backends=(pytorch megatron megatron pytorch)
for index in "${!backends[@]}"; do
    backend="${backends[$index]}"
    run="$((index + 1))"
    result="$RESULT_DIR/${BENCHMARK_TAG}_${run}_${backend}"
    extra_args=()
    if [[ "$BENCHMARK_LIVE_DATA" == "1" ]]; then
        extra_args+=(--live-data --trace-json "${result}_inputs.json")
    fi
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" \
        tests/megatron/train_real_batch_smoke.py \
        --batch "$REAL_BATCH_PATH" --checkpoint "$BASE_CHECKPOINT_PATH" \
        --output-root "outputs/megatron_validation/${BENCHMARK_TAG}_${run}_${backend}" \
        --backend "$backend" --sharding-strategy "${MEGATRON_SHARDING_STRATEGY:-optim_grads}" \
        --steps "${BENCHMARK_STEPS:-20}" --warmup-steps "${BENCHMARK_WARMUP_STEPS:-3}" \
        --und-tokens "${BENCHMARK_UND_TOKENS:-512}" --gen-tokens "${BENCHMARK_GEN_TOKENS:-8192}" \
        --action-tokens "${BENCHMARK_ACTION_TOKENS:-128}" \
        --benchmark-json "$result.json" "${extra_args[@]}" > "$result.log" 2>&1
    echo "Completed $backend run $run: $result.json"
done
