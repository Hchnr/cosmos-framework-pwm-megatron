#!/usr/bin/env bash
# Paired full-model replay benchmark; excludes live loading and checkpoint I/O.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${BASE_CHECKPOINT_PATH:?Set the original trained checkpoint path}"
: "${REAL_BATCH_PATH:?Set the materialized real batch path}"
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
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" \
        tests/megatron/train_real_batch_smoke.py \
        --batch "$REAL_BATCH_PATH" --checkpoint "$BASE_CHECKPOINT_PATH" \
        --output-root "outputs/megatron_validation/${BENCHMARK_TAG}_${run}_${backend}" \
        --backend "$backend" --sharding-strategy "${MEGATRON_SHARDING_STRATEGY:-optim_grads}" \
        --steps "${BENCHMARK_STEPS:-20}" --warmup-steps "${BENCHMARK_WARMUP_STEPS:-3}" \
        --benchmark-json "$result.json" > "$result.log" 2>&1
    echo "Completed $backend run $run: $result.json"
done
