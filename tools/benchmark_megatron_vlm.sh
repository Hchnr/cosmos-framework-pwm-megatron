#!/usr/bin/env bash
# Pretrained VLM compute benchmark; fixed synthetic vision/text, no live loader.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${VLM_MODEL_PATH:?Set a local Qwen3-VL checkpoint directory}"
export LD_LIBRARY_PATH=''
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${COSMOS_MEGATRON_PYTHON:-$PWD/.venv/bin/python}"
BENCHMARK_TAG="${BENCHMARK_TAG:-vlm_2b_1024}"
RESULT_DIR="${RESULT_DIR:-$PWD/artifacts/megatron}"
mkdir -p "$RESULT_DIR"
backends=(pytorch megatron megatron pytorch)
for index in "${!backends[@]}"; do
    backend="${backends[$index]}"
    run="$((index + 1))"
    result="$RESULT_DIR/${BENCHMARK_TAG}_${run}_${backend}"
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" \
        tests/megatron/benchmark_vlm.py \
        --model-path "$VLM_MODEL_PATH" --backend "$backend" --output "$result.json" \
        --replicas "${DP_REPLICAS:-2}" --tokens "${BENCHMARK_TOKENS:-1024}" \
        --steps "${BENCHMARK_STEPS:-35}" --warmup "${BENCHMARK_WARMUP_STEPS:-5}" \
        --sharding-strategy "${MEGATRON_SHARDING_STRATEGY:-optim_grads}" > "$result.log" 2>&1
    echo "Completed $backend run $run: $result.json"
done
