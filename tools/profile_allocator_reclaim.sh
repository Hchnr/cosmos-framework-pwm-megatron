#!/usr/bin/env bash
# Allocation-only controls; each arm runs on four otherwise idle GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${NSYS_ROOT:?Set a new persistent microbenchmark output directory}"
PYTHON_BIN="${COSMOS_MEGATRON_PYTHON:-$PWD/.venv/bin/python}"
for arm in expandable native; do
    for rank in 0 1 2 3; do
        prefix="$NSYS_ROOT/$arm/rank$rank"
        [[ ! -e "$prefix.log" && ! -e "$prefix.nsys-rep" && ! -e "$prefix.json" ]] || {
            echo "Refusing to overwrite $prefix" >&2
            exit 1
        }
    done
done
for arm in expandable native; do
    value=True
    [[ "$arm" != native ]] || value=False
    mkdir -p "$NSYS_ROOT/$arm"
    pids=()
    for rank in 0 1 2 3; do
        prefix="$NSYS_ROOT/$arm/rank$rank"
        LD_LIBRARY_PATH='' PYTORCH_CUDA_ALLOC_CONF="expandable_segments:$value" \
            nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
            --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false \
            --output "$prefix" "$PYTHON_BIN" tools/benchmark_allocator_reclaim.py \
            --device "$rank" --output "$prefix.json" > "$prefix.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" == 0 ]] || exit 1
done
