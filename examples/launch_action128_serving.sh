#!/usr/bin/env bash
set -euo pipefail

# Independent from training NNODES/NPROC_PER_NODE; defaults to one serving GPU.
SERVING_GPUS=${SERVING_GPUS:-1}
SERVING_NNODES=${SERVING_NNODES:-1}
SERVING_NODE_RANK=${SERVING_NODE_RANK:-0}
SERVING_MASTER_ADDR=${SERVING_MASTER_ADDR:-127.0.0.1}
SERVING_MASTER_PORT=${SERVING_MASTER_PORT:-29571}
SERVING_CHECKPOINT_SOURCE=${CHECKPOINT_PATH:-${CHECKPOINT_ROOT:-}}
: "${SERVING_CHECKPOINT_SOURCE:?Set CHECKPOINT_PATH to an explicit iter/model; CHECKPOINT_ROOT can select latest}"
if ! [[ "$SERVING_GPUS" =~ ^[1-9][0-9]*$ && "$SERVING_NNODES" =~ ^[1-9][0-9]*$ ]]; then
    echo 'SERVING_GPUS and SERVING_NNODES must be positive integers' >&2
    exit 2
fi
export LD_LIBRARY_PATH=''
export COSMOS_TRAINING=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# torchrun ignores stale training environment ranks via explicit arguments.
exec torchrun --nnodes="$SERVING_NNODES" --nproc-per-node="$SERVING_GPUS" \
    --node-rank="$SERVING_NODE_RANK" --master-addr="$SERVING_MASTER_ADDR" \
    --master-port="$SERVING_MASTER_PORT" \
    -m cosmos_framework.scripts.action128_server \
    --checkpoint-path "$SERVING_CHECKPOINT_SOURCE" \
    --output-dir "${SERVING_OUTPUT_DIR:-outputs/action_serving}" \
    --host "${SERVING_HOST:-0.0.0.0}" --port "${SERVING_PORT:-8000}" "$@"
