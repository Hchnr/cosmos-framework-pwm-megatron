#!/usr/bin/env bash
# Production action tower: scheme D, common entry for local and multi-node jobs.
set -euo pipefail
TOML_FILE="examples/toml/sft_config/action_policy_1b.toml"
: "${NNODES:=1}"
: "${NPROC_PER_NODE:=8}"
: "${GLOBAL_BATCH_SIZE:=736}"
: "${ACTION_SAMPLES_PER_PACK:=92}"
: "${MASTER_PORT:=29631}"
for name in NNODES NPROC_PER_NODE GLOBAL_BATCH_SIZE ACTION_SAMPLES_PER_PACK; do
    value="${!name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $name must be a positive integer, got '$value'." >&2
        exit 1
    fi
done
ACTION_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
ACTION_MICROBATCH_SIZE=$((ACTION_WORLD_SIZE * ACTION_SAMPLES_PER_PACK))
if (( GLOBAL_BATCH_SIZE % ACTION_MICROBATCH_SIZE != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be a multiple of GPUs * samples per pack = $ACTION_MICROBATCH_SIZE." >&2
    exit 1
fi
ACTION_GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / ACTION_MICROBATCH_SIZE))
if [[ -n "${RDZV_ENDPOINT:-}" ]]; then
    : "${RDZV_ID:?Set the same unique RDZV_ID on all nodes}"
elif (( NNODES > 1 )); then
    : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host address on all nodes}"
    : "${NODE_RANK:?Set NODE_RANK explicitly on each node}"
fi
if [[ -n "${NODE_RANK:-}" ]] && { ! [[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); }; then
    echo "ERROR: NODE_RANK must be in [0, NNODES)." >&2
    exit 1
fi
: "${DATASET_PATH:?Set DATASET_PATH to prepared offline.parquet; see docs/action_policy_1b.md}"
: "${BASE_CHECKPOINT_PATH:=/share/project/eai_pwm/models/Cosmos3-Nano-action-1b-v022/width_only_1b}"
: "${WAN_VAE_PATH:=/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth}"
# The offline recipe does not load VAE weights in the training process.
: "${REQUIRE_WAN_VAE:=0}"
: "${HF_MODEL_PATH:=/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct}"
: "${ACTION_EVAL_FIXTURE:=$(dirname "$DATASET_PATH")/flow_eval_offline.pt}"
: "${ACTION_EVAL_EVERY:=500}"
if ! [[ "$ACTION_EVAL_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ACTION_EVAL_EVERY must be a positive integer." >&2
    exit 1
fi
[[ -f "$ACTION_EVAL_FIXTURE" ]] || { echo "ERROR: Missing evaluation fixture: $ACTION_EVAL_FIXTURE" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] || {
    echo "ERROR: BASE_CHECKPOINT_PATH must be a converted width_only_1b checkpoint root containing model/.metadata." >&2
    exit 1
}
: "${LOG_FILENAME:=action_policy_1b_node${NODE_RANK:-$(hostname)}_sft.log}"
: "${OUTPUT_ROOT:=outputs/action_policy_1b}"
: "${TORCHINDUCTOR_COMPILE_THREADS:=1}"
export OUTPUT_ROOT TORCHINDUCTOR_COMPILE_THREADS
: "${OMP_NUM_THREADS:=2}"
: "${OPENBLAS_NUM_THREADS:=1}"
export NNODES NPROC_PER_NODE MASTER_PORT DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH HF_MODEL_PATH LOG_FILENAME
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS
export LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
export COSMOS_REQUIRE_WANDB_ONLINE=1
TAIL_OVERRIDES=(
    "trainer.seed=42"
    "model.config.parallelism.data_parallel_shard_degree=$ACTION_WORLD_SIZE"
    "trainer.grad_accum_iter=$ACTION_GRAD_ACCUM"
    "dataloader_train.max_samples_per_batch=$ACTION_SAMPLES_PER_PACK"
    "++trainer.callbacks.small_action_audit._target_=cosmos_framework.callbacks.small_action_audit.SmallActionAudit"
    "++trainer.callbacks.small_action_audit.expected_global_batch_size=$GLOBAL_BATCH_SIZE"
    "++trainer.callbacks.fixed_action_validation._target_=cosmos_framework.callbacks.fixed_action_validation.FixedActionValidation"
    "++trainer.callbacks.fixed_action_validation.fixture_path=$ACTION_EVAL_FIXTURE"
    "++trainer.callbacks.fixed_action_validation.every_n=$ACTION_EVAL_EVERY"
    "$@"
)
echo ">>> Action tower D: ${NNODES}x${NPROC_PER_NODE} GPUs; dp_shard=$ACTION_WORLD_SIZE; samples/pack=$ACTION_SAMPLES_PER_PACK; accumulation=$ACTION_GRAD_ACCUM; global batch=$GLOBAL_BATCH_SIZE"
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
