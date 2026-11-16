#!/usr/bin/env bash
# Internal per-node cutover entry; arguments: NODE_RANK LAUNCH_DIR [WATCH_DIR].
set -euo pipefail
set +x
NODE_RANK="${1:?node rank required}"
LAUNCH_DIR="${2:?new launch directory required}"
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < 20 ))
WATCH="${3:-/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/outputs/mainline_bgrecv20_d39afa19_20261103}"
export LD_LIBRARY_PATH=''
mapfile -t PLAN < <(python3 - "$WATCH/restart_plan.json" <<'FIELDS'
import json,sys
p=json.load(open(sys.argv[1]))
for k in ['candidate_node_path','job_name','output_root','prepared_checkpoint','swanlab_name','master_port']:
    print(p[k])
FIELDS
)
[[ "${#PLAN[@]}" == 6 ]]
CANDIDATE="${PLAN[0]}"; RUN_NAME="${PLAN[1]}"
export OUTPUT_ROOT="${PLAN[2]}" BASE_CHECKPOINT_PATH="${PLAN[3]}"
SWANLAB_NAME="${PLAN[4]}"
JOB_PREFIX=job-d39afa19-36a7-4e01-b606-e0bf3b419b69
EXPECTED_HOST="$JOB_PREFIX-master-0"
if (( NODE_RANK > 0 )); then EXPECTED_HOST="$JOB_PREFIX-worker-$((NODE_RANK-1))"; fi
[[ "$(hostname)" == "$EXPECTED_HOST" ]]
[[ "$LAUNCH_DIR" == "$OUTPUT_ROOT"/logs/launch_* ]]
[[ -f "$BASE_CHECKPOINT_PATH/.save_complete" && -f "$BASE_CHECKPOINT_PATH/dataloader/rank_159.pkl" ]]
[[ -f "$CANDIDATE/mainline_staged.json" ]]
mkdir -p "$LAUNCH_DIR/diagnostics" "$LAUNCH_DIR/load_balance_trace" "$LAUNCH_DIR/termination"
exec 9>"$OUTPUT_ROOT/logs/.online_node${NODE_RANK}.lock"
flock -n 9 || { echo 'Managed launch already owns this node'; exit 1; }
trap 'code=$?; printf "%s\n" "$code" > "$LAUNCH_DIR/node${NODE_RANK}.exit_code"' EXIT
# Refresh NFS leaf-directory visibility before any distributed restore.
python3 - "$WATCH/checkpoint_visibility_manifest.json" "$BASE_CHECKPOINT_PATH" <<'CHECKPOINT_PREFLIGHT'
import hashlib,json,pathlib,sys,time
manifest=json.load(open(sys.argv[1])); checkpoint=pathlib.Path(sys.argv[2])
assert str(checkpoint)==manifest["checkpoint"]
for attempt in range(3):
    try:
        for component in ["model","optim","scheduler","trainer","dataloader"]:
            list((checkpoint/component).iterdir())
        for component,digest in manifest["metadata_sha256"].items():
            assert hashlib.sha256((checkpoint/component/".metadata").read_bytes()).hexdigest()==digest, component
        for relative,size in manifest["files_bytes"].items():
            path=checkpoint/relative
            assert path.stat().st_size==size, relative
            with path.open("rb") as stream:
                assert size==0 or stream.read(1), relative
        print("Checkpoint preflight OK: component leaves enumerated, metadata SHA and file readability verified", flush=True)
        break
    except OSError:
        if attempt==2:
            raise
        time.sleep(2)
CHECKPOINT_PREFLIGHT
source /share-mixed/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/activate
export LD_LIBRARY_PATH='' LD_LIBRARY_PATH_OVERRIDE=''
export NCCL_IB_HCA='=mlx5_101'
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NODE_RANK LAUNCH_DIR
export RANK="$NODE_RANK" NNODES=20 NPROC_PER_NODE=8 DP_SHARD=16 MASTER_PORT="${PLAN[5]}"
export MASTER_ADDR="$JOB_PREFIX-master-0"
export ACTION128_PLATFORM_RENDEZVOUS=1 JOB_ID="$RUN_NAME"
export SFT_TOML_FILE="$CANDIDATE/examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml"
export SFT_DIAGNOSTICS_DIR="$LAUNCH_DIR/diagnostics"
export LOG_FILENAME="action128_node${NODE_RANK}_$(basename "$LAUNCH_DIR").log"
export WANDB_API_KEY_FILE=/share-mixed/project/eai_pwm/.secrets/cosmos_wandb_api_key
export WANDB_ENTITY=ybainlp WANDB_MODE=offline
export NUMPY_MADVISE_HUGEPAGE=0 MIMALLOC_ALLOW_LARGE_OS_PAGES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export TOKENIZERS_PARALLELISM=false
export ACTION128_EPISODE_CACHE_DIR=/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1
export SLURM_LOG_DIR="$LAUNCH_DIR/termination"
export COSMOS_DATALOADER_TIMING=1
# Optional, explicit release configuration; legacy plans keep media lookahead off.
mapfile -t MEDIA_PLAN < <(python3 - "$WATCH/restart_plan.json" <<'MEDIA_FIELDS'
import json, sys
p = json.load(open(sys.argv[1])).get("media_prefetch", {})
for key, default in [("sources", ""), ("lookahead", 1024), ("total_bytes", 2147483648),
                     ("stream_bytes", 2147483648), ("minimum_pixel_work", 100000000)]:
    print(p.get(key, default))
print(json.dumps(p.get("source_bytes", {}), separators=(",", ":")))
MEDIA_FIELDS
)
[[ "${#MEDIA_PLAN[@]}" == 6 ]]
export COSMOS_ACTION128_MEDIA_PREFETCH_SOURCES="${MEDIA_PLAN[0]}"
export COSMOS_ACTION128_MEDIA_LOOKAHEAD="${MEDIA_PLAN[1]}"
export COSMOS_ACTION128_MEDIA_PREFETCH_TOTAL_BYTES="${MEDIA_PLAN[2]}"
export COSMOS_ACTION128_MEDIA_PREFETCH_BYTES="${MEDIA_PLAN[3]}"
export COSMOS_ACTION128_MEDIA_MIN_PIXEL_WORK="${MEDIA_PLAN[4]}"
export COSMOS_ACTION128_MEDIA_PREFETCH_SOURCE_BYTES="${MEDIA_PLAN[5]}"
# Camera parallelism has a separate benchmark; this rollout isolates lookahead.
unset COSMOS_ACTION128_PARALLEL_VIEW_SOURCES
printf '>>> selective media sources=%s lookahead=%s rank_output_bytes=%s source_bytes=%s\n' \
    "${MEDIA_PLAN[0]}" "${MEDIA_PLAN[1]}" "${MEDIA_PLAN[2]}" "${MEDIA_PLAN[5]}"
unset COSMOS_REQUIRE_WANDB_ONLINE IMAGINAIRE_OUTPUT_ROOT HF_MODEL_PATH WAN_VAE_PATH
unset MAX_ITER EXTRA_TAIL_OVERRIDES WANDB_RUN_ID WANDB_RESUME SFT_TRAINING_MODULE DATASET_PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
ulimit -Sn 65536
if (( NODE_RANK == 0 )); then
    /share-mixed/project/eai_pwm/home/hcr/venv/cosmos-swanlab-bridge/bin/python \
        /share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/tools/swanlab_sidecar.py \
        --output-root "$OUTPUT_ROOT" --name "$SWANLAB_NAME" \
        --step-log "$LAUNCH_DIR/node0.log" --background 9>&-
fi
cd "$CANDIDATE"
echo ">>> online release $CANDIDATE; W16/K8; full-state resume"
sha256sum cosmos_framework/data/generator/dataflow/shared_prefetch.py
bash examples/launch_pretrain_state_h1_96gpu.sh \
    checkpoint.action128_load_mode=auto \
    job.name="$RUN_NAME" job.wandb_mode=offline \
    trainer.callbacks.load_balance_trace.enabled=true \
    trainer.callbacks.load_balance_trace.flush_every_n_batches=10 \
    trainer.callbacks.load_balance_trace.max_samples_per_record=92 \
    trainer.callbacks.load_balance_trace.output_dir="$LAUNCH_DIR/load_balance_trace"
