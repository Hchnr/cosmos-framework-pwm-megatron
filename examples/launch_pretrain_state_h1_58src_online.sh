#!/usr/bin/env bash
# Production 58-source State-H1 continuation with environment setup and SwanLab.
# Run on every platform container; see examples/README_pretrain_state_h1_online.md.
set -euo pipefail
set +x

# 所有 20 个容器使用同一个 RUN_TAG；每次新实验手动换一个值。
RUN_TAG="${RUN_TAG:-20261103_r29ba8fc7_run01}"
SHARED_REPO=/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm
CODE_COMMIT=29ba8fc703ef5e7f65a345c4c27769a7bdea0fac
export BASE_CHECKPOINT_PATH=/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/outputs/pretrain58src_state_h1_nano_160gpu_ga3_from5200_resize1f_mp4t4_benchphys_run01/cosmos3_action/action128/pretrain58src_state_h1_nano_160gpu_ga3_from5200_resize1f_mp4t4_benchphys_run01/checkpoints/iter_000005800
RUN_NAME=pretrain58src_state_h1_160gpu_ga3_from5800_resize1f_mp4t4_bgrecv_${RUN_TAG}
export OUTPUT_ROOT="$SHARED_REPO/outputs/$RUN_NAME"
SWANLAB_NAME=Cosmos3-Nano_58src_StateH1_160GPU_GA3_Resume5800_Resize1F_MP4T4_BGRecv_BenchPhysNoClip_${RUN_TAG}

# 平台注入：master=0，worker-0=1，...，worker-18=19。
export NODE_RANK="${NODE_RANK:-${RANK:?平台需注入 NODE_RANK 或 RANK}}"
: "${MASTER_ADDR:?平台需注入 master 节点地址 MASTER_ADDR}"
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < 20 ))
export NNODES=20 NPROC_PER_NODE=8 DP_SHARD=16 MASTER_PORT=50022
export ACTION128_PLATFORM_RENDEZVOUS=1 JOB_ID="$RUN_NAME"

mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/logs/entrypoint_node${NODE_RANK}.log") 2>&1
export LOG_FILENAME="action128_node${NODE_RANK}.log"
export LD_LIBRARY_PATH='' LD_LIBRARY_PATH_OVERRIDE=''
source /share-mixed/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/activate

# 使用已有训练 venv。新容器另需系统 FFmpeg 动态库。
# 已安装时跳过；安装只作用于容器，不修改共享 venv。
if ! command -v ffmpeg >/dev/null 2>&1; then
    BOOTSTRAP_PROXY="$(< /share-mixed/project/eai_pwm/.secrets/libero_sft_proxy)"
    env http_proxy="$BOOTSTRAP_PROXY" https_proxy="$BOOTSTRAP_PROXY" apt-get update -qq
    env http_proxy="$BOOTSTRAP_PROXY" https_proxy="$BOOTSTRAP_PROXY" apt-get install -y -qq ffmpeg
    unset BOOTSTRAP_PROXY
fi

# 训练写本地 .wandb；SwanLab 旁路自行读取私有 Key 和代理。
export WANDB_MODE=offline WANDB_ENTITY=ybainlp WANDB_SILENT=true
export WANDB_API_KEY_FILE=/share-mixed/project/eai_pwm/.secrets/cosmos_wandb_api_key
unset COSMOS_REQUIRE_WANDB_ONLINE WANDB_RUN_ID WANDB_RESUME WANDB_API_KEY
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,::1,$MASTER_ADDR,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

export NUMPY_MADVISE_HUGEPAGE=0 MIMALLOC_ALLOW_LARGE_OS_PAGES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export ACTION128_EPISODE_CACHE_DIR=/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1
export HF_MODEL_PATH=/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM
export WAN_VAE_PATH=/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth
export SFT_DIAGNOSTICS_DIR="$OUTPUT_ROOT/logs/diagnostics"
unset IMAGINAIRE_OUTPUT_ROOT MAX_ITER EXTRA_TAIL_OVERRIDES SFT_TRAINING_MODULE DATASET_PATH
ulimit -Sn 65536

# 仅针对当前 d39afa19 节点的已验证 RDMA 绕行；换集群应重新核验。
export NCCL_IB_HCA='=mlx5_101'
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET

# 固定同一 Git 提交并落盘到节点本地，避免共享盘 cwd/import 问题。
cd /tmp
mkdir -p /opt/cosmos-framework-pwm
LOCAL_REPO="$(mktemp -d /opt/cosmos-framework-pwm/online.XXXXXXXX)"
git --git-dir="$SHARED_REPO/.git" --work-tree="$SHARED_REPO" archive "$CODE_COMMIT" |
    tar -x -C "$LOCAL_REPO"
printf '%s\n' "$CODE_COMMIT" > "$LOCAL_REPO/git_commit.txt"
cd "$LOCAL_REPO"
export PYTHONPATH="$LOCAL_REPO"
export SFT_TOML_FILE="$LOCAL_REPO/examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml"

[[ -f "$BASE_CHECKPOINT_PATH/.save_complete" ]]
for component in model optim scheduler trainer; do
    [[ -f "$BASE_CHECKPOINT_PATH/$component/.metadata" ]]
done
for (( checkpoint_rank=0; checkpoint_rank<160; checkpoint_rank++ )); do
    [[ -f "$BASE_CHECKPOINT_PATH/dataloader/rank_${checkpoint_rank}.pkl" ]]
done
python - <<'CHECK'
import torch, torchcodec, zstandard, wandb, mcap, mcap_protobuf, google_crc32c
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 8, "Expected eight visible GPUs per node"
print("Training/media dependencies OK; torch=" + torch.__version__, flush=True)
CHECK

# 只在 master 启动命名明确的 SwanLab 旁路，绑定 rank0 日志。
if (( NODE_RANK == 0 )); then
    SWANLAB_PYTHON=/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-swanlab-bridge/bin/python
    "$SWANLAB_PYTHON" -c 'import swanlab, wandb; from PIL import Image'
    "$SWANLAB_PYTHON" "$LOCAL_REPO/tools/swanlab_sidecar.py" \
        --output-root "$OUTPUT_ROOT" --name "$SWANLAB_NAME" \
        --step-log "$OUTPUT_ROOT/logs/action128_node0.log" --background
fi

# auto：同一 58 源物理归一化契约下完整续训，而非旧的 extend。
# 当前提交默认开启 receive_in_background；配方使用 MP4 四线程。
exec bash examples/launch_pretrain_state_h1_96gpu.sh \
    checkpoint.action128_load_mode=auto \
    job.name="$RUN_NAME" job.wandb_mode=offline \
    trainer.callbacks.load_balance_trace.enabled=true \
    trainer.callbacks.load_balance_trace.flush_every_n_batches=10 \
    trainer.callbacks.load_balance_trace.max_samples_per_record=92 \
    trainer.callbacks.load_balance_trace.output_dir="$OUTPUT_ROOT/logs/load_balance_trace" \
    "$@"
