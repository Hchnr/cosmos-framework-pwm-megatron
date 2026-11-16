# Nano / Edge 训练脚本

本文汇总当前可用的 Cosmos3-Nano 和 Cosmos3-Edge 训练入口、模型路径、数据要求及启动方式。

## 1. 本地模型路径

当前开发机已经准备好以下模型：

| 模型 | 路径 | 用途 |
|---|---|---|
| Cosmos3-Nano DCP | `/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP` | Nano 训练权重，包含 VLM visual tower |
| Cosmos3-Nano HF | `/share/project/eai_pwm/models/Cosmos3-Nano` | Nano tokenizer 和 Qwen3-VL processor |
| Cosmos3-Edge DCP | `/share/project/liyue/model_ckpts/Cosmos3-Edge-DCP` | Edge 训练权重，包含 Edge visual tower |
| Cosmos3-Edge HF | `/share/project/liyue/model_ckpts/Cosmos3-Edge` | Edge tokenizer 和多模态 processor |
| Wan2.2 VAE | `/share/project/liyue/model_ckpts/vae/wan22_vae/Wan2.2_VAE.pth` | Generation 和 Action 的视频编码 |

Nano DCP 约为 30 GB，包含语言模型、Generation/Action 模块和 351 个 visual tower tensor。纯 Understanding
启动时只构造 Understanding 与 visual tower；DCP 中额外的 Generation 参数不会进入模型或后续训练 checkpoint，Wan VAE
也不会初始化。

以上路径已经写入对应 launcher 的默认值，仍可通过同名环境变量覆盖：

```bash
export BASE_CHECKPOINT_PATH=/another/model.dcp
export HF_MODEL_PATH=/another/hf_snapshot
export WAN_VAE_PATH=/another/Wan2.2_VAE.pth
```

## 2. 公共运行环境

镜像 `cosmos-framework:260821` 已经在 `/root/.venv` 中安装好训练环境。除平台专用脚本外，进入项目目录后优先直接激活该环境：

```bash
cd /share/project/liyue/cosmos-framework
source /root/.venv/bin/activate
export LD_LIBRARY_PATH=''
export NPROC_PER_NODE=8
```

先确认解释器和 `torchrun` 可用：

```bash
which python
which torchrun
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

正常情况下前两个路径都应位于 `/root/.venv/bin/`。只有 `/root/.venv` 不存在或确实无法使用时，才检查项目目录下保留的环境：

```bash
cd /share/project/liyue/cosmos-framework
source .venv/bin/activate
```

不要因为切换了工作目录就重新安装依赖；虚拟环境位置与当前目录无关。

可选环境变量：

| 变量 | 含义 |
|---|---|
| `DATASET_PATH` | 单个 Parquet 文件或包含 Parquet 的 catalog 目录 |
| `OUTPUT_ROOT` | 训练输出目录，默认 `outputs/train` |
| `NPROC_PER_NODE` | 每个节点启动的 GPU 进程数，默认 8 |
| `MASTER_PORT` | `torchrun` 端口，默认 50012 |
| `SMOKE_RUN=1` | 两步端到端验证；关闭 compile，并使用独立 smoke job 名称 |
| `EXTRA_TAIL_OVERRIDES` | 追加 Hydra override，例如 `trainer.max_iter=100` |

正式训练时不设置 `SMOKE_RUN`。第一次运行建议先用 `SMOKE_RUN=1` 检查数据、DCP、NCCL 和显存。

## 3. 可直接使用的样例数据

| 任务 | 路径 |
|---|---|
| Understanding | `/share/project/liyue/Project/data-processor/outputs/dataloader_samples/understanding_onevision` |
| Visual Generation | `/share/project/liyue/Project/data-processor/outputs/dataloader_samples/visual_generation_action100m` |
| DROID WAM Action | `/share/project/liyue/Project/data-processor/outputs/dataloader_samples/action_droid_cosmos.parquet` |

DROID 样例 catalog 的 `mode` 是 `wam`。它可以验证 T2V+Action，但不能直接用于 Forward Dynamics；FD 需要单独生成 `mode="forward_dynamics"` 的单行 Action catalog。

## 4. Nano 训练入口

### 4.1 Visual Generation

入口：`examples/launch_sft_vision_nano.sh`

用途：训练 T2I、T2V、TI2V/I2V 和 V2V。Generation target 经过 Wan VAE，assistant reasoning 存在时可同时计算文本 CE。

两步验证：

```bash
DATASET_PATH=/share/project/liyue/Project/data-processor/outputs/dataloader_samples/visual_generation_action100m \
SMOKE_RUN=1 \
bash examples/launch_sft_vision_nano.sh
```

正式训练：

```bash
DATASET_PATH=/path/to/generation_catalog \
OUTPUT_ROOT=/share/project/liyue/training_log/nano_generation \
bash examples/launch_sft_vision_nano.sh
```

### 4.2 Packed Understanding

入口：`examples/launch_sft_videophy2_nano.sh`

用途：纯 VLM Understanding。使用统一 Omni/MoT 模型、Qwen3-VL chat template 和 packed CE；不初始化 Wan VAE、DiT Generation 或 Action head。

```bash
DATASET_PATH=/share/project/liyue/Project/data-processor/outputs/dataloader_samples/understanding_onevision \
SMOKE_RUN=1 \
bash examples/launch_sft_videophy2_nano.sh
```

### 4.3 LLaVA-OneVision Understanding

入口：`examples/launch_sft_llava_ov.sh`

用途：与上一入口使用相同的 Nano Understanding 模型，保留独立 recipe 以配置 LLaVA-OneVision 类数据的学习率和训练周期。

```bash
DATASET_PATH=/path/to/llava_onevision_catalog \
SMOKE_RUN=1 \
bash examples/launch_sft_llava_ov.sh
```

### 4.4 DROID WAM Action（1B 三塔方案 D）

统一入口为 `examples/launch_sft_action_policy_1b.sh`。数据、转换后的 checkpoint、
多机配置和本地测试覆盖见 [Action 使用指南](action_policy_1b.md)。
本地测试使用相同入口的命令行覆盖，不使用 `SMOKE_RUN` 或独立测试配置。

## 5. Edge 训练入口

### 5.1 Visual Generation

通用入口：`examples/launch_sft_vision_edge.sh`

```bash
DATASET_PATH=/share/project/liyue/Project/data-processor/outputs/dataloader_samples/visual_generation_action100m \
SMOKE_RUN=1 \
bash examples/launch_sft_vision_edge.sh
```

当前平台推荐入口：`examples/launch_sft_vision_edge_platform.sh`

该脚本额外设置当前平台的 NCCL 网卡、IB HCA、本地 venv、离线 HF 模式和默认输出目录：

```bash
DATASET_PATH=/share/project/liyue/Project/data-processor/outputs/dataloader_samples/visual_generation_action100m \
SMOKE_RUN=1 \
bash examples/launch_sft_vision_edge_platform.sh
```

### 5.2 Packed Understanding

入口：`examples/launch_sft_videophy2_edge.sh`

用途：使用 Edge 原生 visual tower 和统一 Omni/MoT decoder 训练 packed Understanding；不加载 Wan VAE。

```bash
DATASET_PATH=/share/project/liyue/Project/data-processor/outputs/dataloader_samples/understanding_onevision \
SMOKE_RUN=1 \
bash examples/launch_sft_videophy2_edge.sh
```

当前没有单独的 Edge Action launcher。现有 DROID、LIBERO、FD Action recipes 均基于 Cosmos3-Nano。

## 6. Qwen3-VL-4B 直接启动 Understanding

`understanding_sft_qwen4b.toml` 可从本地 Hugging Face snapshot 直接初始化瘦身 Understanding 模型，不需要先转 DCP：

```bash
export QWEN3_VL_4B_PATH=/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct
export DATASET_PATH=/share/project/liyue/Project/data-processor/outputs/dataloader_samples/understanding_onevision
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/understanding_sft_qwen4b.toml
```

## 7. 将训练后的 Understanding DCP 拼回完整模型

纯 Understanding checkpoint 不含 Generation/Action 参数。需要恢复完整训练能力时，以同架构旧完整 DCP
为骨架，用新 Understanding 权重覆盖同名参数：

```bash
python -m cosmos_framework.scripts.merge_understanding_dcp \
  --generation-checkpoint-path /path/to/old_full_dcp \
  --understanding-checkpoint-path /path/to/trained_und_checkpoint \
  --output-path /path/to/merged_full_dcp
```

命令拒绝覆盖已有输出，通过配置（存在时）及严格的 key、shape、dtype 校验模型 family，并在完整 DCP 含 EMA 时同步更新 Understanding EMA。
输出只含模型权重及模型配置，是新的 warm start，不包含 optimizer、scheduler 或 dataloader cursor。

## 8. 多节点启动

在每个节点运行同一个 launcher，并设置：

```bash
export NNODES=2
export NODE_RANK=0        # 第二个节点设置为 1
export MASTER_ADDR=<rank-0-host>
export MASTER_PORT=50012
export NPROC_PER_NODE=8
```

Action 方案 D 使用 DP shard=总 GPU 数；本地验证的覆盖方式见 [Action 使用指南](action_policy_1b.md)。

## 9. 日志位置

默认日志：

```text
outputs/train/logs/<toml-name>_sft.log
```

设置 `OUTPUT_ROOT` 后：

```text
$OUTPUT_ROOT/logs/<toml-name>_sft.log
```

平台 Edge launcher 默认写入：

```text
/share/project/liyue/training_log/logs/
```
