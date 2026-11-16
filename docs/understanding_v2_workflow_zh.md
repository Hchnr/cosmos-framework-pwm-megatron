# VLM Recipe V2：LMDB → Understanding Parquet → Cosmos 训练工作流

> 最后更新：2026-08-31（UTC）
> 本文记录本轮已经形成的可复现流程、关键决策和规范命令。本机逐次实验状态与监控节点见 `/home/baiyu/project/USR-to-Parquet/UNDERSTANDING_V2_PROGRESS.md`。

## 1. 范围与安全边界

本轮目标是复现 2026-08-14 冻结的 VLM recipe V2：把权威 catalog 中启用的 20 路 reference LMDB 分别转换成 20 个 Understanding Parquet dataset，再用 Cosmos 的统一 `OmniMoTModel` 路线进行 Qwen3-VL-4B 和 Cosmos3-Nano 8B 的读取、smoke test、packing 实验及生产训练。

安全边界：

- `/share` 下已有文件一律只读。
- 唯一曾获授权写入的 `/share` 范围是我们新生成的目标 Parquet root：
  `/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1`。
- 媒体只保留原始绝对路径，不复制、不转码、不生成新媒体。
- 训练日志、checkpoint、报告、备份和临时产物全部写在 `/home/baiyu` 或 `/tmp`。
- 不运行 `scripts/build_reuse_all.sh`，因为它包含大量与本轮冻结 V2 无关的数据。
- 训练出现 traceback、OOM、NCCL/ChildFailed、非有限 loss 或无法解释的停滞时，先停止目标并与用户讨论；不要自行修改训练源码。
- Git 认证仅可按命令临时使用 `/home/baiyu/.git-credentials`；禁止读取、打印、记录、修改或提交凭据内容，也不修改全局 Git 配置。

## 2. 仓库、分支与权威输入

### 2.1 转换仓库

- 工作目录：`/home/baiyu/project/USR-to-Parquet`
- 分支：`ybai_0827/understanding_processing`
- 当前已提交 HEAD：`1d25bd1b8ee7aa96619102c90ac0d06aea7571aa`
- upstream：`origin/ybai_0827/understanding_processing`

初始化和状态核对命令：

```bash
cd /home/baiyu/project/USR-to-Parquet
git status --short --branch
git branch --show-current
git remote -v
git rev-parse HEAD
git rev-parse '@{upstream}'

git -c credential.helper='store --file=/home/baiyu/.git-credentials' fetch origin --prune
git switch -c ybai_0827/understanding_processing origin/main
git -c credential.helper='store --file=/home/baiyu/.git-credentials' push -u origin ybai_0827/understanding_processing
```

上述 `switch -c` 只用于首次创建；分支已经存在时不要重复执行。

### 2.2 Cosmos 训练仓库

- 工作目录：`/home/baiyu/project/cosmos-framework-private`
- 分支：`ybai_0828/understanding_verification`
- 当前本地 HEAD：`e8bbcc0c17d6a890f032c4b7a10a29e7c596137e`
- 当前 upstream HEAD：`ab02c172078e0efe5a0c8192c45541753683d311`
- 本地分支当前相对 upstream ahead 3，且还有未提交修改。
- 初始核验基线包含 `dev-liyue@8092ec7`；后来合入了 `origin/dev-liyue` 的 Understanding-only checkpoint 更新。

规范命令：

```bash
cd /home/baiyu/project/cosmos-framework-private
git status --short --branch
git branch --show-current
git remote -v
git rev-parse HEAD
git rev-parse '@{upstream}'

git -c credential.helper='store --file=/home/baiyu/.git-credentials' fetch origin --prune
git switch -c ybai_0828/understanding_verification origin/dev-liyue
```

如果分支已存在，只执行只读状态核对和经过审查的 `git merge origin/dev-liyue`；不要覆盖本地修改。

### 2.3 权威 recipe 输入

训练配置：

```text
/home/baiyu/project/pwm-training-vlm-ablation-experiments/configs/qwen3vl/v2_0a/qwen3vl_v2_0a_vlm_robotics_fullscan_production_28x8.yaml
```

权威 catalog：

```text
/home/baiyu/project/pwm-training-vlm-ablation-experiments/data_catalogs/qwen3vl/v2_0a/qwen3vl_v2_0a_vlm_robotics_fullscan_repaired_plus_sharerobot_msr3d_robobench_production_28x8.datasets.yaml
```

## 3. 冻结的 20 路输入与清洗结果

原始总量为 14,184,661 行。只清除了 1,678 条只有一帧的非法视频样本：

- `sharerobot_planning_qa_v1_train`：删除 41 行。
- `robobench_official_prompt_all_training_v1_train`：删除 1,637 行。
- 其余 18 路未删行。

清洗后总量为 14,182,983 行。

| # | dataset | 原始行数 | 清洗后行数 | ratio | 周期占比 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `vlm_robotics_fullscan_v2r1_p0_episodic_embodied` | 1,374,549 | 1,374,549 | 6,136 | 20.286% |
| 2 | `vlm_robotics_fullscan_v2r1_p0_metric_3d_scene` | 303,852 | 303,852 | 1,356 | 4.483% |
| 3 | `vlm_robotics_fullscan_v2r1_p0_multiview_memory` | 264,770 | 264,770 | 1,182 | 3.908% |
| 4 | `vlm_robotics_fullscan_v2r1_p0_temporal_physical` | 1,098,554 | 1,098,554 | 4,907 | 16.223% |
| 5 | `vlm_robotics_fullscan_v2r1_p1_episodic_embodied` | 12,831 | 12,831 | 57 | 0.188% |
| 6 | `vlm_robotics_fullscan_v2r1_p1_free_space_affordance` | 13,645 | 13,645 | 61 | 0.202% |
| 7 | `vlm_robotics_fullscan_v2r1_p1_metric_3d_scene` | 197,061 | 197,061 | 879 | 2.906% |
| 8 | `vlm_robotics_fullscan_v2r1_p1_multiview_memory` | 48,063 | 48,063 | 214 | 0.707% |
| 9 | `vlm_robotics_fullscan_v2r1_p1_temporal_physical` | 530,860 | 530,860 | 2,370 | 7.835% |
| 10 | `vlm_robotics_affordance_support_v1_robot_placement` | 31,486 | 31,486 | 140 | 0.463% |
| 11 | `vlm_robotics_affordance_support_v1_semantic_affordance` | 202 | 202 | 1 | 0.003% |
| 12 | `vlm_robotics_fullscan_v2r1_p2_visual_structured_retention` | 3,557,460 | 3,557,460 | 4,944 | 16.345% |
| 13 | `vlm_robotics_general_retention_v1_safe_general` | 4,669,635 | 4,669,635 | 2,472 | 8.172% |
| 14 | `gqa_train_balanced_reference` | 943,000 | 943,000 | 446 | 1.474% |
| 15 | `mindcube_train_spatial_qa_reference` | 10,000 | 10,000 | 45 | 0.149% |
| 16 | `sharerobot_planning_qa_v1_train` | 1,027,990 | 1,027,949 | 4,589 | 15.171% |
| 17 | `sharerobot_affordance_bbox_text_v1_train` | 6,522 | 6,522 | 29 | 0.096% |
| 18 | `sharerobot_trajectory_points_text_v1_train` | 6,870 | 6,870 | 31 | 0.102% |
| 19 | `msr3d_strict_object_image_qa_v1_train` | 81,219 | 81,219 | 362 | 1.197% |
| 20 | `robobench_official_prompt_all_training_v1_train` | 6,092 | 4,455 | 27 | 0.089% |
| | **合计** | **14,184,661** | **14,182,983** | **30,248** | **100%** |

明确排除：

- `vlm_robotics_general_retention_v1_vstar_risk_general`（340,264）
- `vlm_robotics_onevision_optional_p2_recovery`（500,000）
- `model_judge_intphys_provisional_p1_capability`（28,800）

## 4. Parquet 输出格式与目录

20 路数据保持为 20 个独立的 Parquet dataset，以便后续精确恢复 V2 的数据权重。根目录是：

```text
/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1
```

每路数据只含 `part-*.parquet`，统一五列：

| 列 | 含义 |
| --- | --- |
| `id` | 全局稳定样本 ID |
| `type` | 本轮固定为 `understanding` |
| `dataset` | 20 路中的逻辑数据集名 |
| `data` | strict JSON；包含 messages、文本和原媒体路径 |
| `provenance` | strict JSON；记录 LMDB root、原始 key 等来源信息 |

媒体文件不会嵌入 Parquet。Parquet 使用列式存储、dictionary encoding、统计信息和 ZSTD 压缩，因此只有文本、重复字段和路径时压缩率会很高；文件大小不能直接代表行数或图片/视频内容大小。

## 5. LMDB → Parquet 转换流程

### 5.1 安装环境与测试

```bash
cd /home/baiyu/project/USR-to-Parquet
env -u VIRTUAL_ENV uv sync --extra dev
.venv/bin/pytest -q
```

### 5.2 只读查看单路转换计划

```bash
cd /home/baiyu/project/USR-to-Parquet
.venv/bin/data-processor plan \
  --config configs/understanding/qwen3vl_v2_0a_20260814_reference_v1/gqa_train_balanced_reference.yaml
```

调试时把输出定向到 `/tmp`，不要覆盖正式目录：

```bash
.venv/bin/data-processor build \
  --config configs/understanding/qwen3vl_v2_0a_20260814_reference_v1/gqa_train_balanced_reference.yaml \
  --max-rows 32 \
  --output /tmp/gqa_understanding_smoke
```

### 5.3 构建全部 20 路

```bash
cd /home/baiyu/project/USR-to-Parquet
bash scripts/build_v2_understanding.sh
```

也可以只构建指定 config：

```bash
bash scripts/build_v2_understanding.sh \
  configs/understanding/qwen3vl_v2_0a_20260814_reference_v1/gqa_train_balanced_reference.yaml
```

脚本对每一路先执行 `plan` 再执行 `build`，本地日志写到：

```text
/home/baiyu/project/USR-to-Parquet/outputs/qwen3vl_v2_0a_20260814_reference_v1/logs
```

### 5.4 校验与只读抽样

存储级校验：

```bash
PARQUET_ROOT=/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1
.venv/bin/data-processor validate "$PARQUET_ROOT/gqa_train_balanced_reference" --level storage
```

读取前两条记录：

```bash
PARQUET_DATASET=/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1/gqa_train_balanced_reference
.venv/bin/python - "$PARQUET_DATASET" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
part = sorted(root.glob("part-*.parquet"))[0]
table = pq.read_table(part).slice(0, 2)
for row in table.to_pylist():
    row["data"] = json.loads(row["data"])
    row["provenance"] = json.loads(row["provenance"])
    print(json.dumps(row, ensure_ascii=False, indent=2))
PY
```

如果只需要 metadata 或局部 row group，优先使用 `pq.ParquetFile(...)`、`metadata`、`schema_arrow`、`read_row_group(...)`，避免一次读入整个 dataset。

### 5.5 一帧视频清洗

默认命令只读扫描：

```bash
cd /home/baiyu/project/USR-to-Parquet
PARQUET_ROOT=/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1
.venv/bin/data-processor sanitize-understanding "$PARQUET_ROOT"
```

只有再次得到明确授权时才允许 apply，并且备份目录必须是一个尚不存在的 `/home` 路径：

```bash
.venv/bin/data-processor sanitize-understanding "$PARQUET_ROOT" \
  --apply \
  --backup-dir /home/baiyu/parquet_sanitize_backups/qwen3vl_v2_0a_<NEW_TIMESTAMP>
```

本轮已经完成的备份和 manifest 位于：

```text
/home/baiyu/parquet_sanitize_backups/qwen3vl_v2_0a_20260829T015712Z
```

不要重复执行已经完成的 apply。当前正式 root 已独立复扫：1,742 个 shard、14,182,983 行、零一帧视频、零隐藏 staging 文件。

## 6. Cosmos 读取链路

当前训练走统一模型路线：

```text
TOML
  → load_experiment_from_toml
  → build_catalog_training_dataloader
  → 20 × ParquetCatalogDataset / CosmosDataLoader
  → JointCosmosDataLoader(exact_periodic)
  → PoolPackingBatcher
  → Understanding collator / vlm_inputs
  → Cosmos3OmniModel / OmniMoTModel
  → Understanding 自回归 token loss
```

关键点：

- `vision_gen=false`、`action_gen=false`，只训练 VLM Understanding。
- `load_vlm_visual=true`；生产配置显式 `train_vlm_visual=true`，视觉塔已解冻。
- Understanding 使用 next-token 自回归 loss，不是 diffusion denoise loss。
- 直接视频由当前 `dev-liyue` reader 按视频时间范围进行均匀抽帧；Parquet 只保存媒体路径和消息结构。
- `IterSpeed` 已兼容 Understanding 的 `vlm_inputs.pixel_values` 和 `pixel_values_videos`。

## 7. 数据配比：不是普通 round-robin

### 7.1 精确机制

当前 `build_catalog_training_dataloader(...)` 创建：

```python
JointCosmosDataLoader(loaders, seed=seed, selection_mode="exact_periodic")
```

20 个 ratio 的最大公约数为 1，总和为 30,248，因此：

1. 先把 dataset index 按各自 ratio 重复，构成长度 30,248 的周期表。
2. 每个周期使用由 `seed + cycle` 派生的稳定 hash seed 洗牌。
3. 每个 microbatch 从周期表当前位置选择且只选择一路 dataset。
4. 30,248 个 microbatch 内，每一路被选择的次数严格等于其 ratio；顺序是确定性洗牌后的顺序。
5. 下一个周期重新进行确定性洗牌。

因此它具有“按整数配额轮转”的效果，但不是固定的 `dataset 1 → 2 → ... → 20` round-robin，也不是每步独立随机抽样。

### 7.2 小数据集是否会过多遍

会发生重复遍历，但需要区分两个层次：

- **dataset 选择频率**由 ratio 决定。多数小数据集的 ratio 约按每 220–225 行一个 ratio 单位缩放，所以它们不是被等权 round-robin 强行高频选择。
- **一次选择实际消耗多少行**由 packing 决定。`max_samples_per_batch=48` 只是每卡候选上限，不代表每卡一定有 48 行；文本/图片/视频 token 长度不同，实际 packed 行数会变化。
- 每个 `ParquetBlockDistributor` 是无限流。某一路当前 epoch 用完后，会增加 epoch，并用稳定 seed 重排 row group 和行，再继续供数。
- 因此短文本、小数据集在一次被选中时可能装入更多行，更容易跨 epoch；长视频流则可能每次只装入较少行。
- retention/general/GQA 等大数据集的 ratio 明显低于按行数等比例值，这是 recipe 的主动降权，而不是 loader bug。

结论：小数据集可能被重复，但不能只根据“dataset 小”推断它被过采样。正确审计量是：

```text
某路实际曝光比例
≈ 该路被选择的 microbatch 数
× 该路每次选择时 8 卡实际 packed 行数
÷ 该路总行数
```

### 7.3 gradient accumulation 对调度的影响

- 当前 Nano8B 生产训练：`grad_accum_iter=1`，每个 optimizer step 对应一次 20 路选择。
- Qwen4B 最终生产配置：`grad_accum_iter=2`，每个 optimizer step 对应两次 microbatch 选择。
- packing 上限 48 是每卡、每 microbatch 的候选上限，不是全局 optimizer step 固定样本数。
- 全局样本吞吐应从 8 个 rank 的实际 packed sample 数求和，不能用 `step × 48`。

### 7.4 resume 与正常重复不是同一问题

- 当前 Nano8B TOML 使用 `enable_map_style_resume=true`、`map_style_resume_mode="at_most_once"`；完整 DCP 含 8 个 rank 的 dataloader state。
- `at_most_once` 用于恢复时跳过可能已经消费的候选，避免因 batcher 重排而重放；它不能阻止正常进入下一 epoch。
- Qwen4B TOML 当前使用 `enable_map_style_resume=false`，所以不依赖 map-style loader state 做精确续训。
- 无论哪种模式，连续加权/周期采样都不是“全局每条唯一数据恰好一次”的 epoch 训练。

## 8. Cosmos 环境和回归测试

```bash
cd /home/baiyu/project/cosmos-framework-private
env -u VIRTUAL_ENV uv sync
```

Shell 静态检查：

```bash
bash -n examples/_sft_launcher_common.sh
bash -n examples/launch_sft_qwen3vl_v2_understanding_qwen4b.sh
bash -n examples/launch_sft_qwen3vl_v2_understanding_nano8b.sh
bash -n examples/launch_sft_qwen3vl_v2_understanding_qwen4b_production.sh
bash -n examples/retain_latest_dcp_checkpoint.sh
```

相关 Python 回归：

```bash
.venv/bin/pytest -q \
  cosmos_framework/configs/toml_config/qwen3vl_v2_understanding_config_test.py \
  cosmos_framework/callbacks/iter_speed_test.py \
  cosmos_framework/data/test/catalog_test.py \
  cosmos_framework/data/test/resume_test.py \
  cosmos_framework/model/generator/omni_mot_model_test.py \
  cosmos_framework/model/generator/mot/parallelize_visual_test.py \
  cosmos_framework/scripts/convert_qwen3vl_understanding_to_dcp_test.py
```

2026-08-31 提交前实测结果：

- 五个 shell 脚本的 `bash -n` 全部通过。
- 上述组合 pytest 为 `72 passed, 3 failed, 1 error`。
- 三个 failure 都来自 action/DROID catalog 测试：本机 `torchcodec` 无法加载匹配的 FFmpeg `libavutil` shared library；它们不是 Understanding 数据调度或训练回归。
- 一个 error 来自 untracked DCP converter 测试：`_build_target_config()` 留下了 `QWEN3_VL_4B_PATH` 环境变量，被仓库的环境泄漏 fixture 捕获。提交 converter 前应先决定在函数中 `try/finally` 恢复环境，或由测试使用 `monkeypatch` 管理；本轮审计没有擅自修改源码。
- 与当前 untracked 直接相关的 IterSpeed、visual parallelization 及 converter 其余测试已经包含在 72 个通过项中。

## 9. Qwen3-VL-4B → Understanding DCP

DCP（Distributed Checkpoint）是训练框架的分布式 checkpoint 格式。它把模型、optimizer、scheduler、trainer 和可选 dataloader state 分组件、分 rank 保存，使 FSDP 模型能够并行保存和恢复，而不要求先在单进程中拼出一个完整大 state dict。

本轮转换脚本生成的是 **model-only warm-start DCP**：

- 从 Qwen3-VL-4B Hugging Face safetensors 加载语言塔和视觉塔。
- 目标拓扑是统一 `OmniMoT` 的 pure Understanding 配置。
- 为严格统一语言塔兼容性，用 Understanding tower 初始化所需的 `*_moe_gen` 镜像。
- 不包含视觉生成、动作、音频、EMA、optimizer、scheduler、trainer 或 dataloader state。
- 源模型只读；输出若已存在会拒绝覆盖；先写 `.inprogress-*`，验证后原子 rename。

默认转换命令：

```bash
cd /home/baiyu/project/cosmos-framework-private
.venv/bin/python -m cosmos_framework.scripts.convert_qwen3vl_understanding_to_dcp
```

显式路径命令：

```bash
.venv/bin/python -m cosmos_framework.scripts.convert_qwen3vl_understanding_to_dcp \
  --source-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
  --output-path /home/baiyu/project/model_ckpts/Qwen3-VL-4B-OmniMoT-Understanding-DCP \
  --recipe-path examples/toml/sft_config/qwen3vl_v2_0a_understanding_qwen4b.toml
```

注意：当前 Qwen4B TOML 的 `checkpoint.load_path=""`，常规 launcher 是从 HF 权重初始化；converter 是独立的统一 DCP 兼容工具，不应把它误写成当前生产 run 已经使用的 base checkpoint。

## 10. 8 卡 smoke test

### 10.1 Qwen3-VL-4B

```bash
cd /home/baiyu/project/cosmos-framework-private
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=50142 \
OUTPUT_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/qwen4b_v2_smoke \
SMOKE_RUN=1 \
SMOKE_MAX_ITER=2 \
SMOKE_CHECKPOINT_SAVE_ITER=2 \
SMOKE_GRAD_ACCUM_ITER=1 \
bash examples/launch_sft_qwen3vl_v2_understanding_qwen4b.sh
```

### 10.2 Cosmos3-Nano 8B

```bash
cd /home/baiyu/project/cosmos-framework-private
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=50143 \
OUTPUT_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/nano8b_v2_smoke \
SMOKE_RUN=1 \
SMOKE_MAX_ITER=2 \
SMOKE_CHECKPOINT_SAVE_ITER=2 \
SMOKE_GRAD_ACCUM_ITER=1 \
bash examples/launch_sft_qwen3vl_v2_understanding_nano8b.sh
```

smoke 验收至少包括：8 个 rank 启动、能读取 20 路 Parquet、forward/backward/optimizer step 完成、loss 有限、完整 DCP 能保存、无 `/share` 写入。

## 11. Packing 吞吐实验

统一实验模板：

```bash
OUTPUT_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/<RUN_NAME> \
SMOKE_RUN=1 \
SMOKE_MAX_ITER=55 \
SMOKE_CHECKPOINT_SAVE_ITER=55 \
SMOKE_GRAD_ACCUM_ITER=<1_OR_2> \
EXTRA_TAIL_OVERRIDES='model.config.max_num_tokens_after_packing=<TOKENS> dataloader_train.max_sequence_length=<TOKENS> dataloader_train.max_samples_per_batch=<CANDIDATES> dataloader_train.packing_pool_size=<POOL> model.config.train_vlm_visual=true' \
bash examples/launch_sft_qwen3vl_v2_understanding_<qwen4b_OR_nano8b>.sh
```

已覆盖的主要候选：

- token budget：8,192、12,288、16,384、24,576、32,768、49,152、65,536。
- 每卡候选/packing pool：32、48、64；曾讨论继续上探，但 64 已出现更差长尾，未采用 96。

最终结论：

| 模型 | 最终 token budget | 每卡候选 | pool | grad accumulation | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-VL-4B | 49,152 | 48 | 48 | 2 | 稳定生产实测约 175.64 examples/optimizer-step；优于直接扩大 batch 的路线 |
| Cosmos3-Nano 8B | 65,536 | 48 | 48 | 1 | 当前生产配置；近期按 rank-0 loader-index 口径约 109 个全局 packed rows/step |

候选上限只控制一次 packing 可查看多少样本；最终 batch 仍受 token budget 和各样本多模态长度约束。

## 12. 生产训练

### 12.1 Qwen4B production wrapper

```bash
cd /home/baiyu/project/cosmos-framework-private
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OUTPUT_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/qwen4b_v2_production \
JOB_NAME=qwen3vl_v2_0a_understanding_qwen4b_production \
CHECKPOINT_SAVE_ITER=500 \
CHECKPOINT_KEEP_STEPS=1500,2000,3000,5000,8000,10000,20000,40000 \
MONITOR_INTERVAL_SECONDS=1200 \
bash examples/launch_sft_qwen3vl_v2_understanding_qwen4b_production.sh
```

Qwen4B 最终核心参数：

- `max_num_tokens_after_packing=49152`
- `max_samples_per_batch=48`
- `packing_pool_size=48`
- `grad_accum_iter=2`
- `lr=1e-5`
- `warm_up_steps=200`
- `grad_clip.clip_norm=1.0`
- `train_vlm_visual=true`

### 12.2 当前 Nano8B 生产训练

当前生产 run 的等价启动命令：

```bash
cd /home/baiyu/project/cosmos-framework-private
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=50144 \
OUTPUT_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/nano8b_v2_pack65536_fullvlm_production_20260830 \
LOG_FILENAME=nano8b_v2_pack65536_fullvlm_production.log \
EXTRA_TAIL_OVERRIDES='model.config.max_num_tokens_after_packing=65536 model.config.train_vlm_visual=true dataloader_train.max_sequence_length=65536 dataloader_train.max_samples_per_batch=48 dataloader_train.packing_pool_size=48 trainer.max_iter=40000 trainer.grad_accum_iter=1 trainer.logging_iter=1 scheduler.cycle_lengths=[40000] scheduler.warm_up_steps=[200] checkpoint.save_iter=500 job.name=qwen3vl_v2_0a_understanding_nano8b_pack65536_fullvlm_production' \
bash examples/launch_sft_qwen3vl_v2_understanding_nano8b.sh
```

tmux 形式：

```bash
tmux new-session -d \
  -s nano8b_v2_pack65536_fullvlm_production_20260830 \
  "cd /home/baiyu/project/cosmos-framework-private && <上面的生产命令>"
```

Nano8B 当前有效参数：

- `max_num_tokens_after_packing=65536`
- `max_samples_per_batch=48`
- `packing_pool_size=48`
- `grad_accum_iter=1`
- `lr=1e-6`
- `weight_decay=0.1`
- `warm_up_steps=200`
- `grad_clip.clip_norm=1.0`
- `train_vlm_visual=true`
- base DCP：`/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP`（只读）

## 13. DCP 保存与 retention

生产每 500 optimizer steps 保存一次。额外永久保留：

```text
1500, 2000, 3000, 5000, 8000, 10000, 20000, 40000
```

retention 命令：

```bash
cd /home/baiyu/project/cosmos-framework-private
CHECKPOINT_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/nano8b_v2_pack65536_fullvlm_production_20260830/cosmos3/understanding_verification/qwen3vl_v2_0a_understanding_nano8b_pack65536_fullvlm_production/checkpoints
CHECKPOINT_RETENTION_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/nano8b_v2_pack65536_fullvlm_production_20260830 \
bash examples/retain_latest_dcp_checkpoint.sh \
  "$CHECKPOINT_ROOT" \
  60 \
  1500,2000,3000,5000,8000,10000,20000,40000
```

该脚本只允许 `/home` 范围，先验证 `latest_checkpoint.txt` 和新 DCP 完整性，然后才删除旧的非里程碑 checkpoint。不要把 root 指到 `/share` 或仓库根目录。

只跑一轮 retention 审计：

```bash
RETENTION_ONCE=1 \
CHECKPOINT_RETENTION_ROOT=/home/baiyu/project/cosmos-framework-private/outputs/train/nano8b_v2_pack65536_fullvlm_production_20260830 \
bash examples/retain_latest_dcp_checkpoint.sh \
  "$CHECKPOINT_ROOT" \
  60 \
  1500,2000,3000,5000,8000,10000,20000,40000
```

## 14. 每 20 分钟只读监控

当前会话：

```text
nano8b_v2_pack65536_fullvlm_production_20260830
nano8b_v2_pack65536_fullvlm_retention_20260830
nano8b_v2_pack65536_fullvlm_monitor_20m_20260830
```

基础检查命令：

```bash
tmux list-panes -t nano8b_v2_pack65536_fullvlm_production_20260830 \
  -F 'pane_dead=#{pane_dead} pid=#{pane_pid} cmd=#{pane_current_command}'
tmux list-panes -t nano8b_v2_pack65536_fullvlm_retention_20260830 \
  -F 'pane_dead=#{pane_dead} pid=#{pane_pid} cmd=#{pane_current_command}'
tmux list-panes -t nano8b_v2_pack65536_fullvlm_monitor_20m_20260830 \
  -F 'pane_dead=#{pane_dead} pid=#{pane_pid} cmd=#{pane_current_command}'

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits

tail -n 100 \
  outputs/train/nano8b_v2_pack65536_fullvlm_production_20260830/logs/monitor_20m.log

grep -Ei 'Traceback|CUDA out of memory|ChildFailedError|NCCL.*error|Loss: (nan|inf|-inf)' \
  outputs/train/nano8b_v2_pack65536_fullvlm_production_20260830/logs/nano8b_v2_pack65536_fullvlm_production.log
```

checkpoint 完整性检查：

```bash
find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -printf '%f %y\n' | sort
sed -n '1p' "$CHECKPOINT_ROOT/latest_checkpoint.txt"
find "$CHECKPOINT_ROOT/<LATEST_ITER>" -type f -printf '%P\n' | sort
```

一个完整 8 卡训练 DCP 应有：

- model：1 个 `.metadata` + 8 个 distcp shard
- optim：1 + 8
- scheduler：1 + 8
- trainer：1 + 8
- dataloader：8 个 rank state
- 合计 44 个文件

截至 2026-08-31 06:59 UTC 的只读快照：训练已超过 step 7,600，8 个 rank 和三路 tmux 均存活；`iter_000007500` 完整，旧的非里程碑 `iter_000007000` 只在 7,500 完成并更新 latest pointer 后才被清理；保留的 1,500/2,000/3,000/5,000 均完整。

## 15. Loss 曲线

当前导出的对比图：

```text
/home/baiyu/project/USR-to-Parquet/artifacts/training_curves/qwen4b_vs_nano8b_loss_curves_20260830.png
```

当前曲线以 rank-0 `IterSpeed` 日志中的每步 loss 为观察口径。它不是按 token 数重新加权后的全局 loss，也不是 8 个 rank loss 的显式平均，因此必须配合 rolling mean/median、异常值计数和长期趋势解释，不能只看单步尖峰。

## 16. 提交前审计与测试

### 16.1 通用只读审计

```bash
git status --short --branch
git diff --stat
git diff --check
git diff
git ls-files --others --exclude-standard
```

逐个查看 untracked 文件：

```bash
sed -n '1,240p' <TEXT_FILE>
file <BINARY_FILE>
wc -l -c <TEXT_FILE>
```

### 16.2 USR-to-Parquet 建议测试

```bash
cd /home/baiyu/project/USR-to-Parquet
.venv/bin/pytest -q \
  tests/test_sanitize_understanding.py \
  tests/test_understanding_new_sources_20260829.py
.venv/bin/pytest -q
git diff --check
```

### 16.3 Cosmos 建议测试

使用第 8 节列出的定向 pytest 和 `bash -n`，再执行：

```bash
cd /home/baiyu/project/cosmos-framework-private
git diff --check
```

### 16.4 精确暂存、复核、提交和推送

不要使用 `git add .`。按审查后的逻辑分组精确暂存：

```bash
git add <FILE_1> <FILE_2> ...
git diff --cached --stat
git diff --cached
git commit -m '<MESSAGE>'
git -c credential.helper='store --file=/home/baiyu/.git-credentials' push origin HEAD
```

本轮文档编写阶段不自动 commit，也不自动 push；必须先把下节 untracked 文件和完整 diff 复核完毕。

## 17. 2026-08-31 untracked 文件审计摘要

### 17.1 USR-to-Parquet

| 文件/组 | 主要内容 | 建议 |
| --- | --- | --- |
| `artifacts/training_curves/qwen4b_vs_nano8b_loss_curves_20260830.png` | 2,586×1,755 的 4B/8B loss 对比图 | 作为独立文档/实验 artifact 决定是否提交 |
| `configs/understanding/qwen3vl_v2_0a_20260829_new_sources_v1/*` | DROID、Ego-Exo4D、HoloAssist、RoboMIND2.0、TACO 的探索性 batch spec | 与冻结 V2 分开提交 |
| `scripts/build_understanding_new_sources_20260829.py` | 约 1,789 行的一次性新来源 builder，包含 20 个新 alias 和有限媒体提取 | 体量大，单独严格 review/commit |
| `scripts/build_understanding_new_sources_report_20260829.py` | 静态审计站点、bbox 预览、JSON/CSV/checksum/zip 报告 | 与新来源 builder 配套提交 |
| `src/data_processor/sanitize_understanding.py` | 默认 dry-run；apply 时 checksum 备份、staging、原子替换、回滚、manifest | 与 CLI 和单测一起提交 |
| `tests/test_sanitize_understanding.py` | dry-run、apply、checksum、schema、幂等、临时文件测试 | 与 sanitizer 同组 |
| `tests/test_understanding_new_sources_20260829.py` | 新来源 20 任务、TACO 时间戳、bbox schema/provenance 测试 | 与新来源工作同组 |

新来源 2026-08-29 工作不是冻结 V2 的一部分，不能和 V2 LMDB→Parquet 核心变更混为一个 commit。

### 17.2 cosmos-framework-private

| 文件/组 | 主要内容 | 建议 |
| --- | --- | --- |
| `cosmos_framework/callbacks/iter_speed_test.py` | Understanding 图片/视频/text-only 的 IterSpeed 回归 | 与 `iter_speed.py` 同组 |
| `cosmos_framework/model/generator/mot/parallelize_visual_test.py` | 只对可训练视觉块做 AC/FSDP、冻结视觉参数忽略逻辑 | 与视觉塔解冻/并行化改动同组 |
| `cosmos_framework/scripts/convert_qwen3vl_understanding_to_dcp.py` | Qwen3-VL-4B HF → pure Understanding OmniMoT model-only DCP | 与 converter test 独立成组 |
| `cosmos_framework/scripts/convert_qwen3vl_understanding_to_dcp_test.py` | 目标 config、视觉覆盖、tied lm_head、`*_moe_gen` 镜像测试 | 与 converter 同组 |
| `examples/launch_sft_qwen3vl_v2_understanding_qwen4b_production.sh` | Qwen4B production supervisor、500-step 保存和 20 分钟监控 | 与 launcher/recipe 组一起 review |
| `examples/retain_latest_dcp_checkpoint.sh` | 保留 latest+里程碑、删除 `/home` 下旧非里程碑 DCP | 含删除逻辑，提交前重点安全审查 |
| `docs/understanding_v2_workflow_zh.md` | 本文；跨转换、训练、DCP 和监控的中文工作流 | Cosmos 文档组 |

## 18. 建议提交拆分

USR-to-Parquet：

1. V2 sanitizer：CLI、`sanitize_understanding.py`、focused test。
2. V2 持续进度记录：`UNDERSTANDING_V2_PROGRESS.md`。
3. 2026-08-29 新来源探索：config、builder、report builder、test。
4. loss PNG：如确需进 Git，作为独立 artifact commit。

Cosmos：

1. catalog exact-periodic 调度与 map-style resume/callback 测试。
2. pure Understanding OmniMoT、视觉塔训练和并行化修复及测试。
3. IterSpeed 的 Understanding batch 兼容及测试。
4. Qwen4B/Nano8B TOML、launcher、production/retention 脚本。
5. Qwen3-VL-4B → DCP converter 及测试。
6. 中文端到端工作流：`docs/understanding_v2_workflow_zh.md`。

最终提交前仍需查看每组完整 staged diff，确认没有训练输出、凭据、`/share` 文件、临时文件或无关实验混入。
