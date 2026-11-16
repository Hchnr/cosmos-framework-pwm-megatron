# T2V 双塔联合 attention 实验

实现记录与结果入口：[`record.md`](../../plans/v0.2.7_unified_attn/record.md)。生产默认保持 `split_default`。

在 SFT TOML 的 `[model]` 中设置 `joint_attn_backend`，或使用 Hydra override `model.config.joint_attn_backend=...`：

| 配置 | 含义 |
|---|---|
| `split_default` | 原框架两次 attention，原 FA3 NV provider |
| `split_fa3_fork` | 本地 FA3 fork，两次 varlen |
| `unified_fa3_grouped` | 同一 FA3 fork，一次 mixed causal/full grouped |
| `split_fa4` | FA4 两次 varlen，自定义算子支持 compile |
| `unified_fa4_mask` | FA4 一次 varlen，以 sequence/batch flag 选择 causal/full |
| `split_flex_triton` | Triton Flex 两次调用 |
| `unified_flex_triton` | Triton Flex 一次 UND/GEN 联合调用；共享原三塔 helper |

本轮覆盖 dense `two_tower` / `two_way`、CP1、BF16、head_dim128、H100；多视角、NATTEN、temporal causal、autoregressive memory、multi-control 不属于新增路径，冲突配置明确报错，且显式候选要求UND/GEN各有至少一个有效token。三塔继续使用原 `action_tower_attention_backend=flex` 开关。FA4 封装仅提供训练所需的一阶梯度，依赖记录版本的内部 forward/backward API。

FA3 NV 与 fork 的扩展注册相同 Torch namespace，**不能在同一进程加载**。fork 运行须在启动前设置 `COSMOS_FA3_PROVIDER=fork`，并把 fork 的 `hopper` 放入 `PYTHONPATH`；本实验入口自动处理。NV baseline 必须另起进程。

## 环境与运行

所有命令从当前 test 工作区运行，使用本地 `.venv-unified-attn/bin/python`。新增依赖、安装日志和完整 freeze 在 `plans/v0.2.7_unified_attn/env/`，fork 源码与二进制副本在 `artifacts/deps/hopper/`。不修改旧工作区或共享 `/opt/venv`。

```bash
LD_LIBRARY_PATH='' .venv-unified-attn/bin/python tools/unified_attn/run.py \
  --backend unified_flex_triton --name flex_round0 --warmup 10 --steps 30

LD_LIBRARY_PATH='' .venv-unified-attn/bin/python tools/unified_attn/run.py \
  --backend unified_flex_triton --name flex_nsys --warmup 10 --nsys
```

`--audit` 可在首个 optimizer update 保存选定层的本地参数 shard、clipped gradients 与更新后参数，供数值对照（位于预热窗口）。

`--name` 必须使用新目录名，防止 auto-resume 或覆盖证据。线上 TOML、checkpoint、数据混合与 0/1/2 帧条件分布保留；固定8卡 DP shard8 / replicate1 / CP1、GA2、seed42。默认 UND3072 + GEN98304；只有统一降档时设置 `--gen-tokens`。`--dryrun` 可检查配置。

正常加载模型/optimizer/scheduler/step25000；实验专用 checkpointer 跳过输出 checkpoint。每个 run 保留有效配置、命令/revision manifest、完整日志、退出状态、8个 rank 的 JSONL 和第一批真实 packing 几何。输入指纹覆盖样本 key、文本、latent shapes、CFG/条件帧信息和 sample/split 长度；它用于核对批次顺序，不是完整 latent 内容哈希。

正式时间来自无 profiler 的运行，排除统一预热窗口，以每个 optimizer step 的八 rank 最大耗时统计。时间包括 dataloading、metadata、attention 拼接、完整前反向、optimizer 和一致的回调开销。`summarize.py <run-dir> ...` 强制检查完整结束、GA2 和有限 loss，再输出 step、吞吐和显存。nsys 单独运行，rank0 捕获预热后的连续2个 optimizer step，其他7卡照常训练并记录指标；不能用 profile run 的时间宣称提速。

`kernel_bench.py` 使用某个成功 run 的 `geometry_rank*.json`，按线上 Q32/KV8/head_dim128 构造输入，单独报告 metadata 与包含 packing/dispatch 的 attention F/B 耗时。纯内核归因以真实训练 nsys 为准；它不代表整个训练收益。

## 测试

```bash
LD_LIBRARY_PATH='' PYTHONPATH=.:plans/v0.2.7_unified_attn/artifacts/deps/hopper \
COSMOS_FA3_PROVIDER=fork COSMOS_TEST_JOINT_FLASH=1 COSMOS_TEST_COMPILED=1 \
.venv-unified-attn/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/unified_attn/tests.py cosmos_framework/model/generator/mot/joint_attention_test.py
```

该入口在 pytest 导入前绑定 LOCAL_RANK，避免仓库默认 pytest-env 改写可见 GPU。正确性使用独立 FP32 SDPA oracle；实际完成状态以 record 和原始日志为准。

## 受控随机性与诊断

所有实验在每个逻辑 microbatch 前按 `(step, micro, rank)` 重设 Python、NumPy、Torch RNG；采样算法和分布不变。这样避免首次 compile/autotune 消耗随机数干扰跨 backend 对照。`compare_runs.py` 要求三轮完整运行、所有rank输入/RNG指纹一致及生产源码哈希一致。

`--audit` 首步额外保存真实 epsilon / xt / vt / sigma 的完整内容哈希，在梯度裁剪和 force_finite 清理前扫描非有限梯度，并记录 global norm / clip coefficient。实验显式启用 `error_if_nonfinite=true`；生产默认配置不变。基线自身梯度重复性存在明显差异，必须将单算子数值正确性与完整训练更新重复性分别审查，不能仅凭接近的loss宣称训练等价。

真实激活诊断只运行一至两步，不能混入性能或nsys数据：

```bash
LD_LIBRARY_PATH='' .venv-unified-attn/bin/python tools/unified_attn/run.py \
  --backend split_default --name capture_example --warmup 0 --steps 1 --audit --capture-attention
LD_LIBRARY_PATH='' .venv-unified-attn/bin/python tools/unified_attn/replay_suite.py \
  --run plans/v0.2.7_unified_attn/artifacts/runs/capture_example \
  --output plans/v0.2.7_unified_attn/artifacts/replay_example
```

采集0/17/35层首个microbatch的post-RoPE QKV、输出及真实dO。先校验B0回放与采集结果，再让六个候选复用同一份QKV/dO，对照输出及全部Q/K/V源梯度。审查最终profile可运行 `review_nsys.py <run-dir>`，导出SQLite、统计CSV、报告SHA256并检查完整的4个microbatch/2个optimizer更新。
