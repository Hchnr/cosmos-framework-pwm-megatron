# Megatron 接入中文总结报告

报告日期：2026 年 11 月 17 日（北京时间）  
结果依据：实现与验收记录提交 [`c26104a`](https://github.com/Hchnr/cosmos-framework-pwm-megatron/commit/c26104ab9e07b028803ecf37ada27a3f9f3b2a23)。

## 一、结论

原始 58-source 训练已接入 Megatron-Core。在用户指定的 **4 张 H100 80GB** 上，已跑通原始 checkpoint 的续训，并完成原始容量配置下的 BF16 性能比较及独立的全模型 FP32 梯度对照。

| 验收项 | 主要结果 | 当前结论 |
| --- | --- | --- |
| 原始状态续训 | 从 `iter_000001200` 恢复模型、优化器、scheduler、trainer/RNG；保留旧归一化 | 四卡续训通过 |
| FP32 正确性 | 36 层、252 个网络输入张量逐位一致；811 个参数梯度全局相对 L2 误差 **4.27×10⁻⁸** | 独立 FP32 对照通过 |
| BF16 数值表现 | 两组跨 backend 配对的 loss 最大相对差分别为 **0.238% / 0.294%** | 通过本次 loss 误差门槛 |
| 性能 | 含数据等待的整次更新均值 **40.11 → 38.39 秒**，吞吐提升 **4.48%** | 本次四卡配置有收益 |
| 显存 | 峰值分配显存 **81.764 → 81.835 GB**，增加约 **0.087%** | 基本持平，无 OOM |

**结论限于本次四卡 ZeRO-3 验证范围。尚不能据此宣称 192 卡性能已验收，也不能宣称 BF16 全模型梯度始终满足统一的 2% 误差上界。**

证据：[性能验收结果:1](../artifacts/megatron/source_full_20261117_qualification.json)、[FP32 验收结果:1](../artifacts/megatron/full_depth_fp32_20261117_qualification.json)。

## 二、任务范围与实现方式

原始训练入口：

```text
/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/examples/launch_pretrain_state_h1_58src_192gpu.sh
```

新实现仓库：

```text
/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron
https://github.com/Hchnr/cosmos-framework-pwm-megatron.git
```

本次采用 **Megatron-FSDP backend**：由 Megatron-Core 管理参数、梯度及优化器状态分片，以及相关通信；继续使用 Cosmos 原有的数据处理、三塔模型计算、loss、优化器参数分组、FP32 Adam master weights、scheduler、编译和 activation checkpointing。原生 PyTorch backend 保留，作为对照路径。

本路径未启用 TP/PP，生产三塔配置保持 CP=1。需要区分两项配置：

- **本次四卡完整配置验收使用 ZeRO-3**，即 `optim_grads_params`。
- **192 卡生产 launcher 的默认值仍为 ZeRO-2**，即 `optim_grads`；四卡 ZeRO-3 的结果不替代该默认布局的 192 卡验收。

主要适配内容包括复杂模型输出的 backward hooks、activation recomputation、梯度通信与裁剪顺序、空分片统计归约，以及 DCP checkpoint 分片元数据处理。

实现参考：[megatron.py:94](../cosmos_framework/training_backends/megatron.py#L94)、[生产 launcher:61](../examples/launch_pretrain_state_h1_58src_192gpu.sh#L61)。

## 三、旧归一化与完整训练状态续接

### 发现的问题

原始最新 checkpoint 为 `iter_000001200`。其保存的契约与当前 58-source 数据配置之间，有 **12 个旧 Bench2Dex 域的归一化 scale/bias/mask 或摘要发生变化**。原生 PyTorch 的严格 `extend` 校验同样会拒绝这一差异，因此不能直接跳过校验加载。

根据用户确认，本次采用“**保留 checkpoint 旧归一化，继续原始训练状态**”的语义。

### 实际处理

1. 在模型构建前读取源 checkpoint 的契约，按域名保留 **106 个旧域**的动作归一化；**6 个新增域**使用当前配置。
2. 模型和数据 loader 使用同一份合并后的 Action128 契约，继续严格校验域身份、embodiment、投影路由、版本和空间 RoPE 等信息。
3. State128 输入有独立的归一化，不能仅从动作 normalizer 推导。已纳入发布方升级前的 State 快照，并通过显式路径映射加载。
4. 恢复后，Bench2Dex 处理指纹与 checkpoint 保存值完全一致，原有数据流指纹校验保持启用。
5. 恢复模型、优化器、scheduler、trainer/RNG。**追加数据源后重置数据流游标**，遵循原始 `extend` 语义，并非从旧 52-source 的原游标直接延续到新 58-source 调度。

对应配置为 `checkpoint.action128_preserve_source_normalization=true` 与 `dataloader_train.action_state_normalization_paths`，生产 launcher 已提供配置。

需要保留一项数据边界：发布方升级还改过部分被动 Action 的原始值和 mask。本次恢复的是旧归一化及训练状态，**不等同于恢复升级前每一个原始数据 payload**。

证据与实现：[归一化保留记录:1](../artifacts/megatron/source_normalization_preservation.json)、[首次严格校验失败记录:1](../artifacts/megatron/source_extend_contract_failure.json)、[action128.py:11](../cosmos_framework/checkpoint/action128.py#L11)、[配置组装:1336](../cosmos_framework/configs/toml_config/sft_config.py#L1336)。

## 四、正确性验证

### 1. 原始状态、原始容量的 BF16 续训

两端先分别从第 1200 步续训至第 1203 步，随后进行四次配对运行，每次从同一源 checkpoint 续训至第 1208 步。

四次运行中，每张卡各记录 **24 个微批**；跨卡共 96 个不同内容哈希的记录，在四次运行间逐项一致，包括输入内容、游标和恢复 RNG 信息。未出现非有限 loss、非有限裁剪范数、跳过优化器更新、实际媒体错误或 OOM。

BF16 loss 对照结果：

| 比较范围 | 最大相对差 |
| --- | ---: |
| 第一组 PyTorch / Megatron | 0.23769% |
| 第二组 PyTorch / Megatron | 0.29355% |
| PyTorch 自身两次重复 | 0.43786% |
| Megatron 自身两次重复 | 0.35769% |

因此，全部四次 loss 轨迹相对首轮 PyTorch 的最大差为 **0.43786%**，该最大值来自 PyTorch 自身重复运行。上述结果是 loss 层面的验证，不能代替完整梯度误差比较。

证据：[原始状态续训检查:1](../artifacts/megatron/source_extend_preserved_qualification.json)、[ABBA 数值与输入检查:1](../artifacts/megatron/source_full_20261117_qualification.json)。

### 2. 独立的全模型 FP32 梯度对照

为进一步区分 backend 接入误差与 BF16 参考波动，保留完整 **36 层模型**进行临时 FP32 对照：

- 使用真实 batch 回放、GA=3、编译和完整 activation checkpointing。
- 容量缩小为 UND/GEN/ACTION = **512 / 8192 / 128**。
- 从原始 checkpoint 做模型初始化，使用当前 catalog 归一化及新优化器状态；该测试与上面的原始状态 BF16 续训是两项互补验证。
- 强制 SDPA math backend、确定性算法，关闭 matmul/cuDNN TF32，并检查参数与梯度 dtype。

| 指标 | 实测值 | 测试前设定门槛 |
| --- | ---: | ---: |
| 网络输入 | 252 个张量逐位一致 | 一致 |
| 比较范围 | 全部 811 个参数梯度 | 全量比较 |
| 全局梯度相对 L2 误差 | **4.2703274×10⁻⁸** | ≤ 10⁻⁴ |
| 梯度余弦相似度 | **0.9999999999025** | ≥ 0.999999 |
| 两端最终微批 loss | **0.1289336234331131** | 有限 |

全部门槛通过。FP32 测试改变了精度、attention backend 和 token 容量，并包含梯度导出开销，其耗时明确排除在性能结论之外。

早期 BF16 全模型梯度对照仍完整保留：对新 PyTorch 参考的相对 L2 误差约 **1.8611%**，对归档参考则约 **2.2247%**，后者未通过当时的 2% 门槛；两份 PyTorch 参考之间也存在约 **0.7312%** 的梯度差。因此，本报告不宣称统一的 BF16 2% 梯度上界。新增 FP32 结果独立验证 backend 梯度计算误差。

证据：[FP32 全量梯度结果:1](../artifacts/megatron/full_depth_fp32_20261117_gradients.json)、[FP32 验收:1](../artifacts/megatron/full_depth_fp32_20261117_qualification.json)、[保留的 BF16 失败记录:1](../artifacts/megatron/zero3_20261117_gradient_qualification.json)。

## 五、性能验证

### 测试设置

| 项目 | 配置 |
| --- | --- |
| 硬件与环境 | 4 × H100 80GB；PyTorch 2.10.0+cu130；Megatron-Core 0.20.0 |
| 模型 | 原始三塔结构，完整 36 层，BF16，FusedAdam FP32 master weights |
| Token 容量 | UND 11776 / GEN 62592 / ACTION 2944；保留 short UND 5120 策略 |
| 训练 | GA=3、梯度裁剪、编译、完整 activation checkpointing |
| 生产 loader | 16 workers、prefetch 8、packing pool 16、最多 92 样本 |
| 状态 | 从原始第 1200 步恢复优化器、scheduler、trainer/RNG及旧归一化 |
| 运行顺序 | PyTorch → Megatron → Megatron → PyTorch（ABBA） |
| 采样 | 每次更新 8 步，剔除前 3 步；每端共 10 个稳定阶段观测 |

主指标是**包含数据等待的整次更新耗时**。两端均包含内容哈希检查开销，排除 checkpoint 写入和 tokenizer 编译。测试前已提交验收门槛：输入完全一致、无非有限/跳过更新及媒体错误、loss 相对差不超过 1%，整体及两组配对吞吐均不得下降超过 3%。

### 测试结果

| 指标 | 原生 PyTorch | Megatron ZeRO-3 |
| --- | ---: | ---: |
| 整次更新平均耗时 | 40.11032 s | 38.39142 s |
| 整次更新 P95 | 42.15494 s | 39.91611 s |
| 平均数据等待 | 0.00469 s | 0.00444 s |
| 峰值分配显存（十进制 GB） | 81.76409 | 81.83536 |

吞吐收益按 `PyTorch 平均耗时 / Megatron 平均耗时 − 1` 计算，整体为 **4.4773%**；两组正反顺序配对分别为 **4.6080%** 和 **4.3465%**。显存增加 **71,267,328 bytes，约 0.0872%**。本次测试显示稳定的正向收益，且显存基本持平；短程样本不支持向其他规模直接外推。

证据：[性能汇总:1](../artifacts/megatron/source_full_20261117_summary.json)、[预设门槛:1](../artifacts/megatron/source_full_20261117_acceptance_policy.json)。

## 六、适用边界与后续验证

已完成的验证针对当前批准的四卡范围，仍有以下边界：

1. **192 卡尚未实测**，包括原始 shard16 × replica12 布局的通信、显存、长尾与实际吞吐。
2. **本次 4.48% 收益不包含 checkpoint 写入及 tokenizer 编译**，不能直接作为包含这些开销的长程训练总收益。
3. 四次运行各 8 步，稳定阶段为 5 个更新 profile 的两次重复；实际采样覆盖 **15/58 个配置数据流**，未遍历全部数据窗口。
4. FP32 对照使用缩小容量和 SDPA math；BF16 性能测试使用原始容量及生产 attention 路径，两者用途不同。
5. 全部失败、通过及参考波动记录均保留，未通过放宽旧校验来消除不兼容或失败结果。

后续建议按顺序补充：更长程四卡训练与 checkpoint 保存/重启检查；192 卡原始并行布局实测；扩大数据流与窗口覆盖。再据此评估大规模上线范围。

## 七、复现与交付

在仓库根目录、使用已配置的训练环境执行以下命令，可复现原始状态、原始容量的四卡 ABBA 测试。`BASE_CHECKPOINT_PATH` 需要指向实际存在且完整保存的源 checkpoint。

```bash
BASE_CHECKPOINT_PATH=/path/to/checkpoints/iter_000001200 \
BENCHMARK_LIVE_DATA=1 BENCHMARK_PRODUCTION_CAPACITIES=1 \
BENCHMARK_PRODUCTION_LOADER=1 BENCHMARK_RESTORE_SOURCE_STATE=1 \
BENCHMARK_EXPECTED_START_ITERATION=1200 BENCHMARK_STEPS=1208 \
BENCHMARK_WARMUP_STEPS=3 BENCHMARK_TAG=source_extend_full_capacity \
MEGATRON_SHARDING_STRATEGY=optim_grads_params \
bash tools/benchmark_megatron_full_depth.sh
```

其中 `BENCHMARK_STEPS=1208` 是绝对结束迭代，表示从第 1200 步继续更新 8 步。

交付内容包括实现代码、生产启动入口、旧 State 归一化快照、复现工具、原始测量 JSON、16 份 rank 输入记录、逐步耗时、全量梯度比较和验收门槛。大体积原始张量位于忽略跟踪的 `outputs/megatron_validation/`，未纳入 Git。

复现入口：[benchmark_megatron_full_depth.sh:1](../tools/benchmark_megatron_full_depth.sh)、[FP32 梯度审计入口:1](../tests/megatron/full_depth_gradient_audit.py)。完整技术记录见 [megatron_migration.md:1](./megatron_migration.md)。
