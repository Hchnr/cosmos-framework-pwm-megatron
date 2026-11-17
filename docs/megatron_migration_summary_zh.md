# Megatron 接入中文总结报告

更新日期：2026 年 11 月 17 日

## 一、接入结果

原始 58-source、State128 三塔训练已接入 **Megatron-Core 0.20.0 的 Megatron-FSDP backend**。模型、数据、loss 和训练配置继续使用 Cosmos 实现，由 Megatron 管理数据并行分片与通信。

当前在 **4 张 H100 80GB** 上完成验证：

| 项目 | 结果 |
| --- | --- |
| 原始 checkpoint 续训 | 保留旧归一化，恢复模型、优化器、scheduler、trainer/RNG |
| FP32 全模型正确性 | 36 层、811 个参数梯度的全局相对 L2 误差 **4.27×10⁻⁸** |
| BF16 loss 对照 | 两组跨 backend 比较最大相对差 **0.238% / 0.294%** |
| 整次更新平均耗时 | 原生 PyTorch **40.11 s** → Megatron ZeRO-3 **38.39 s** |
| 吞吐收益 | **+4.48%** |
| 峰值分配显存 | **81.764 → 81.835 GB**，增加约 **0.087%** |

性能结论对应四卡、原始 token 容量的短程测试，包含数据等待，排除 checkpoint 写入和 tokenizer 编译。FP32 正确性测试保留完整模型层数，使用缩小 token 容量和 SDPA math；其耗时不计入性能比较。192 卡性能尚未实测，BF16 全模型梯度尚不能保证统一的 2% 误差上界。

性能来源的实测分析、原始 nsys 文件位置和 Nsight App timeline 对比，见 [四卡性能归因报告](./megatron_fsdp_nsys_attribution_zh.md)。

## 二、分布式策略

### 2.1 ZeRO-3 具体 shard 什么

本实现的 ZeRO-3 配置值为 `optim_grads_params`，在 **DP shard group** 内分摊以下持久训练状态：

| 对象 | ZeRO-3 中的处理 |
| --- | --- |
| 模型计算权重 | 通常为 BF16，各卡长期持有其中一片；计算相应模块前 all-gather 完整权重 |
| FP32 master weights | 各卡持有自己负责更新的权重分片 |
| 参数梯度 | 通过 reduce-scatter 等通信聚合并分片，各卡负责对应部分 |
| Adam 优化器状态 | 一阶矩 `m`、二阶矩 `v` 随参数分片，各卡只更新本地部分 |
| 激活、输入 batch | **ZeRO-3 本身不沿序列维分片这些内容**；激活显存主要通过重计算控制 |

分片以 Transformer block 等 FSDP unit 组织。前向和反向需要某个 unit 时聚合权重，按需释放完整计算权重；梯度聚合完成后，各 rank 用本地梯度、master weights 和 Adam 状态更新本地分片。

因此，“4 卡 ZeRO-3”表示主要持久状态按 4 卡分摊，**不表示每卡峰值显存等于单卡的四分之一**：计算时仍需临时完整权重、激活、通信 buffer 和工作区。

实现位置：[megatron.py:94](../cosmos_framework/training_backends/megatron.py#L94)、[optimizer.py:393](../cosmos_framework/utils/generator/optimizer.py#L393)。

### 2.2 其他可选分片策略

通过 `MEGATRON_SHARDING_STRATEGY` 或 `model.config.parallelism.megatron_sharding_strategy` 切换：

| 配置值 | 对应策略 | 计算权重 | 梯度 | FP32 master weights / Adam 状态 |
| --- | --- | --- | --- | --- |
| `no_shard` | 普通数据并行 | 复制 | 复制并同步 | 复制 |
| `optim` | ZeRO-1 | 复制 | 复制并同步 | 分片 |
| `optim_grads` | ZeRO-2 | 复制 | 分片 | 分片 |
| `optim_grads_params` | ZeRO-3 | 分片、按需聚合 | 分片 | 分片 |

**生产启动脚本默认 ZeRO-2，本次四卡原始容量测试使用 ZeRO-3。** ZeRO-2 常驻完整计算权重，减少模块级反复 all-gather，代价是更多显存；ZeRO-3 进一步节省权重存储，增加按需聚合通信。应结合模型容量与集群通信开销选择。

配置定义：[sft_config.py:138](../cosmos_framework/configs/toml_config/sft_config.py#L138)；启动默认值：[launch_pretrain_state_h1_58src_192gpu.sh:58](../examples/launch_pretrain_state_h1_58src_192gpu.sh#L58)。

### 2.3 HSDP、CP、TP、PP 的支持情况

| 策略 | 当前实现及使用情况 |
| --- | --- |
| DP / FSDP | 已接入；不同 rank 处理各自数据，用分片通信同步训练状态 |
| HSDP：分片 + 外层复制 | 已接入；由 `data_parallel_shard_degree` 和 `data_parallel_replicate_degree` 配置 |
| CP：上下文并行 | Cosmos 两塔路径已支持与 Megatron 分片组合，已有 CP=2/4 的功能验证；**本次三塔模型强制 CP=1** |
| TP：张量并行 | **本接入路径未实现**；runtime 的 `tp` 维固定为 1，没有切分 Linear 的矩阵计算 |
| PP：流水线并行 | **本接入路径未实现**；没有把不同模型层分配到不同 stage，也没有流水线调度 |

生产计划布局为 **192 卡 = shard16 × replica12**：每组 16 卡分摊训练状态，共 12 组副本，副本之间同步梯度。ZeRO-3 下状态按 **16** 分摊，而非按 192 分摊。四卡测试则为 **shard4 × replica1**。

CP 沿序列维切分 token 与激活；本仓库的 CP 叠加在现有 DP mesh 上，不额外乘出一组 GPU。仓库包含 CP 配置不代表三塔训练可直接设为 CP>1，三塔代码有显式限制。Megatron 库具备其他并行能力，也不代表当前 Cosmos 接入已经启用。

实现位置：[mesh 与 HSDP:118](../cosmos_framework/training_backends/megatron.py#L118)、[CP 配置:165](../cosmos_framework/configs/toml_config/sft_config.py#L165)、[三塔 CP 限制:317](../cosmos_framework/model/generator/omni_mot_model.py#L317)、[两塔 CP 安装:304](../cosmos_framework/model/generator/mot/parallelize_unified_mot.py#L304)。

## 三、训练优化的启用情况

### 3.1 Attention：已使用 FlashAttention，但按分支选择

生产配置为 `action_tower_attention_backend="flash3_video"`：

- **VIDEO 分支使用 FlashAttention 3 的 varlen 路径**，按样本组织 Q/KV 和累计长度。
- **UND / ACTION 分支使用 FlexAttention 块稀疏路径**，按样本和因果可见性构造 BlockMask，跳过跨样本不可见区域。
- Attention metadata 在 decoder 层外构造并复用，避免逐层、逐次重计算重复构建。

因此当前训练并非所有 attention 都使用 FlashAttention。该混合策略沿用原始三塔训练设置，Megatron 接入保留其计算路径。

实现位置：[small_action_tower.py:172](../cosmos_framework/model/generator/mot/small_action_tower.py#L172)、[small_action_attention.py:1](../cosmos_framework/model/generator/mot/small_action_attention.py#L1)。

### 3.2 Compile：已启用逐 block 编译

`[model.compile]` 中 `enabled=true`、`compiled_region="language"`、`compile_dynamic=false`。语言 / MoT decoder 按 Transformer block 使用 `torch.compile(fullgraph=True, dynamic=False)`，配合固定 token 容量减少形状变化和重编译。

Tokenizer / VAE 另有编译 callback：第 3 次迭代后执行预设分辨率的编译与覆盖检查。**跨进程 compile cache 当前关闭**，不能据此认为重启后无需编译。

Megatron 适配保留原有编译范围，并处理其参数存储检查与 Dynamo 的兼容性。

配置与实现：[模型编译配置:40](../examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml#L40)、[逐 block 编译:185](../cosmos_framework/model/generator/mot/parallelize_unified_mot.py#L185)、[Tokenizer 编译:399](../examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml#L399)。

### 3.3 重计算：已启用完整 activation checkpointing

`[model.activation_checkpointing] mode="full"`，对每个 Transformer block 包装 activation checkpoint。前向保留 block 边界所需张量，反向重新计算 block 内中间激活，以增加计算换取激活显存下降；不是只重算 attention，也不是对整个训练 step 重跑。

Megatron 的通信 hooks 已适配 Cosmos 的嵌套输出结构，保证反向重计算前能正确聚合所需权重。重计算与 ZeRO-3 解决不同的显存来源，当前同时启用。

实现位置：[full AC:114](../cosmos_framework/model/generator/mot/parallelize_unified_mot.py#L114)、[逐层包装:155](../cosmos_framework/model/generator/mot/parallelize_unified_mot.py#L155)、[Megatron backward hooks:20](../cosmos_framework/training_backends/megatron.py#L20)。

### 3.4 Packing：已启用多样本 packing 与固定容量

多个样本按 token 预算装入同一微批，再将文本、视频、动作编码后的 token 组织为 packed sequence。样本边界、position 与 attention mask 保持独立，避免不同样本互相注意。

当前固定配置为 **UND 11776 / GEN 62592 / ACTION 2944**；其中 GEN 包含 Action，三塔实际拆为 **VIDEO 59648 + ACTION 2944**，UND 另有 **5120** 的短桶；每个微批最多 **92 个样本**，实际数量由各模态 token 预算决定。固定容量会保留 padding，用来稳定编译形状，packing 不代表完全没有 padding。

当前固定长度路径采用顺序 token-budget / packing batcher，按顺序收集能容纳的样本；溢出的样本留给后续微批，不依赖池内重排。配置 schema 中虽有 `packing_pool_size=16` 默认值，**当前固定长度路径不使用该池式 packing 分支**。

配置与实现：[容量配置:34](../examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml#L34)、[batcher 选择:969](../cosmos_framework/data/generator/catalog_processors.py#L969)、[序列 packing:1](../cosmos_framework/data/generator/sequence_packing/packers.py#L1)。

### 3.5 通信、优化器与数据准备

| 优化 | 当前状态与作用 | 来源 |
| --- | --- | --- |
| 梯度通信 / 参数聚合与计算重叠 | `overlap_grad_reduce=true`、`overlap_param_gather=true`，降低通信暴露时间 | Megatron backend |
| 梯度累积 | GA=3；按最后一个微批标记完成梯度同步，裁剪前等待通信完成；ZeRO-3 的按需参数聚合仍会发生 | 原训练设置 + Megatron 适配 |
| 梯度范数与裁剪 | 本地 FP32 范数聚合后，在 shard group 做一次范数归约，使用 foreach 缩放；兼容空分片，避免逐参数范数 collective | Megatron 适配 |
| 混合精度与优化器 | BF16 模型计算，FusedAdam 与 FP32 master weights，保留原有学习率及 weight decay 分组 | 沿用原训练 |
| 模态路由投影 | VIDEO / ACTION 只计算各自 token 对应的投影和 MLP，避免冗余的全 GEN 计算 | 沿用原训练 |
| 数据准备 | shared prefetch，16 workers、prefetch 8，复用媒体 seek groups 与 episode metadata cache | 沿用原训练 |
| 主机内存与代码读取 | queued video 使用 pageable memory；关闭 NumPy / mimalloc 大页建议；训练代码暂存到节点本地 | 沿用原训练 |
| Checkpoint | DCP 异步保存，每 200 步保存；Megatron 分片元数据适配原有保存 / 恢复链路 | 原机制 + Megatron 适配 |

额外的 QKV / MLP 输入投影融合已有可选实现，但 `megatron_fuse_projections=false`，**当前默认未启用**。三塔 recipe 的 EMA 也关闭。

以上多数计算和数据优化在原训练中已经存在，本次接入主要新增分片通信及其与原训练链路的适配。**4.48% 是整体 backend 对比收益，没有逐项消融，不能给每项优化单独分配收益。**

实现位置：[通信配置:143](../cosmos_framework/training_backends/megatron.py#L143)、[累积同步:262](../cosmos_framework/training_backends/megatron.py#L262)、[梯度裁剪:298](../cosmos_framework/training_backends/megatron.py#L298)、[数据配置:69](../examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml#L69)、[可选投影融合:465](../cosmos_framework/model/generator/omni_mot_model.py#L465)。

## 四、训练入口与配置位置

仓库：`/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron`

远端：<https://github.com/Hchnr/cosmos-framework-pwm-megatron>

| 内容 | 位置 |
| --- | --- |
| 58-source Megatron 生产入口 | [launch_pretrain_state_h1_58src_192gpu.sh:1](../examples/launch_pretrain_state_h1_58src_192gpu.sh#L1) |
| 训练主配置：模型、容量、优化器、58 源配比、loader、checkpoint | [action128_pretrain_state_h1_58src_192gpu.toml:1](../examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml#L1) |
| 多节点公共启动器 | [_sft_launcher_common.sh:175](../examples/_sft_launcher_common.sh#L175) |
| Python 训练入口 | [train.py:1](../cosmos_framework/scripts/train.py#L1) |
| TOML schema 与配置组装 | [sft_config.py:1](../cosmos_framework/configs/toml_config/sft_config.py#L1) |
| Megatron runtime | [megatron.py:94](../cosmos_framework/training_backends/megatron.py#L94) |
| 四卡对比测试入口 | [benchmark_megatron_full_depth.sh:1](../tools/benchmark_megatron_full_depth.sh#L1) |

启动链路为：58-source shell → 52-source 拓扑与本地 staging shell → 96gpu 路径包装 → Action128 公共 shell → `python -m torch.distributed.run ... -m cosmos_framework.scripts.train --sft-toml=...`。中间脚本名是复用的历史名称，实际数据配置仍由 58-source TOML 决定。

主 TOML 定义模型和训练行为；58-source shell 额外选择 `backend=megatron`、分片策略和旧归一化续训配置；最后的命令行 Hydra overrides 可覆盖前面的设置。直接使用该 TOML 启动 Python 时，需要显式选择 Megatron backend。

默认生产布局为 `NNODES=24`、`NPROC_PER_NODE=8`、`DP_SHARD=16`，各节点由平台提供一致的 rendezvous 信息。启动示例：

```bash
# 生产入口：默认 ZeRO-2，各节点执行
bash examples/launch_pretrain_state_h1_58src_192gpu.sh

# 4 卡 ZeRO-3：将拓扑覆盖为 shard4 × replica1
NNODES=1 NPROC_PER_NODE=4 DP_SHARD=4 \
MEGATRON_SHARDING_STRATEGY=optim_grads_params \
BASE_CHECKPOINT_PATH=/path/to/checkpoints/iter_000001200 \
OUTPUT_ROOT="$PWD/outputs/megatron_58src_4gpu" \
bash examples/launch_pretrain_state_h1_58src_192gpu.sh
```

常用环境变量：`BASE_CHECKPOINT_PATH` 指定续训 checkpoint；`OUTPUT_ROOT` 指定输出位置；`SFT_TOML_FILE` 替换主配置；`MEGATRON_SHARDING_STRATEGY` 选择分片策略；`MAX_ITER` 覆盖绝对结束迭代。

## 五、续训语义与当前边界

当前按原始续训目标，恢复模型、优化器、scheduler、trainer/RNG，并保留 checkpoint 中 106 个旧域的归一化；6 个新增域使用当前配置。State128 使用配套旧统计快照，模型与 loader 绑定同一份契约。追加数据源后按原始 `extend` 语义重置数据流游标。

旧归一化配置由生产 shell 注入：`checkpoint.action128_preserve_source_normalization=true` 和 `dataloader_train.action_state_normalization_paths`。恢复旧归一化不等于恢复发布方升级前的所有原始数据值与 mask。

当前可用于四卡范围的继续验证；192 卡吞吐、长程稳定性和更大范围数据覆盖仍需补充。详细测试记录保留在独立的 [megatron_migration.md:1](./megatron_migration.md#L1)，本报告集中说明最终实现、配置和使用方式。
