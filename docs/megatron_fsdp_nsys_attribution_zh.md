# Megatron-FSDP 性能收益归因：四卡 Nsight Systems 对比

测试日期：2026-11-17；范围为当前 Megatron-FSDP 接入，TP/PP 尚未开始实现。

## 结论

此前无 profiler 的四卡 ABBA 测试测得吞吐提升 **4.48%**。本次用 Nsight Systems 采集相同训练路径，两个稳态更新的整次耗时为 **40.73 s → 39.09 s**，方向与量级一致。Profile 用来分析来源，不用它取代无 profiler 的性能结论。

**收益主要体现在分片通信与计算的组织方式，以及更少的 GPU 空闲和参数/梯度整理开销；没有观察到 Attention 或主要 decoder GEMM 本身的大幅提速。**

- 原生 FSDP2 与 Megatron 的 all-gather / reduce-scatter kernel 调用次数接近，Megatron 略多；收益不是简单地少执行了一批 collective。
- Megatron 的 NCCL 活动时间更短，但其中包含跨卡等待，不能直接解释为网络带宽提升。
- 原生 FSDP 的部分 all-gather 拆分拷贝、梯度打包 kernel 在 Megatron 路径中消失。
- 已标注的 `optimizer_step` 范围只节省几十毫秒，无法解释约 1.65 秒/更新的主要差距；该范围不保证包含全部梯度统计与裁剪工作。

## 采集条件

4 × H100 80GB，完整 36 层三塔，原始 token 容量、BF16、GA=3、逐 block compile、full activation checkpointing、原始 packing 与生产 loader。两端从同一 `iter_000001200` 恢复完整训练状态和旧归一化；对比原生 PyTorch FSDP2 与 Megatron ZeRO-3。每端运行 5 次更新，前 3 次预热，采集第 **1204、1205** 次更新，每个 rank 一份报告。

两端 **60 个微批**的输入内容、游标与恢复 RNG 记录逐项一致。关闭 checkpoint 写入和 tokenizer 编译，两端保留相同的内容哈希检查。采集 `cuda,nvtx,nccl`，关闭 CPU sampling/context-switch sampling；使用 Nsight Systems **2026.2.1**。

## Nsight App 原生 timeline

以下图片直接截取同版本 Nsight Systems App 的 timeline 区域，保留原生事件、stream 与时间轴；没有用绘图工具重建或修改事件。每张图均可从相应原始 `.nsys-rep` 重现。

### 两个完整更新总览（rank 0）

原生 PyTorch FSDP2：

![原生 FSDP2 两个更新的 Nsight timeline](./images/megatron_fsdp_nsys/pytorch_overview.png)

Megatron ZeRO-3：

![Megatron ZeRO-3 两个更新的 Nsight timeline](./images/megatron_fsdp_nsys/megatron_overview.png)

每张图的横轴都是实际秒数，总览自动适配各自约 81.5 / 78.2 秒的采集窗口；不能只按图片像素宽度比较耗时。主线程 NVTX 行包含 2 个更新 × GA3 的 6 个微批。

### 首个微批反向附近，约 7–14 秒（rank 0）

原生 FSDP2：

![原生 FSDP2 反向区段](./images/megatron_fsdp_nsys/pytorch_backward.png)

Megatron ZeRO-3：

![Megatron ZeRO-3 反向区段](./images/megatron_fsdp_nsys/megatron_backward.png)

原生 FSDP2 的通信 stream 在后半段出现明显的 ReduceScatter 活动块；Megatron 的对应活动更短、更分散。原生微批约 **13.89 s**，Megatron 约 **13.06 s**。这是一个直观例子，整体结论使用下述全部四个 rank 的统计，而非只挑 rank 0 的局部画面。

## GPU 时间差在哪里

以两次完整更新、四个 rank 的 **8 个观测均值**计算。通过所有 CUDA stream 的时间区间取并集，避免将并发 kernel 时间重复相加。标题中的 40.73 / 39.09 秒取 benchmark 每步最慢 rank 的计时；下表取各 rank NVTX 完整范围的均值，两者统计口径略有不同。

| 互斥时间分区 | 原生 FSDP2 | Megatron ZeRO-3 | 原生减去 Megatron |
| --- | ---: | ---: | ---: |
| GPU 无 kernel / memop 活动的间隙 | 3.892 s | 2.954 s | **0.939 s** |
| 只有 NCCL、没有其他 GPU 工作重叠的区间 | 1.546 s | 1.075 s | **0.471 s** |
| 非 NCCL kernel / memop 活跃区间，含其与 NCCL 重叠部分 | 35.289 s | 35.049 s | **0.240 s** |
| 整个更新 | **40.728 s** | **39.078 s** | **1.650 s** |

这是一种**观测时间分解**：GPU 空闲减少约占差值的 57%，NCCL-only 减少约占 29%，其余 GPU 工作活跃区间减少约占 15%（四舍五入）。它不是三个独立优化开关的因果占比。

GPU 空闲可能来自主机发射间隔、依赖等待和 rank 间节奏；没有采集 CPU 调度采样，不能进一步把全部 0.939 秒都归为 Python。NCCL-only 也不等于全部暴露在最终 critical path 上的网络成本。

### 通信与同步

| 指标：每卡每更新平均 | 原生 FSDP2 | Megatron ZeRO-3 |
| --- | ---: | ---: |
| AllGather kernel 次数 | 220 | 221 |
| ReduceScatter kernel 次数 | 111 | 117 |
| AllGather kernel 累计时长 | 2.830 s | 2.274 s |
| ReduceScatter kernel 累计时长 | 2.412 s | 1.450 s |
| NCCL 活动时间并集 | 5.249 s | 3.741 s |
| NCCL 与其他 GPU 工作重叠时间 | 3.702 s | 2.666 s |
| `cudaStreamSynchronize` 次数 | 903.9 | 833.1 |
| `cudaStreamSynchronize` 主机 API 累计时长 | 6.327 s | 4.391 s |

通信重叠的绝对秒数降低，并不意味着重叠优化变差：整个 NCCL 活动窗口也更短。NCCL kernel 时间包含等待其他 rank 的部分；CUDA 同步 API 的主机等待又可能与 GPU 执行重叠，上表各行不能相加当作总收益。

### 数据整理与主要计算

| Kernel / 阶段 | 原生 FSDP2 | Megatron ZeRO-3 | 解释 |
| --- | ---: | ---: | --- |
| `split_with_sizes_copy_out_contiguous_no_cast_kernel` | 219 次，0.136 s | 0 次 | 原生 all-gather 输出拆分拷贝 |
| `chunk_cat_cuda_kernel` | 111 次，0.035 s | 0 次 | 原生梯度 reduce-scatter 输入整理 |
| 主要 FlashAttention 3 forward kernel | 216 次，2.740 s | 216 次，2.722 s | 相同调用数，时长基本持平 |
| 主要 FlashAttention 3 backward kernel | 108 次，3.653 s | 108 次，3.657 s | 基本持平 |
| 代表性 decoder GEMM：`nvjet…192x192…TNN` | 432 次，3.485 s | 432 次，3.455 s | 无大幅计算加速 |
| optimizer 的 CPU NVTX 阶段 | 0.054 s | 0.024 s | 有收益，但量级较小 |

两端 FA3 forward/backward 调用数相同，未通过少做重计算获得收益。前五行是 kernel 累计服务时间，不是互斥 wall time。消失的两组整理 kernel 共约 0.17 秒，说明缓冲区布局确实减少了实际 GPU 工作；它只解释一部分收益，不能与前面的 1.650 秒分解再相加。

原生 PyTorch FSDP collectives 的拆分与打包调用，与实际 kernel 名对应；Megatron 使用自己的连续 buffer / 参数视图和通信 pipeline。当前 runtime 开启参数聚合与梯度归约重叠，见 [megatron.py:131](../cosmos_framework/training_backends/megatron.py#L131)。两个 backend 都使用相同的 Cosmos 模型计算路径，并非 Megatron 换用了更快的一套 Attention/MLP 模型。

## 参数预取消融

真正的消融直接拦截 `AllGatherPipeline.all_gather_params`，将 lookahead `prefetch` 设为 false，保留 ZeRO-3 必需的异步通信机制。rank 0 在初始化与全部 5 次更新中实际拦截 **1106 次**预取请求；两端 60 个微批输入/游标/RNG 记录一致，loss 最大相对差为 **0.272%**。

| 对比项 | 默认预取 | 禁止 lookahead prefetch |
| --- | ---: | ---: |
| 两个稳态更新平均耗时，按最慢 rank | 39.091 s | 39.333 s |
| NCCL 与其他 GPU 工作重叠时间，四 rank 均值 | 2.666 s | 0.022 s |
| NCCL-only 时间，四 rank 均值 | 1.075 s | 1.992 s |
| 非 NCCL 工作活跃时间，四 rank 均值 | 35.049 s | 34.342 s |
| GPU 空闲时间，四 rank 均值 | 2.954 s | 2.985 s |

关闭预取后整步慢 **0.242 秒，约 0.62%**，仍明显接近默认 Megatron，而非回到原生 FSDP2 的 40.73 秒。这支持“参数预取有帮助，但不是 backend 收益的唯一或主要解释”。这里每组只有两个 profiled 更新，不将 0.62% 当作稳定的独立收益比例。

还可看到一个抵消关系：NCCL-only 增加约 0.917 秒，但非 NCCL 工作活跃时间缩短约 0.707 秒。减少通信与计算并发可能降低资源竞争，但内核选择、显存布局和运行波动也可能参与；本次未进一步隔离。**不能把默认运行的 2.666 秒重叠直接宣称为节省 2.666 秒。**

有效消融的原生 timeline：

![Megatron 禁止 lookahead prefetch 后的反向区段](./images/megatron_fsdp_nsys/megatron_no_prefetch_backward.png)

可复核统计见 [ablation_comparison.json](../artifacts/megatron/nsys_attribution_20261117/ablation_comparison.json) 与 [消融时间分析](../artifacts/megatron/nsys_attribution_20261117/ablation_no_prefetch_summary.json)。

另保留一次无效尝试：仅给构造器传 `overlap_param_gather=false` 时，MCore 会在 ZeRO-3 初始化中强制改回 true。因此 `ablation_no_param_overlap/` 是重复默认行为的记录，**不能用它证明关闭预取没有影响**。后续有效消融位于 `ablation_no_prefetch/`。

## 原始文件与人工 review

原始报告永久保留在共享工作区；大文件未纳入 Git。路径根目录：

```text
/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron/outputs/megatron_validation/nsys_attribution_20261117/
```

| 子路径 | 内容 |
| --- | --- |
| `profile/pytorch/rank{0,1,2,3}.nsys-rep` | 原生 FSDP2 四卡原始报告 |
| `profile/megatron/rank{0,1,2,3}.nsys-rep` | Megatron ZeRO-3 四卡原始报告 |
| `review/pytorch_fsdp2_rank0.nsys-rep` | rank 0 原始报告的相同字节副本，使用可读文件名 |
| `review/megatron_zero3_rank0.nsys-rep` | 同上，Megatron 路径 |
| `ablation_no_prefetch/megatron/rank{0,1,2,3}.nsys-rep` | 有效禁止 lookahead prefetch 的四卡报告 |
| `ablation_no_param_overlap/megatron/rank{0,1,2,3}.nsys-rep` | 无效配置尝试，明确标注，禁止用作消融结论 |
| `nsys_raw_reports.tar` | 全部 16 份原始报告与 SHA256 清单，约 218 MB |
| `profile/<backend>/rank*.sqlite` | CLI 导出的 SQLite，便于独立统计 |
| `profile/<backend>/rank0_stats_*.csv` | 官方 nsys stats 输出 |
| `profile/<backend>/launch.log` | 启动、训练和 profiler 导出日志 |

人工查看建议：用 Nsight Systems 2026.2.1 或兼容版本分别打开两个 `review/` 报告，展开 `CUDA HW → All Streams` 与通信 stream，查看主线程 NVTX 微批。框选约 7–14 秒，使用 **Shift+Z** 放大；跨 rank 检查请打开对应 `rank1/2/3` 原始文件。原生截图显示的 stream 编号可能不同，应按名称和 NCCL 事件识别，不能假定编号相同。

原始报告与数据包的校验清单见 [raw_reports_manifest.json](../artifacts/megatron/nsys_attribution_20261117/raw_reports_manifest.json) 和 [archive.json](../artifacts/megatron/nsys_attribution_20261117/archive.json)。

版本化的统计、原始报告 SHA256、输入一致性结果保存在 [artifacts/megatron/nsys_attribution_20261117](../artifacts/megatron/nsys_attribution_20261117)。完整 kernel 名及每卡每更新服务时间可查 [summary.json](../artifacts/megatron/nsys_attribution_20261117/summary.json)，逐 rank / update 的并集统计见 [profile_analysis.json](../artifacts/megatron/nsys_attribution_20261117/profile_analysis.json)。

## 复现

设置源 checkpoint 和一个新的输出目录，执行：

```bash
BASE_CHECKPOINT_PATH=/path/to/checkpoints/iter_000001200 \
NSYS_ROOT="$PWD/outputs/megatron_validation/my_nsys_run" \
bash tools/profile_megatron_fsdp.sh
```

参数预取消融额外设置 `PROFILE_BACKENDS=megatron PROFILE_PARAM_PREFETCH=0`，使用独立的 `NSYS_ROOT`。工具入口：[采集脚本:1](../tools/profile_megatron_fsdp.sh#L1)、[时间区间分析:1](../tools/analyze_megatron_fsdp_nsys.py#L1)、[统计汇总:1](../tools/summarize_megatron_fsdp_nsys.py#L1)。

本次是四卡、两个稳态更新的诊断，不能替代更长程、多窗口或 192 卡性能归因。Nsight 中 bounded capture 会产生部分初始化 NVTX 范围不完整和 worker 无 NVTX 的提示；核心更新范围与 CUDA kernel 数据已单独检查，详细诊断消息保存在统计文件中。
