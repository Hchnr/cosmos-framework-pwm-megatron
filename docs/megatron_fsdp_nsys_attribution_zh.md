# Megatron-FSDP 性能收益归因：四卡 Nsight Systems 对比

测试日期：2026-11-17；范围为当前 Megatron-FSDP 接入，TP/PP 尚未开始实现。

## 结论

此前无 profiler 的四卡 ABBA 测试测得吞吐提升 **4.48%**。本次用 Nsight Systems 采集相同训练路径，两个稳态更新的整次耗时为 **40.73 s → 39.09 s**，方向与量级一致。Profile 用来分析来源，不用它取代无 profiler 的性能结论。

**收益主要体现在分片通信与计算的组织方式，以及更少的 GPU 空闲和参数/梯度整理开销；没有观察到 Attention 或主要 decoder GEMM 本身的大幅提速。**

- 原生 FSDP2 与 Megatron 的 all-gather / reduce-scatter kernel 调用次数接近，Megatron 略多；收益不是简单地少执行了一批 collective。
- Megatron 的 NCCL 活动时间更短，但其中包含跨卡等待，不能直接解释为网络带宽提升。
- 原生 FSDP 的部分 all-gather 拆分拷贝、梯度打包 kernel 在 Megatron 路径中消失。
- 扩大更新范围复测后，从最后一次 autograd 返回到相关 GPU 工作完成为 **1.259 → 1.061 s**，差约 **198 ms**。旧标记确实漏计了大段等待；其中大部分是前面排队的反向工作收尾，不能全部归为 Adam / 裁剪计算的收益。

## 采集条件

4 × H100 80GB，完整 36 层三塔，原始 token 容量、BF16、GA=3、逐 block compile、full activation checkpointing、原始 packing 与生产 loader。两端从同一 `iter_000001200` 恢复完整训练状态和旧归一化；对比原生 PyTorch FSDP2 与 Megatron ZeRO-3。每端运行 5 次更新，前 3 次预热，采集第 **1204、1205** 次更新，每个 rank 一份报告。

两端 **60 个微批**的输入内容、游标与恢复 RNG 记录逐项一致。关闭 checkpoint 写入和 tokenizer 编译，两端保留相同的内容哈希检查。采集 `cuda,nvtx,nccl`，关闭 CPU sampling/context-switch sampling；使用 Nsight Systems **2026.2.1**。

## Nsight App 原生 timeline

以下图片直接截取同版本 Nsight Systems App 的 timeline 区域，保留原生事件、stream 与时间轴；没有用绘图工具重建或修改事件。每张图均可从相应原始 `.nsys-rep` 重现。

### 局部 A：all-gather 后确实有额外拆分拷贝（约 13 ms 窗口）

原生 FSDP2，rank 0，更新 1204，第 1/3 个微批，时间 **5.606–5.619 s**：

![FSDP2 all-gather 后 600.776 微秒拆分拷贝的局部原生 timeline](./images/megatron_fsdp_nsys/local_pytorch_split_copy.png)

看 `Default stream 7` 中鼠标指向的事件：上一笔 all-gather 在 **5.612541 s** 完成，随即执行 `split_with_sizes_copy_out_contiguous_no_cast_kernel`，**5.612554–5.613155 s，600.776 µs**，随后才是该层的计算 kernel。图中另一笔 all-gather 同时在 stream 24 进行，不要把它误读成同一笔通信尚未完成。

这张图支持“原生路径存在真实的参数拆分拷贝”。Megatron 中该 kernel 的调用数为 0，依据全部四 rank 的 kernel 清单；单张截图不能证明整次运行中某个 kernel 不存在。这组整理开销只是部分收益，不能单独解释整步约 1.65 s 的差距。

### 局部 B：同一 decoder block 的参数预取，有无重叠（各约 16 ms 窗口）

两图均来自 **rank 0、更新 1204、第 1/3 个微批、decoder layer_10 的 forward 发射区段**。通过 CPU NVTX layer 范围与 CUDA launch correlation 对齐，比较相同逻辑位置，不按两个运行的绝对时间对齐。

默认 Megatron，时间 **5.690–5.706 s**：

![默认预取下 AllGather 与 GEMM 重叠的毫秒级局部 timeline](./images/megatron_fsdp_nsys/local_megatron_prefetch.png)

看 `Stream 19` 的 AllGather：**5.695972–5.698404 s，2.432 ms**。这段时间内，`Default stream 7` 仍在执行 Triton / GEMM，尤其 **5.697173–5.700220 s** 的 GEMM 与它重叠，计算没有等这笔预取完成才开始。

禁止 lookahead prefetch，时间 **5.729–5.745 s**：

![禁止预取后 AllGather 期间计算 stream 等待的毫秒级局部 timeline](./images/megatron_fsdp_nsys/local_megatron_no_prefetch.png)

看相同两行：AllGather 在 **5.736003–5.738096 s，2.093 ms** 执行时，默认计算 stream 出现空档；通信结束后，后续 Triton / GEMM 才继续。这是预取消融实际改变执行依赖的局部例子。它支持“预取可以隐藏部分通信”，整步收益仍以下文全部 rank 的消融统计为准，不能把这里的约 2 ms 直接乘层数当作净收益。

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
| 旧 `optimizer_step` CPU 标记（窄范围） | 0.054 s | 0.024 s | 有收益，但量级较小 |

两端 FA3 forward/backward 调用数相同，未通过少做重计算获得收益。前五行是 kernel 累计服务时间，不是互斥 wall time。消失的两组整理 kernel 共约 0.17 秒，说明缓冲区布局确实减少了实际 GPU 工作；它只解释一部分收益，不能与前面的 1.650 秒分解再相加。

原生 PyTorch FSDP collectives 的拆分与打包调用，与实际 kernel 名对应；Megatron 使用自己的连续 buffer / 参数视图和通信 pipeline。当前 runtime 开启参数聚合与梯度归约重叠，见 [megatron.py:131](../cosmos_framework/training_backends/megatron.py#L131)。两个 backend 都使用相同的 Cosmos 模型计算路径，并非 Megatron 换用了更快的一套 Attention/MLP 模型。

## 扩大更新范围复测：旧标记漏掉了什么

**旧标记范围不足，已扩大并重新采集两端。** 新范围 `cosmos.optimizer.full_post_autograd` 从最后一个微批的 `loss.backward()` 返回开始，覆盖模型反向收尾、反向后 callbacks、梯度同步上下文退出、全部更新前 callbacks、参数更新、scheduler 与清梯度。子标记进一步区分每个 callback。GPU 完成边界根据 CUPTI launch correlation 和实际执行终点计算，包含范围开始前已经发射但尚未完成的工作；没有额外插入同步改变重叠行为。

仍为三次预热、采集更新 1204/1205，四卡共 8 个观测；60 个微批的输入、游标与 RNG 记录逐项一致。完整更新均值 **40.457 → 38.864 s**，差 **1.593 s**，与上一轮方向一致。两端 loss 均有限，本轮 5 个更新的平均 loss 最大相对差约 0.99%；这是归因补测，不增加正确性验收结论。

| 扩大范围后的指标，每卡每更新均值 | 原生 FSDP2 | Megatron ZeRO-3 | 原生减去 Megatron |
| --- | ---: | ---: | ---: |
| 完整 post-autograd CPU 范围 | 1.237 s | 1.037 s | 200 ms |
| 同一起点，计到相关 GPU 工作完成 | **1.259 s** | **1.061 s** | **198 ms** |
| 范围内，开始前已排队工作的 GPU 活跃并集 | 1.152 s | 0.986 s | 166 ms |
| `SkipNaNStep` CPU 范围，含等待 | 1.154 s | 0.960 s | 194 ms |
| `GradClip` CPU 范围，含 norm 与裁剪 | 21.43 ms | 17.67 ms | 3.75 ms |
| 参数更新及 scheduler CPU 范围 | 59.00 ms | 25.04 ms | 33.96 ms |
| 清梯度 CPU 范围 | 1.74 ms | 6.08 ms | −4.34 ms |
| 保留的旧 `optimizer_step` 窄标记 | 59.22 ms | 25.25 ms | 33.97 ms |

**扩大范围后，差距不是只有约 30 ms，而是约 198 ms。** 但扩大范围内的绝大多数 GPU 活动，来自范围开始前已排队的任务；`SkipNaNStep` 内的同步在等这些反向任务完成。因此，这 198 ms 包含反向执行节奏和排队位置的差异，不能全部归到“优化器算法更快”。裁剪与参数更新的独立 CPU 子区间确实缩短，量级为几毫秒、几十毫秒；CPU 范围缩短也不等于对应 GPU kernel 同幅提速。

这些行互相包含、可能重叠，**不能相加**。本轮 198 ms 也不能再加到上一轮 GPU 空闲 / NCCL-only 的时间分解上；它是另一轮 profile、另一种切分角度。相对于本轮完整更新约 1.593 s 的差异，扩大后的更新尾部仍只覆盖其中一部分。

### 局部 C：反向等待结束 → 梯度裁剪 → 参数更新（各约 166 ms）

两图均为 **rank 0、更新 1204、第 3/3 个微批**。下方保留 Nsight 原生 Events View 的 `GradClip` 选中行及开始/结束时间，避免只依靠较粗的 NVTX 父范围判断。表中均值来自全部四 rank，不要求等于这一个局部样本。

原生 FSDP2，约 **40.660–40.826 s**：

![FSDP2 更新尾部 166 毫秒：同步等待、裁剪与参数更新](./images/megatron_fsdp_nsys/local_pytorch_optimizer.png)

看绿色 `cudaStreamSynchronize` 与上方 GPU 行：主线程等待时，GPU 仍在执行前面排队的 GEMM。等待结束后，Events View 显示 `GradClip` 在 **40.704020–40.729350 s，25.330 ms** 执行；之后 `cosmos.optimizer.update` 才开始。这直接说明旧 `optimizer_step` 开始之前还有真实工作和等待。

Megatron，约 **39.000–39.167 s**：

![Megatron 更新尾部 166 毫秒：同步等待、裁剪与参数更新](./images/megatron_fsdp_nsys/local_megatron_optimizer.png)

相同观察顺序：等待结束，`GradClip` 在 **39.048273–39.064327 s，16.054 ms** 执行，再进入旧 `optimizer_step`。扩大后的 CPU 范围在 **39.086474 s** 结束，相关 GPU 工作到 **39.110777 s** 完成；图右侧 **39.113 s 以后**的 AllReduce 属于 benchmark 计时汇总，不能算作优化器工作。

原始补测数据见 [raw 报告目录](../artifacts/megatron/nsys_optimizer_scope_20261117/raw)，8 份 `.nsys-rep` 均用 Git LFS 保留；[逐区间分析](../artifacts/megatron/nsys_optimizer_scope_20261117/optimizer_analysis.json) 保留 CPU 边界、GPU 完成边界和子阶段时间。8 份报告的 GPU 事件均能找到 launch correlation。范围实现见 [trainer:430](../cosmos_framework/trainer/__init__.py#L430)，callback 标记见 [callback:104](../cosmos_framework/utils/callback.py#L104)，分析入口为 [analyze_optimizer_nsys.py](../tools/analyze_optimizer_nsys.py)。GPU 关联口径与 [NVIDIA NVTX GPU projection 定义](https://docs.nvidia.com/nsight-systems/AnalysisGuide/#nvtx-gpu-proj-trace)一致；已排队工作额外计入等待窗口。

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

有效消融的毫秒级原生 timeline 对比见前文“局部 B”。

可复核统计见 [ablation_comparison.json](../artifacts/megatron/nsys_attribution_20261117/ablation_comparison.json) 与 [消融时间分析](../artifacts/megatron/nsys_attribution_20261117/ablation_no_prefetch_summary.json)。

另保留一次无效尝试：仅给构造器传 `overlap_param_gather=false` 时，MCore 会在 ZeRO-3 初始化中强制改回 true。因此 `ablation_no_param_overlap/` 是重复默认行为的记录，**不能用它证明关闭预取没有影响**。后续有效消融位于 `ablation_no_prefetch/`。

## 原始文件与人工 review

首轮原始报告保留在共享工作区，大文件未纳入 Git；扩大范围补测的 8 份报告另通过上述 Git LFS 目录保留。首轮路径根目录：

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

人工查看建议：用 Nsight Systems 2026.2.1 或兼容版本分别打开两个 `review/` 报告，展开 `CUDA HW → All Streams` 与通信 stream，查看主线程 NVTX 微批。按前文局部图给出的秒数框选，再用 **Shift+Z** 逐级放大至毫秒级；跨 rank 检查请打开对应 `rank1/2/3` 原始文件。原生截图显示的 stream 编号可能不同，应按名称和 NCCL 事件识别，不能假定编号相同。

原始报告与数据包的校验清单见 [raw_reports_manifest.json](../artifacts/megatron/nsys_attribution_20261117/raw_reports_manifest.json) 和 [archive.json](../artifacts/megatron/nsys_attribution_20261117/archive.json)。

版本化的统计、原始报告 SHA256、输入一致性结果保存在 [artifacts/megatron/nsys_attribution_20261117](../artifacts/megatron/nsys_attribution_20261117)。完整 kernel 名及每卡每更新服务时间可查 [summary.json](../artifacts/megatron/nsys_attribution_20261117/summary.json)，逐 rank / update 的并集统计见 [profile_analysis.json](../artifacts/megatron/nsys_attribution_20261117/profile_analysis.json)。

补测的 8 份原始报告同时保存在本仓库共享目录 `artifacts/megatron/nsys_optimizer_scope_20261117/raw/` 与 Git LFS，已逐份核对 SHA256。节点本地完整目录为 `/opt/cosmos-nsys-optimizer-scope/final/`，另含 SQLite、启动日志和输入记录。

## 复现

设置源 checkpoint 和一个新的输出目录，执行：

```bash
BASE_CHECKPOINT_PATH=/path/to/checkpoints/iter_000001200 \
NSYS_ROOT="$PWD/outputs/megatron_validation/my_nsys_run" \
bash tools/profile_megatron_fsdp.sh
```

参数预取消融额外设置 `PROFILE_BACKENDS=megatron PROFILE_PARAM_PREFETCH=0`，使用独立的 `NSYS_ROOT`。工具入口：[采集脚本:1](../tools/profile_megatron_fsdp.sh#L1)、[时间区间分析:1](../tools/analyze_megatron_fsdp_nsys.py#L1)、[统计汇总:1](../tools/summarize_megatron_fsdp_nsys.py#L1)。

本次是四卡、两个稳态更新的诊断，不能替代更长程、多窗口或 192 卡性能归因。Nsight 中 bounded capture 会产生部分初始化 NVTX 范围不完整和 worker 无 NVTX 的提示；核心更新范围与 CUDA kernel 数据已单独检查，详细诊断消息保存在统计文件中。
