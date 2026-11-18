# Megatron-FSDP 四卡性能收益归因

测试日期：2026-11-17。比较 Cosmos 原生 PyTorch FSDP2 与当前 Megatron-FSDP ZeRO-3；TP/PP 尚未开始实现。

## 结论

**主要差异出现在分片缓冲区准备、显存分配时的主机停顿，以及由此造成的跨卡等待；优化器更新不是主要来源。** Attention 和主要 decoder GEMM 没有明显提速。

首轮 profile 的完整更新差值 **1.650 s** 中，**0.905 s 在 VAE 完成后至首个 decoder GEMM 之间，0.574 s 在反向及重计算阶段**。补采 Python 栈后，两处都找到了具体的分配停顿；GPU 拆分/打包拷贝也有减少。参数预取消融另测得约 **0.242 s** 的整步影响。它们存在重叠，不能直接相加成独立优化贡献。

**2026-11-18 根因补测：**完整训练中只把 FSDP2 参数打包移到计算 stream，分配重试从 **4.250 → 1.375 次/卡/更新**，VAE 后入口窗口缩短约 **1.026 秒**，整步净缩短约 **0.834 秒**，验证了跨 stream 缓存复用限制和失败回收的因果作用。受控分配实验进一步确认，关键是避免回收重分配，不能仅归因于 expandable segments 的 API 次数。完整解释、局部 multi-view 和原始文件见[根因补充报告](megatron_fsdp_allocator_cause_zh.md)。

关闭自动 Python GC 的四卡对照仍保留 **1.435 s/更新**的后端差距，GC 不是主要解释。此前无 profiler 的短程 ABBA 测试吞吐提升为 **4.48%**；本报告分析该训练路径的来源，不把两个采样更新当作长期或 192 卡性能结论。

## 1. 先把 1.650 秒完整对账

下表来自**首轮两个完整更新 × 四 rank，共 8 个观测的均值**。原生 FSDP2 为 **40.728 s**，Megatron 为 **39.078 s**。使用 CUDA launch correlation 找到各逻辑阶段的实际 GPU 执行边界，每个时间点只归入一行；反向最后一个非 NCCL GPU 任务完成后进入最后一行，剩余通信尾部计入最后一行。

| GPU 执行阶段 | 原生减去 Megatron |
| --- | ---: |
| 前向准备，至 VAE GPU 工作完成 | 19 ms |
| **VAE 完成 → 首个 decoder GEMM**，含参数准备和等待 | **905 ms** |
| decoder 前向 | 76 ms |
| 输出头与 loss | 2 ms |
| **反向及重计算**，计至其非 NCCL GPU 工作完成 | **574 ms** |
| 输入及微批之间的过渡 | 59 ms |
| 最后一次反向计算完成 → 更新结束，含剩余通信 | 15 ms |
| **合计** | **1,650 ms** |

首个 decoder 边界取 `layer_00` 中首个 `nvjet` GEMM；该层之前的归一化等少量工作计在第二行。不能把第二行简称为“VAE 加速”：VAE 本身只相差约 **19 ms**。同样口径应用于扩大 optimizer 标记的另一轮 profile，首层入口与反向分别相差 **852 ms、510 ms**，主要位置一致。

逐 rank / update 的边界和闭合检查见 [semantic_windows.json](../artifacts/megatron/nsys_attribution_20261117/semantic_windows.json)，实现见 [analyze_megatron_semantic_windows.py:1](../tools/analyze_megatron_semantic_windows.py#L1)。上述秒数定位差距发生的位置，尚不是每个优化开关的独立因果效应。

## 2. 首层入口：申请 AllGather buffer 的慢 rank 拖住其他卡

下面均为 Nsight Systems App **multi-view 原生上下停靠截图**：**上方 baseline，下方 Megatron**。Events View 已收起，展开 CUDA HW、All Streams、计算与通信 stream，并保留 NVTX 的更新号、微批号和 forward/backward 标签。上下按**相同逻辑阶段**选取局部窗口；各报告的绝对秒数不同。

### 约 800 ms：首个微批进入 decoder

补采默认策略，均为 **rank 0、更新 1204、micro_01_of_03**：上方约 **5.13–5.96 s**，下方约 **4.88–5.71 s**。

![上 baseline 下 Megatron：首层入口的局部 GPU、通信及 NVTX 对照](./images/megatron_fsdp_nsys/multiview_first_block_entry.png)

上方 `Stream 24` 的 AllGather kernel 执行 **5.156025–5.584144 s，428.118 ms**，期间默认计算 stream 长时间没有后续计算。跨 rank 对齐后，同一 collective 的 GPU 进入时间如下：

| rank | 对齐到 rank 0 时钟的进入时间 |
| --- | ---: |
| 0 | 5.156025 s |
| 1 | 5.580824 s |
| 2 | 5.581237 s |
| 3 | 5.582146 s |

**最后一个 rank 到齐前已经过去约 426 ms；到齐后约 2 ms 就完成了。** 因而这次 428 ms 不能解释为传输本身需要 428 ms。下方 Megatron 对应入口的 AllGather 是毫秒级事件，随后进入连续计算。

补采的 **250 Hz Python 周期采样**进一步找到慢 rank 的位置：baseline rank 1 在本地时间 **5.175–5.615 s** 的连续采样驻留于：

```text
DefaultAllocMixin.allocate                  _fsdp_collectives.py:59
  return torch.empty(*size, dtype=dtype, device=device)
foreach_all_gather                         _fsdp_collectives.py:262
unshard → pre_forward
```

相关 runtime 源码摘录及文件 SHA256 见 [runtime_source_excerpts.json](../artifacts/megatron/nsys_full_attribution_20261117/runtime_source_excerpts.json)。

rank 2、3 同一入口也有约 **440 ms** 的连续分配采样。这形成了明确链路：**AllGather 输出缓冲区分配时主机停顿 → collective 发射较晚 → 其他 rank 在 NCCL 中等待 → 后续计算无法开始。** CPU 分配停顿、NCCL-only 和 GPU 空闲可能是同一问题的不同表现，不能分别算三次收益。

进一步逐一核对两个更新的 6 个入口：

| 更新 / 微批 | 该入口的后端耗时差 | baseline 最长连续分配采样跨度，取一个 rank |
| --- | ---: | ---: |
| 1204 / 1 | 451 ms | 444 ms |
| 1204 / 2 | 227 ms | 176 ms |
| 1204 / 3 | 291 ms | 269 ms |
| 1205 / 1 | 227 ms | 221 ms |
| 1205 / 2 | 212 ms | 160 ms |
| 1205 / 3 | 373 ms | 364 ms |
| **按两个更新平均** | **891 ms/更新** | **817 ms/更新** |

右列按每个逻辑入口取最长的一段，只统计 VAE GPU 完成之后、首个 decoder GEMM 之前，不跨 rank 累加。它与左列量级接近，支持分配停顿是前向入口差距的主要来源；二者仍不是可以直接相减或相加的独立优化量。补采的 891 ms 与首轮的 905 ms 来自不同运行，不能混作同一组精确计时。逐入口记录见 [forward_entry_allocation.json](../artifacts/megatron/nsys_full_attribution_20261117/forward_entry_allocation.json)。

跨卡时间见 [first_allgather_cross_rank.json](../artifacts/megatron/nsys_full_attribution_20261117/first_allgather_cross_rank.json)；采样栈和长驻留区间见 [python_samples.json](../artifacts/megatron/nsys_full_attribution_20261117/python_samples.json)。采样区间不是精确函数入口/出口计时，不能把每个 rank 的驻留时间相加。后续已通过 allocator 计数器、CUDA 显存管理 API、源码和改变分配 stream 的完整训练对照，确认失败回收与跨 stream 缓存复用限制，见[根因补充报告](megatron_fsdp_allocator_cause_zh.md)。

## 3. 反向：分配停顿使 GPU 队列耗尽

### 约 220 ms：第一个微批从 forward 进入 backward

同为补采默认策略、**rank 0、更新 1204、micro_01_of_03**；上方约 **7.72–7.95 s**，下方约 **7.23–7.46 s**，对齐各自 forward GPU 工作结束的位置。

![上 baseline 下 Megatron：反向入口的分配停顿、GPU 空档及 NVTX 参照](./images/megatron_fsdp_nsys/multiview_backward_entry.png)

上方 baseline 在 **7.797970–7.892470 s** 出现 **94.5 ms** 的空档，所有 GPU stream 都没有 kernel 或 memop；下方对应阶段持续计算。两栏 NVTX 都能看到 `forward → backward`，这里不是输入加载或 optimizer 阶段。

该空档内的 Python 采样驻留于反向编译代码的 `empty_strided_cuda` 分配语句。rank 0 的连续分配采样跨 **7.744–7.892 s**，前半段与已排队计算重叠，随后 GPU 队列耗尽，才形成上述 94.5 ms 空档。**不能把整个 148 ms 主机驻留都当作 GPU 损失。** baseline 其他微批和 rank 也出现此类分配驻留；Megatron 四 rank 的本轮采样中，没有观察到这两类分配路径连续驻留超过 20 ms 的区间。

首轮反向阶段的 **574 ms** 差值，按 GPU 状态拆开为：**空闲减少 408 ms、非 NCCL 活跃时间减少 161 ms、NCCL-only 减少 5 ms**。分配停顿是空闲的具体来源之一，缓冲区整理 kernel 则参与非 NCCL 工作差异；本轮没有逐个禁用这些行为，不能声称 574 ms 全部来自一个开关。

补采还对全部四 rank 做了交叉检查：反向 GPU 空闲为 **430 → 64 ms/更新**，减少约 **366 ms**；baseline 其中约 **253 ms/更新**与长分配采样区间重合，已扣除 GC NVTX 覆盖部分，Megatron 对应重合为 0。这里只匹配连续采样跨度至少 20 ms 的区间；剩余短空档未继续按 CPU 子路径分摊，因此不把全部反向空闲都声称为某一次分配的收益。逐区间结果见 [backward_allocation_idle_overlap.json](../artifacts/megatron/nsys_full_attribution_20261117/backward_allocation_idle_overlap.json)，复算脚本为 [analyze_megatron_allocation_windows.py:1](../tools/analyze_megatron_allocation_windows.py#L1)。

## 4. 确实减少了缓冲区整理工作

### 各约 12 ms：同一首层的 AllGather → 计算

同为补采默认策略、**rank 0、更新 1204、micro_01_of_03、decoder 首层入口**；上方约 **5.579–5.592 s**，下方约 **5.135–5.147 s**。

![上 baseline 下 Megatron：首层参数聚合后拷贝与计算的毫秒级对照](./images/megatron_fsdp_nsys/multiview_allgather_copy.png)

看上方 `Default stream 7`：第一笔聚合在 **5.584144 s** 完成，随后 `split_with_sizes_copy_out_contiguous_no_cast_kernel` 执行 **5.584161–5.584765 s，603.299 µs**，再进入 Triton / GEMM。上方同时还有下一笔 AllGather，不能把它误认成前一笔尚未完成。下方聚合后直接进入对应归一化 / GEMM 路径，后续预取与计算有重叠。

首轮全部四 rank 的 kernel 清单给出：

| 整理 kernel，每卡每更新平均 | 原生 FSDP2 | Megatron |
| --- | ---: | ---: |
| AllGather 输出拆分 `split_with_sizes_copy…` | 219 次，136 ms | 0 次 |
| 梯度 ReduceScatter 输入整理 `chunk_cat_cuda_kernel` | 111 次，35 ms | 0 次 |
| BF16→BF16 的 `multi_tensor_apply…Copy` | 1,128 次，57 ms | 0 次 |

第三行包含原生 FSDP2 AllGather 输入的 `torch._foreach_copy_` 打包路径，不能笼统叫作优化器或梯度更新。三类 kernel 累计服务时间约 **228 ms**；扣掉其与其他非 NCCL 工作重叠的部分，**仅这些整理工作占用的非 NCCL 活跃区间约为 175 ms**。这仍是观测开销，不是关闭这些 kernel 后整步必然快 175 ms 的独立消融结果。

由此可把首轮 1.650 秒再按互斥 GPU 状态完整对账：

| 时间分区的差值 | 原生减去 Megatron |
| --- | ---: |
| GPU 无 kernel / memop 的空闲 | 939 ms |
| 只有 NCCL、没有其他 GPU 工作 | 471 ms |
| 仅上述三类整理工作占用的非 NCCL 活跃区间 | 175 ms |
| 其余非 NCCL 活跃区间 | 65 ms |
| **合计** | **1,650 ms** |

这是与第一节相同的总时间、另一种切分角度，**不能把两张表再相加**。统计见 [buffer_copy_occupancy.json](../artifacts/megatron/nsys_attribution_20261117/buffer_copy_occupancy.json)。

## 5. 消融：哪些解释成立，哪些不是主因

### 自动 GC 不是主要来源

两端保持相同 checkpoint、输入、Nsight 和 Python 采样设置；实验组只在完成预热更新 **1203** 后调用 `gc.disable()`，保留生产手动 GC，不主动提前回收。

| 两个采样更新的最慢 rank 计时均值 | 默认策略 | 预热后关闭自动 GC |
| --- | ---: | ---: |
| 原生 FSDP2 | 40.462 s | 40.343 s |
| Megatron | 38.884 s | 38.908 s |
| **后端差距** | **1.578 s** | **1.435 s** |

默认 baseline rank 1 确有一次 **677 ms** 的第 2 代 GC，但很多时间与 GPU 已排队计算重叠。关闭自动 GC 后，后端差距仅缩小约 **143 ms**；这两个采样更新的对照排除了“收益主要来自这次 GC”的解释，不能把 677 ms 原值算成整步损失。

生产配置为 `every_n=5, warm_up=1, gc_level=1`，从 1200 恢复后，原回调在完成更新 **1205** 时才首次禁用自动 GC；因此“预热了三步”不代表这两个采样更新已关闭自动 GC。实现见 [manual_gc.py:30](../cosmos_framework/callbacks/manual_gc.py#L30)、[every_n.py:54](../cosmos_framework/callbacks/every_n.py#L54)。实验组仍有生产手动回收，不能称为“完全无 GC”。

四组各 **60 个微批**的输入内容、游标和恢复 RNG 记录逐项一致，且与首轮 60 条记录也逐项一致（[跨轮核对](../artifacts/megatron/nsys_full_attribution_20261117/input_cross_round_comparison.json)）；原始计时和 GC 记录见 [comparison.json](../artifacts/megatron/nsys_full_attribution_20261117/comparison.json)。这是短窗口诊断，不把 143 ms 当作精确长期因果份额。

### 参数预取有帮助，但不能解释全部差距

直接拦截 `AllGatherPipeline.all_gather_params` 的 lookahead 请求，默认 **39.091 s**，禁用预取 **39.333 s**，整步慢约 **242 ms / 0.62%**。实际拦截 1,106 次请求，60 个微批输入记录一致。NCCL 与计算重叠从 **2.666 s** 降到 **0.022 s**，但同时非 NCCL 活跃时间也变化，不能把全部重叠时间当作净收益。见 [ablation_comparison.json](../artifacts/megatron/nsys_attribution_20261117/ablation_comparison.json)。

仅传 `overlap_param_gather=false` 的旧尝试会被 MCore ZeRO-3 强制改回 true，已标为无效对照。当前实际启用参数聚合和梯度归约重叠，见 [megatron.py:144](../cosmos_framework/training_backends/megatron.py#L144)。当前未启用 MCore 的持久 double-buffer 或 NCCL userbuffer，不能将收益归因于这些未开启的开关；临时 bucket 仍会分配和释放 storage。

### 优化器尾部不是 1.65 秒的主要来源

扩大 `full_post_autograd` 范围后，从最后一次 autograd 返回计到相关 GPU 工作完成为 **1.259 → 1.061 s**，差 **198 ms**；裁剪 CPU 范围 **21.43 → 17.67 ms**，参数更新 CPU 范围 **59.00 → 25.04 ms**。其中大量时间在等此前排队的反向工作，与第一节反向阶段重叠。按最后一个非 NCCL 反向 GPU 任务完成后再切分，首轮真正的最终尾部只相差 **15 ms**，因此不能用这 198 ms 解释整步主要收益，也不能再加到反向收益上。

范围和各子阶段见 [optimizer_analysis.json](../artifacts/megatron/nsys_optimizer_scope_20261117/optimizer_analysis.json)，范围实现见 [trainer:430](../cosmos_framework/trainer/__init__.py#L430)。

### 核心计算基本相同

首轮主要 FA3 forward 为 **2.740 → 2.722 s**，backward 为 **3.653 → 3.657 s**；调用次数分别同为 216、108。代表性 decoder GEMM 为 **3.485 → 3.455 s**，同为 432 次。这里是 kernel 累计服务时间；没有观察到换用更快 Attention、少做重计算或主要 GEMM 大幅加速的现象。完整清单见 [summary.json](../artifacts/megatron/nsys_attribution_20261117/summary.json)。

## 6. 条件、原始文件与人工 review

4 × H100 80GB，完整 36 层三塔，生产 token 容量、BF16、GA=3、逐 block compile、full activation checkpointing、原始 packing 与生产 loader。两端从 `iter_000001200` 恢复完整训练状态和 checkpoint 旧归一化，关闭 checkpoint 写盘与 tokenizer 编译。每端运行 5 次更新，前三次预热，采集 **1204、1205**；两端都保留输入哈希检查。Nsight Systems **2026.2.1**，采集 `cuda,nvtx,nccl`，关闭 CPU scheduling sampling；补采增加 **250 Hz Python periodic sampling** 和 GC NVTX。

本次补采的 **16 份原始 `.nsys-rep`** 均保留在共享目录，未修改报告内容，合计约 **306 MB**：

```text
/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron/outputs/megatron_validation/nsys_full_attribution_20261117/
  gc_observed_v2/{pytorch,megatron}/rank{0,1,2,3}.nsys-rep
  gc_disabled/{pytorch,megatron}/rank{0,1,2,3}.nsys-rep
  review/baseline_pytorch_rank0.nsys-rep
  review/megatron_rank0.nsys-rep
```

截图使用 `gc_observed_v2`；两个 `review/` 文件是相同字节的可读文件名副本。每份文件的路径、大小和 SHA256 见 [raw_reports_manifest.json](../artifacts/megatron/nsys_full_attribution_20261117/raw_reports_manifest.json)。完整数据包为同一根目录下的 `nsys_full_attribution_raw.tar`，包含全部 16 份报告与 SHA256 清单；见 [archive.json](../artifacts/megatron/nsys_full_attribution_20261117/archive.json)。对应目录另有 SQLite、训练日志、输入与 GC 记录。首次缺少媒体路径别名的失败运行位于 `gc_observed/`，不用于任何结果。

首轮 profile / 参数预取消融原始目录仍为 `outputs/megatron_validation/nsys_attribution_20261117/`，全部 16 份报告另打包在 `nsys_raw_reports.tar`，约 218 MB；见 [首轮文件清单](../artifacts/megatron/nsys_attribution_20261117/raw_reports_manifest.json)。扩大 optimizer 范围的 8 份报告保留于 [Git LFS raw 目录](../artifacts/megatron/nsys_optimizer_scope_20261117/raw)。所有原始文件均保留，文档展示改为上述局部 multi-view。

人工 review 时分别打开两个 `review/` 报告，将 Megatron 标签拖出后，从浮动窗口标题栏拖回主窗口的**下方停靠箭头**，baseline 放上方；收起两栏 Events View，展开 CUDA HW / All Streams 与 CPU NVTX。按各图的独立秒数框选，使用 **Shift+Z** 放大。上下分别保留原报告时间轴，不按两个不同运行的绝对时钟强行对齐。操作对应 [NVIDIA 多报告分栏说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-timeline-views)。

复现默认采集：

```bash
BASE_CHECKPOINT_PATH=/path/to/checkpoints/iter_000001200 \
NSYS_ROOT="$PWD/outputs/megatron_validation/new_capture" \
PROFILE_PYTHON_GC=1 PROFILE_PYTHON_SAMPLING=1 \
bash tools/profile_megatron_fsdp.sh
```

GC 对照另设 `PROFILE_DISABLE_AUTO_GC=1` 并使用新输出目录；参数预取消融另设 `PROFILE_BACKENDS=megatron PROFILE_PARAM_PREFETCH=0`。工具入口：[采集:1](../tools/profile_megatron_fsdp.sh#L1)、[GPU 并集统计:1](../tools/analyze_megatron_fsdp_nsys.py#L1)、[缓冲区拷贝统计:1](../tools/analyze_megatron_buffer_copy.py#L1)、[逻辑阶段切分:1](../tools/analyze_megatron_semantic_windows.py#L1)、[Python 栈分析:1](../tools/analyze_megatron_python_samples.py#L1)、[GC 观测及对照:1](../tests/megatron/python_gc_trace.py#L1)。
