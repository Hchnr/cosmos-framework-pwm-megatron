# Megatron FSDP 收益的局部 timeline 对照

以下均为已有 Nsight Systems App 截图，没有重新运行训练。每张图上方为 PyTorch FSDP2 baseline，下方为 Megatron FSDP；保留 NVTX 和 CUDA HW。上下按同一逻辑阶段比较，不能直接比较两次运行的绝对时钟。截图是局部示例，整更新的 905 ms、574 ms 来自全部四 rank、两个更新的统计，不能从单张图直接读出。

## 1. 显存回收期间，GPU 工作队列耗尽

两端均为 rank 1、更新 1204、第一个微批的 forward。上方约 4.994–5.833 s，下方约 4.984–5.785 s。

![显存回收及 GPU 空档：上 baseline，下 Megatron](images/megatron_fsdp_nsys/multiview_allocator_reclaim.png)

看上方约 5.17–5.61 s：CUDA API 轨道有密集的显存管理调用，而 CUDA HW / All Streams 出现长空档；下方对应入口进入连续计算。API 名称和次数需放大原报告核对，截图自身不能读出全部调用名。该区间的原报告查询记录约 1,750 次 `cuMemUnmap` / `cuMemRelease`，Python 栈对应 AllGather 输出缓冲区申请。

这支持“主机在缓冲区分配/回收路径停顿，GPU 没有后续任务”，而非模型算子本身变快。跨 stream 缓存失配的因果判断还依赖[只改 copy-in stream 的对照](megatron_fsdp_allocator_cause_zh.md#2-只改分配-stream-的完整训练对照)：入口缩短 1.026 s，整步缩短 0.834 s；这是另一轮诊断结果，不能当成原 905 ms 的精确份额。

## 2. 慢 rank 延迟提交，其他 rank 在 AllGather 中等待

两端均为 rank 0、更新 1204、第一个微批的 forward。上方约 5.13–5.96 s，下方约 4.88–5.71 s。

![AllGather 等待：上 baseline，下 Megatron](images/megatron_fsdp_nsys/multiview_first_block_entry.png)

上方 AllGather kernel 从 5.156025 s 持续至 5.584144 s，约 428 ms，计算 stream 没有进入后续 decoder 计算。下方对应入口很快进入计算。截图展示长通信区间；[跨 rank 时间记录](../artifacts/megatron/nsys_full_attribution_20261117/first_allgather_cross_rank.json)进一步确认：最后一个 rank 在对齐后的 5.582146 s 才进入，约 2 ms 后该 collective 完成。因此长区间主要是等待其他 rank 到达，不能解释成实际数据传输需要 428 ms。

图 1 和图 2 是同一类停顿在不同 rank 上的表现，不是两笔可以相加的收益。它们支持原约 900 ms 入口差距的原因链：申请/回收停顿 → collective 晚提交 → 其他 rank 等待 → decoder 延迟开始。

## 3. 反向入口减少了 GPU 空闲

两端均为 rank 0、更新 1204、第一个微批。上方约 7.72–7.95 s，下方约 7.23–7.46 s，NVTX 显示 forward → backward。

![反向入口空档：上 baseline，下 Megatron](images/megatron_fsdp_nsys/multiview_backward_entry.png)

上方 7.797970–7.892470 s，约 94.5 ms，所有 GPU stream 都没有 kernel 或 memop；下方对应逻辑阶段持续执行计算。该例不是 dataloader 或 optimizer 区间。

截图直接证明 GPU 空档；空档与分配停顿的对应关系来自同一报告的 Python 栈采样：`empty_strided_cuda` 分配驻留约 7.744–7.892 s，前半段仍与已排队 GPU 工作重叠。不能把整个约 148 ms 的 CPU 驻留都当成 GPU 损失。全部区间匹配见 [backward_allocation_idle_overlap.json](../artifacts/megatron/nsys_full_attribution_20261117/backward_allocation_idle_overlap.json)。

原反向 574 ms 差距中，408 ms 是 GPU 空闲减少，161 ms 是非 NCCL 活跃时间减少，5 ms 是 NCCL-only 减少。此图是空闲减少的一个局部实例，不能单独证明全部 408 ms 均由分配造成，也不能证明全部 574 ms 的独立因果份额。

## 4. 参数聚合后，baseline 多了一次拆分拷贝

两端均为 rank 0、更新 1204、第一个微批的 decoder 首层入口。上方约 5.579–5.592 s，下方约 5.135–5.147 s。

![AllGather 后参数整理：上 baseline，下 Megatron](images/megatron_fsdp_nsys/multiview_allgather_copy.png)

上方第一笔 AllGather 在 5.584144 s 完成；Default stream 随后先执行 `split_with_sizes_copy_out_contiguous_no_cast_kernel`（5.584161–5.584765 s，约 603 µs），才进入归一化 / GEMM。下方对应路径直接进入计算。上方同时存在下一笔 AllGather，不要将其误认成前一笔仍未结束。完整 kernel 名与精确边界来自原报告查询，截图中的名称有截断。

这说明额外参数整理工作确实存在。它是前向局部示例，不能拿来直接证明反向 161 ms 全是拷贝收益。全更新三类整理 kernel 的独占活跃时间差约 175 ms，也不能全部归入反向。统计及口径见[原报告第 4 节](megatron_fsdp_nsys_attribution_zh.md#4-确实减少了缓冲区整理工作)。

## 原始数据

四张图均来自 `outputs/megatron_validation/nsys_full_attribution_20261117/gc_observed_v2/{pytorch,megatron}/rank{0,1}.nsys-rep`。文件和校验值见[原始报告清单](../artifacts/megatron/nsys_full_attribution_20261117/raw_reports_manifest.json)；显存回收图的两份 rank 1 报告另见[精确截图来源](../artifacts/megatron/nsys_allocator_cause_20261118/figure_sources.json)。原始 `.nsys-rep` 保持不变，可在 Nsight App 按上述秒数放大 review。
