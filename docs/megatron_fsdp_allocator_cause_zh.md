# Megatron FSDP：显存分配停顿的原因

此前“首层入口节省 905 ms”只是定位。完整训练的反事实对照进一步确认：**FSDP2 的临时通信缓冲区在独立 CUDA stream 上申请，无法直接复用计算 stream 的大量空闲缓存；物理显存不足时触发缓存回收、stream 同步和重新分配，推迟计算和跨卡通信。Megatron 的连续 bucket、参数视图和缓冲区生命周期减少了这类显存周转。**

## 1. 有空闲缓存，为什么仍然申请失败

PyTorch allocator 按 stream 查找缓存。总缓存有空闲，并不表示发起申请的 stream 有足够大的可复用块。申请新显存失败后，allocator 会回收缓存再重试；回收还可能等待相关 GPU stream 完成。

补采中一次约 **105 MiB** 的申请，申请前张量占用 **48.2 GiB**、allocator 保留 **76.8 GiB**；申请后保留量降到 **48.9 GiB**，`num_alloc_retries` 增加 1。这是失败后的回收重试，最终训练成功，因此不一定看到 OOM 异常。

原始 trace 的一个前向入口中，rank 1 约 440 ms 的停顿包含约 1,750 次 `cuMemUnmap` 和 `cuMemRelease`。晚到的 rank 尚未提交 AllGather，早到的 rank 已在 NCCL kernel 中等待；原报告中 428 ms 的 AllGather，主要耗时来自这种到达时间差。

下图为原生 Nsight App 的局部 multi-view：上方 PyTorch，下方 Megatron，均为 rank 1、更新 1204 的第一个微批。保留 CUDA HW、kernel、CPU CUDA API 和微批/forward NVTX。上方 GPU 空档对应密集的显存管理 API；下方已开始 decoder 计算。静态截图未显示 API 悬浮提示，`cuMemUnmap` / `cuMemRelease` 名称与次数来自原报告查询。上、下视窗分别约 **4.994–5.833 s**、**4.984–5.785 s**；绝对时刻不用于跨运行对齐。各项收益对应的图与阅读位置见[局部 timeline 对照](megatron_fsdp_timeline_evidence_zh.md)。

截图对应原报告的 `gc_observed_v2` 捕获，两份原始文件也收进本轮数据包；文件映射见[截图来源](../artifacts/megatron/nsys_allocator_cause_20261118/figure_sources.json)。

![显存回收局部对照：上方 baseline，下方 Megatron](images/megatron_fsdp_nsys/multiview_allocator_reclaim.png)

C++ 源码按运行环境的 `torch.version.git_version` 获取；Python 源码取自实际 runtime。相关代码及 SHA256 见[源码摘录](../artifacts/megatron/nsys_allocator_cause_20261118/allocator_source_excerpts.json)。

## 2. 只改分配 stream 的完整训练对照

保留 PyTorch FSDP2、模型计算、数据、参数预取、AllGather/ReduceScatter 通信 stream 和 `expandable_segments=True`，**只把 AllGather copy-in 的缓冲区分配与打包改到默认计算 stream**，使其能复用计算缓存。同步依赖继续由同一个 FSDP communication context 管理。

仍使用四张 H100、原始 token 容量、GA=3、原 checkpoint 完整续训，预热三次更新、采集后两次。三组各 60 个微批的输入、游标与恢复 RNG 记录一致。

入口窗口按每次更新的三个微批相加，再对四 rank × 两次更新取均值；整步耗时采用同步后最慢 rank 的计时，两者不混用。

| 项目 | PyTorch 默认 | PyTorch 只改 copy-in stream | Megatron 默认 |
|---|---:|---:|---:|
| 更新耗时，秒；GPU 同步、最慢 rank | 40.902 | 40.068 | 38.863 |
| VAE GPU 完成 → 首个 decoder GEMM，秒/更新 | 3.007 | 1.981 | 1.723 |
| 分配重试，次/卡/更新 | 4.250 | 1.375 | 0 |
| 通信缓冲区申请超过 20 ms 的区间 | 有 | 无 | 无 |
| 显存申请/释放 API 累计 CPU 时间，秒/卡/更新 | 1.237 | 0.227 | 0.061 |

**只改分配 stream，入口窗口缩短约 1.026 秒，整步净缩短约 0.834 秒。** 这验证了缓存复用限制与失败回收确实造成显著开销。它与原先 905 ms 的入口差距属于同一机制，但来自不同轮次，不视为相同的一笔精确计时。

改动也影响重叠：反向 GPU 空闲从约 **408 → 231 ms/更新**，NCCL 独占时间却从约 **70 → 254 ms/更新**，反向总时间基本不变。因此，减少 CPU 分配耗时不一定等量缩短训练；这个诊断改动也不是默认建议启用的生产优化。

本轮增加了 allocator 观测，短窗口耗时有所变化，不将本轮差值除以原先的 1.65 秒来报告贡献百分比。计时见 [allocator 对照](../artifacts/megatron/nsys_allocator_cause_20261118/observed_allocator_analysis.json)、[stream 对照](../artifacts/megatron/nsys_allocator_cause_20261118/copy_in_default_allocator_analysis.json)及[输入核对](../artifacts/megatron/nsys_allocator_cause_20261118/input_comparison.json)。

## 3. 为什么回收重试昂贵

受控微基准固定驻留预算，在默认 stream 留下 10、20、35 GiB 空闲缓存，再申请 4 GiB。对比原 stream 复用与另一 stream 申请；每格为四卡各三次的中位数。它只验证分配机制，不代表训练吞吐。

| 空闲缓存 | Expandable：跨 stream 申请 | 普通 allocator：跨 stream 申请 | 原 stream 直接复用 |
|---|---:|---:|---:|
| 10 GiB | 268 ms | 232 ms | 约 0.02–0.03 ms |
| 20 GiB | 451 ms | 473 ms | 约 0.02–0.03 ms |
| 35 GiB | 717 ms | 812 ms | 约 0.02–0.03 ms |

跨 stream 用例均触发一次失败回收重试，原 stream 用例均未触发。35 GiB 的 expandable 用例实际记录到 **1,792 次 `cuMemUnmap`**，与 20 MiB 大块粒度对应；普通 allocator 则将成本放在 `cudaMalloc`/`cudaFree` 内。表格计的是**申请、失败回收、重试的完整路径**，不是某一个 `cudaFree` 的耗时。

因此，关键是避免缓存失配后的物理显存回收和重新分配。不能仅根据 API 次数，把全部成本归因于 expandable segments 的小块循环。关闭 expandable segments 的完整训练对照在预热阶段申请 1.37 GiB 时 OOM，已标为无效性能运行。

保持原 stream 安排、只切换为 `backend:cudaMallocAsync` 的完整训练对照也在预热阶段 OOM。因此，这两个 allocator 替换都没有形成有效的原容量训练性能结果；当前结论针对缓冲区布局、生命周期与缓存复用，不能据此建议直接替换生产 allocator。

实现见 [benchmark_allocator_reclaim.py](../tools/benchmark_allocator_reclaim.py)，结果见 [micro_comparison.json](../artifacts/megatron/nsys_allocator_cause_20261118/micro_comparison.json)及 [micro_api_comparison.json](../artifacts/megatron/nsys_allocator_cause_20261118/micro_api_comparison.json)。

## 4. Megatron 的实现差别与归因边界

FSDP2 的 AllGather 路径会准备连续输入、申请聚合输出，再将输出复制到各参数的计算存储。MCore 从已有连续 shard 直接 AllGather 到 bucket，并让参数成为该 bucket 的视图。这既减少参数打包/拆包拷贝，也改变了临时缓冲区的申请大小、存活区间与缓存分布。

本轮 Megatron 四卡的观测更新均没有分配重试。这不表示它不申请显存，也不表示启用了持久 double-buffer。两端峰值张量显存接近，收益取决于何时、在哪个 stream、以何种形状复用缓存，不能简化成峰值显存更少。

原报告中约 574 ms 的反向差距包含分配停顿、额外梯度/参数缓冲区拷贝和重叠安排的共同影响。目前已验证缓存回收是主要机制之一；**尚未将这些相互影响的因素分解为可独立相加、恰好覆盖 574 ms 的因果份额**。原先 175 ms 的拷贝独占时间差、242 ms 的预取消融差值，也不能直接加到本轮的 834 ms 上。

## 5. 原始文件与复现

本轮原始报告保留在以下目录，训练与微基准分开存放：

```text
outputs/megatron_validation/nsys_allocator_cause_20261118/
  observed/{pytorch,megatron}/rank[0-3].nsys-rep
  copy_in_default/pytorch/rank[0-3].nsys-rep
  micro/{expandable,native}/rank[0-3].nsys-rep
  native/INVALID_RUN.json  # 完整训练 OOM，不用于性能结论
  async/INVALID_RUN.json   # cudaMallocAsync 完整训练 OOM
```

共 **20 份新增有效报告：12 份完整训练、8 份分配微基准**。同目录的 `nsys_allocator_cause_raw.tar` 约 237 MiB，还包含截图对应的两份旧捕获，共 22 份 `.nsys-rep`，以及运行记录、失败日志和 SHA256 清单。路径与校验值见[数据包索引](../artifacts/megatron/nsys_allocator_cause_20261118/archive.json)，逐报告信息见[原始文件清单](../artifacts/megatron/nsys_allocator_cause_20261118/raw_reports_manifest.json)。

训练入口为 [profile_megatron_fsdp.sh](../tools/profile_megatron_fsdp.sh)：观测设置 `PROFILE_ALLOCATOR=1 PROFILE_PYTHON_GC=1`；stream 对照额外设置 `PROFILE_BACKENDS=pytorch PROFILE_COPY_IN_DEFAULT_STREAM=1`，使用不同的 `NSYS_ROOT`。微基准入口为 [profile_allocator_reclaim.sh](../tools/profile_allocator_reclaim.sh)。两者都需要空闲四卡；训练还需要 `BASE_CHECKPOINT_PATH`。
