# 52 源预训练：慢步与具体样本分析

核对时间：2026-10-30T21:15:45+08:00。任务 `job-93f37403-b199-413b-a201-25c441e1a976`；样本分析范围为 rank0，第 1230、1234–1237 步。

结论：近期严重慢步的主要增量不在 rank0 领取 batch 的区间。存在明确的样本级视频解码成本差异，但慢步输入大小与正常步接近；线上主要增量在 forward 或包含同步的 optimizer 区间，并与节点内存压力/内核开销同时出现。尚不能由现有共享日志确定哪个远端 rank 的哪个调用最先受阻，也不能断言内存压力的具体分配器、NUMA 或文件系统根因。

## 线上计时：取数之后仍可能处理视频

以下取数/forward/backward 均已乘 GA3，表示一个 optimizer step 的三次主机计时合计；optimizer 区间每步一次。它们不是纯 GPU kernel 时间，也不覆盖全部 callback、H2D 和日志开销。forward 包含 `_get_training_inputs` 的视频/VAE等处理和模型计算，不能把 forward 全部当成 Transformer。代码见 `cosmos_framework/trainer/__init__.py:329`、`:434`、`:455`，`cosmos_framework/model/generator/omni_mot_model.py:1418`。

| 步 | 整步秒 | 取 batch 秒 | forward 秒 | backward 秒 | optimizer 区间秒 |
|---|---:|---:|---:|---:|---:|
| 1230 | 34.86 | 0.013 | 14.70 | 13.92 | 3.27 |
| 1234 | 84.54 | 0.019 | 58.51 | 18.75 | 1.90 |
| 1235 | 100.01 | 0.113 | 40.46 | 16.52 | 37.22 |
| 1236 | 108.25 | 0.070 | 63.80 | 22.04 | 4.40 |
| 1237 | 86.15 | 0.205 | 57.36 | 19.08 | 4.06 |
| 1239 | 127.47 | 0.193 | 80.90 | 29.78 | 2.09 |
| 1241 | 159.56 | 0.013 | 14.57 | 13.63 | 121.67 |
| 1247 | 32.85 | 0.012 | 14.39 | 13.38 | 2.38 |

例如 1235 步总计 100.01 秒，领取三个 batch 只用约 0.113 秒。至少对 rank0，不能用“worker 没及时供上数据”解释其余约 100 秒。其他 rank 的 >=10 秒 receive 日志也未出现，但这不能排除小于阈值或 receive 之外的停顿。原始计时见 [W&B API 数据](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/wandb_phase1230_to1250.json)。

## 样本如何还原、实际消费了哪些

使用实际 rank0 日志中的每步 committed_global_id 与每源 epoch/index 水位，结合生产 EpisodeBlockDistributor 的 seed42、episode block shuffle、DP rank0/world192，映射为原始 dataset index、episode/window。12 个来源的生产构造指纹与实际 1200 步 checkpoint 全部匹配；第二份只读元数据重建与生产构造得到的 135 个映射完全一致。再实际调用生产 get_item 和首帧 color jitter，核对样本 key、有效性与线上 effective_samples。不是将同一时刻后台预取的慢读文件当作当前样本。代码见 `cosmos_framework/data/generator/catalog.py:353`、`cosmos_framework/data/generator/action/unified_action128.py:760`。

第 1235 步的三个 microbatch 为：

| microbatch | 来源 | episode/window | episode 内时间窗口 | 样本数 | 单样本本机 CPU 中位秒 |
|---|---|---|---|---:|---:|
| 3702 | Galbot-joint:31 | 55487/4–11 | 8–24s | 8 | 1.74 |
| 3703 | RoboMIND2.0-joint:37 | 26336/2–9 | 4–19.02s | 8 | 5.36 |
| 3704 | Galaxea-joint:32 | 5591/45–57 | 90–116s | 13 | 2.28 |

全部样本的精确数据文件、视频绝对路径、时间窗口、形状和分段计时在 [sample_inventory.csv](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/sample_inventory.csv)；[映射核对](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/sample_mapping_validation.json)记录指纹及微批次顺序。表中的 key 需与 source/root 一起使用，不同来源可能同名 joint/eef。

## 昂贵样本：RoboMIND2.0 长 GOP

具体样本是 `RoboMIND2.0-joint:37`、`joint/26336/2`，dataset index299289，episode 内 4–6 秒；三路原视频均 640×480、50 FPS，训练视频采样上限30 FPS。原视频 main 的关键帧大致间隔5秒，读取4秒起点时须从0秒关键帧补解约4秒。包级只读检查确认 main 的 seek preroll 为4秒；解码器该次最多每组解250帧，另两路为107/242帧。Galbot 对照视频关键帧间隔约0.0667秒、该样本每组最多4帧。

本机首次 `joint/26336/2` 总处理9.86秒，其中视频解码7.27、缩放0.44、Parquet episode提取0.53、caption索引/读取1.40秒；后续同 episode 样本中位约5.36秒。Galbot `joint/55487/4` 首次总处理2.34秒，其中解码1.29秒。这里确实有视频编码结构造成的 CPU 开销差异；长 GOP 并不代表坏数据，也不意味着该次处理在训练领取 batch 时才开始。证据：[关键帧检查](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/representative_video_gops.json)、[逐样本重放](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/local_sample_replay.jsonl)。

第1235步三个代表性 Parquet 文件仅约5.2–6.2 MB，所在 row group 的全部列未压缩约19.7–28.6 MB；并不是此前 M1 的1.8GB大 group。见 [Parquet footer](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/representative_parquet_metadata.json)。

## 完整重放与大小对照

| 步 | 线上整步秒 | 候选/有效样本 | 原始视频 GiB（含候选） | 本机逐样本 CPU 串行合计秒 | 单样本最大秒 |
|---|---:|---:|---:|---:|---:|
| 1230 | 34.86 | 27/27 | 1.820 | 63.77 | 10.88 |
| 1234 | 84.54 | 32/31 | 1.890 | 45.72 | 7.44 |
| 1235 | 100.01 | 29/29 | 1.833 | 92.64 | 9.86 |
| 1236 | 108.25 | 23/23 | 1.831 | 54.63 | 6.70 |
| 1237 | 86.15 | 24/24 | 1.797 | 71.18 | 8.82 |

135 个候选已全部实际重放，无读取/解码错误；134 个有效。测试为本机单进程、Torch CPU线程1，包含生产动作/状态提取、视频解码、缩放、caption/tokenize、首帧color jitter；不含 collation/IPC/pin/H2D/VAE/模型或线上多进程压力。串行合计不是线上步时，也不是一个 worker 池的等待；不同来源和首次调用的缓存状态不同，不能把上表当严格提速 A/B。没有复现样本自身持续几十到几百秒的确定性阻塞。

1234步的32个候选中，`omnivitac/9038/13` 是26.000–26.004秒的4毫秒尾窗口，实际时间范围内动作行数为0，生产代码在视频解码前正常丢弃，仅耗0.00149秒。其余31个有效样本与线上计数一致；它不能解释该步84.54秒的慢。见 [尾窗口检查](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/dropped_tail_sample.json)。

## 内存压力与全局同步

按实际 step 时间窗口对齐 cgroup 诊断，1230步主节点 memory PSI full 停顿比例约0.155%，内核态CPU消耗约9.14核；1234步约65.95%、37.24核；1239步约75.16%、69.04核，并有页扫描/回收计数增长。1241步主节点压力已回落至约0.14%，但 worker2 同期仍约67.87%，rank0 optimizer区间变成121.67秒。该模式符合本节点慢时 forward 受拖累、其他节点慢时在同步点等待，但没有直接堆栈/全rank阶段计时，尚不能认定具体阻塞调用。原始对齐见 [内存/内核统计](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/memory_reclaim_alignment.json)。

这里的65%–75%是 memory.pressure full累计计数增量除以所采样时间，不是内存占用率，也不是GPU空闲比例或某个训练线程的精确等待比例。占用高本身不必然慢；可回收页较多时可以保持正常。发生内存分配/缺页/回收等等待时，即使尚未触及memory.max或OOM，也会拖慢主机侧处理和GPU工作提交，其他rank又可能在同步点等待。实际系统/NUMA/分配器成因尚未细分。

当前仍不支持仅增加物理worker：52条逻辑流固定绑定52个worker，64个物理worker中12个无任务；更深预取还可能增加内存驻留。下一步应优先对照每rank阶段计时与内存压力，再决定是否做更小预取预算的控制实验；本次不改64/64或模型。

## 补充诊断与验证

本轮运行的 LoadBalanceTrace 原为 enabled=false。已在52源启动壳的CLI尾参中开启它，每10个microbatch刷盘、最多记录92个样本，保留样本/形状/主机阶段与CUDA事件计时；显式调用者参数仍排在最后。只有平台下次启动才加载，当前没有重启，也不能宣称当前已拥有全192rank的新trace。代码见 `examples/launch_pretrain_state_h1_52src_192gpu.sh:29`。

现有 LoadBalanceTrace 的8项回归通过，包含训练期间不额外同步CUDA事件的检查；shell语法及隔离的真实启动壳参数转发检查通过。证据见 [回归输出](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/load_balance_trace_validation.log)、[启动转发检查](../outputs/dataloader_fix_20261030/slow_step_samples_20261030/trace_launcher_validation.json)。详细故障时序仍见 [pretrain52src_incidents.md](pretrain52src_incidents.md)。
