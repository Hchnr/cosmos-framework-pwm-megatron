# 52 源 State H1 预训练：改动说明

本文按改动组织，回答“改了哪里、运行行为如何变化、验证到什么程度”。具体故障时序和原始现场见 [pretrain52src_incidents.md](pretrain52src_incidents.md)。

核对范围：2026-10-30 的共享工作树，以及截至当日 20:27（北京时间）已归档的验证结果。这里的“修改前”指本次排障过程中的对应版本，不代表所有文件都来自同一个 Git 提交。线上步数是验证样本，不是实时状态；实时进展和故障时间线见故障文档。

## 改动概览

本轮主要收益是减少数据供给的长尾：增加加载并发和预取，同时降低每个 worker 的线程、缓存与原生内存开销；补齐结果传输、坏媒体处理和定位手段。没有通过修改模型计算或关闭 compile 来提速。

| 类别 | 最终改动 | 作用 |
|---|---|---|
| 实验配置 | 24×8 GPU；物理 worker / 全局预取为 64/64 | 提高提前准备数据的能力 |
| Parquet 读取 | 大 row group 分批解码并逐批过滤 episode | 降低过滤前的临时内存峰值 |
| Parquet 缓存 | 保留当前 episode 的帧表，释放大 row group；回收 Arrow 空闲页 | 减少长期驻留的无关数据与分配器保留内存 |
| CPU 线程 | 关闭 tokenizer / Arrow 内部并行，单视频流单解码线程 | 防止多进程与内部线程池叠加 |
| 结果传输 | 同步序列化；pin 线程异常传回主线程 | 让传输错误直接报出，避免批次静默丢失 |
| 诊断信号 | 结果序列化和传输期间屏蔽 SIGUSR1，结束后恢复 | 避免诊断信号打断共享内存分配 |
| 媒体容错 | 在本配方开启坏媒体跳过和连续失败上限 | 单个损坏窗口不再直接终止整个任务 |
| 启动部署 | 本地工作树副本、源码指纹、fd limit、环境设置 | 让每个节点实际加载的版本可核对 |
| checkpoint 恢复 | 显式读取 scheduler metadata；针对已有 checkpoint 同内容重新发布小状态文件 | 暴露真实读取异常，恢复已有检查点的可读取性 |
| 可观测性 | 慢读路径、worker 栈、资源压力、故障归档和告警确认 | 缩短下一次故障的定位时间 |

## 1. 配置：64 个物理 worker，64 个全局在途批次

入口为 [`examples/launch_pretrain_state_h1_52src_192gpu.sh:11`](../examples/launch_pretrain_state_h1_52src_192gpu.sh)，配方为 [`action128_pretrain_state_h1_52src_192gpu.toml:66`](../examples/toml/sft_config/action128_pretrain_state_h1_52src_192gpu.toml)。

| 项目 | 当前值 | 含义 |
|---|---|---|
| 节点 / GPU | 24×8=192 | launcher 默认 NNODES=24；外部同名环境变量仍可覆盖 |
| DP shard / GA | 16 / 3 | 当前并行和梯度累积配置 |
| execution_backend | shared_prefetch | 多源共用物理执行池和全局预取预算 |
| num_workers | 64 | 每个 rank 的物理 worker 数 |
| prefetch_factor | 64 | 每个 rank、所有源合计的在途批次上限 |
| logical_workers | 52 个 0 | 每个源一条逻辑流；物理并发调整不改变逻辑分片 |
| worker_timeout | 450 秒 | 消费端开始等待目标批次后的期限 |
| decode_timeout_sec | 300 秒 | 视频解码请求期限 |
| pin_memory / seed | true / 42 | 后台 pin 和固定数据种子 |

配置曾按要求逐步增至 32/32、64/64，最终保留 64/64。**这里的 64 不是每个 worker 预取 64 批，也不是 4096 批**：执行、结果传输、pin 和 ready 都占用同一份全局 credit。已交给训练端的批次、逻辑流内部 carry/pool 和其他缓存不等于都包含在此预算中。实现见 `cosmos_framework/data/generator/dataflow/shared_prefetch.py:478`。

**池的共享范围是单个训练 rank，不能跨 GPU/rank 借用 worker。** 当前不是动态抢任务：每条 `(source, logical_worker)` 流通过固定 affinity 派给物理进程（`shared_prefetch.py:268`、`:478`）。本轮实际 `config.yaml` 的 `worker_assignment=null`，1200 步 rank0 游标确认 52 个源各有 1 条 active stream，因此默认只会向 worker0–51 派发任务，worker52–63 没有分配流。单纯把物理 worker 从 64 再调高不会增加有效并发；32→64 时可用绑定槽位从 32 增至 52，不能描述为持续线性扩展。每个 rank 的预取预算仍为 64，而不是全节点共用 64。证据见 [`new_run93_worker_affinity_audit.json`](../outputs/dataloader_fix_20261030/failure_micro2490_20261030/new_run93_worker_affinity_audit.json)。

增加每源逻辑流属于另一个改动：会改变逻辑分片、轮转与 RNG 拓扑；现有恢复代码要求 checkpoint 的逻辑 topology 完全匹配（`shared_prefetch.py:350`），不能把它当成无影响的物理 worker 调参直接用于当前恢复任务。本次仅记录限制，保持用户要求的 64/64 和线上逻辑拓扑。

| 配置变化 | 最终行为与取舍 |
|---|---|
| 用户最初入口 `NNODES=23` → 本轮 24 | 192 GPU 可按 shard 16 分组；launcher 默认值不能覆盖外部仍然导出的 23，入口命令也必须正确 |
| worker / prefetch 逐步加至 32/32 → 64/64 | 增大物理执行池和提前准备的预算，同时增加 CPU、内存及共享存储负载 |
| worker_timeout 900 → 450 秒 | 缩短消费端无结果时的诊断和失败等待；不加速底层读取 |
| 本配方坏媒体跳过关闭 → 开启，连续上限 100 | 从后续候选补批，保留失败记录；接受的样本集合可能变化 |

450 秒用于限制等待并保留诊断时间，不能保证总是先于 NCCL 的 600 秒超时报错；各 rank 的通信计时和数据等待计时起点不同。增加 worker/prefetch 提高供给余量，但也放大内存和线程开销，因此必须配合下面的代码修改。

## 2. Parquet：从保留大 row group 改为复用 episode 数据

代码：[`cosmos_framework/data/generator/action/unified_action128.py:363`](../cosmos_framework/data/generator/action/unified_action128.py)。

修改前，窗口读取会保留完整 row group；同一个 group 可能包含许多无关 episode。多源、多 rank 同时持有这些表，会放大驻留内存。即使 Python 引用释放，Arrow 分配器也可能继续保留大量空闲页。

现在的路径是：

1. 大 group（metadata 至少 64 MiB 或帧数至少 65536）使用 `iter_batches(batch_size=8192, use_threads=False)` 分批解码、逐批过滤当前 episode；小 group 保留原路径。继续检查类型、帧数、索引和 episode 边界。
2. 以文件、episode 标识、索引范围、长度、是否读取 state 和容量限制作为缓存键；同 episode 后续窗口直接复用过滤后的帧表。
3. 当结果符合容量限制时，丢弃完整 group 的缓存和临时引用，只保留 episode 表。
4. 大 group 或大量帧读取后，调用 `pa.default_memory_pool().release_unused()`，返还分配器中已经空闲的页；仍在使用的 episode 数据保持有效。

仍保留有界缓存和 PID 感知逻辑，fork 后不复用父进程句柄。直接调用 `row_group()` 的接口仍存在，并非删除所有 group 缓存。容量阈值、数据校验也没有被绕过。回收位置见 `unified_action128.py:459`。

这项改动针对的是内存放大和反复读取，不承诺共享存储上每次读取都能在固定时间内完成。

后续补充的分批读取解决的是另一个阶段：原先“读完整 group 后再过滤、回收”仍会在解码时产生大峰值。现在 episode 路径逐批过滤并只积累当前 episode；完整扫描仍保持原有行顺序与校验，原 metadata 大小限制也在解码前执行。直接调用 `row_group()` 的接口保持原实现。入口见 `unified_action128.py:335`。

这里的 row group 是 Parquet 内部按行划分的存储分组，与训练 batch size 无关。实测 M1 `file-053.parquet` 文件压缩后约 117 MB，只有 1 个 group，含 837,542 条动作/状态采样记录；所有列的 Parquet 未压缩大小约 1.81 GB。其中 action 约 417 MB、observation.state 约 420 MB；previous_action 和 eef_state_revision.original_action 各约 417 MB，但训练读取器本来就不读取这两个历史字段。当前 state 配方所选 9 列合计约 897 MB；动作、状态与有效性 mask 均有训练用途，索引和时间戳用于选取与校验。这里的未压缩大小来自 Parquet metadata，不等于 Arrow 表大小或进程 RSS。这里的行不是图像像素或视频字节；视频按相机元数据路径另行解码（`unified_action128.py:935`），动作频率也不一定等于视频帧率。修复减少同时展开的数值记录行数，不删除训练字段，也不保证减少整组扫描的存储字节数。列级测量见 [`real_m1_rowgroup_column_sizes.json`](../outputs/dataloader_fix_20261030/failure_micro2490_20261030/real_m1_rowgroup_column_sizes.json)。

真实 M1 file-053 的同一 356 行 episode 对照中，action/state 等全部投影列的规范化 Arrow IPC SHA256 一致，进程峰值 RSS 从 4.08 GiB 降到 0.40 GiB，读取约 5.84→3.41 秒（[整组实测](../outputs/dataloader_fix_20261030/pressure_step813_20261030/streaming_probe_full_real.json)、[分批实测](../outputs/dataloader_fix_20261030/pressure_step813_20261030/streaming_probe_stream_real.json)）；合成数据的同 episode 复用与跨 row group 输出也一致。已发布实现的 14 项回归及 lint/format 通过，包含流式边界、空值、重复索引、episode 边界、容量限制和 fork。此处是单机样本，不是 192-rank 压力复现；file-053 是故障 worker 早先的慢读文件，不能断言它就是 microbatch 2490 当时读取的文件。

`job-93f37403…` 的 24 节点已核对加载该实现；线上已由全部 192 ranks 完成原第 831 步，rank54 耗时 33.87 秒；长期内存和步速验证仍在进行。worker / 全局预取继续为 64/64。


## 3. 线程：由物理 worker 提供并行，限制内部线程池

| 位置 | 改动 | 原因 |
|---|---|---|
| `examples/launch_pretrain_state_h1_52src_192gpu.sh:4` | `TOKENIZERS_PARALLELISM=false` | 在父进程初始化 tokenizer 前关闭额外 Rayon 并行 |
| `cosmos_framework/data/generator/dataflow/shared_prefetch.py:151` | worker 初始化时再次设置同一变量 | 在 worker 入口明确执行策略 |
| `cosmos_framework/data/generator/action/unified_action128.py:329` | `read_row_group(..., use_threads=False)` | 避免每个加载进程再开 Arrow 列读取线程池 |
| `cosmos_framework/data/generator/media_decode.py:58` | 每条视频流 `thread_count=1` | 避免每个缓存的视频流再创建自动解码线程池 |

这些改动保持多进程加载，不会把整个数据管线改为单线程。worker 中已有的 `torch.set_num_threads(1)` 单独不足以约束 tokenizer 和视频解码库。

实测 tokenizer 在 OMP=1 时仍能把单进程线程数从 2 增到 82；关闭内部并行后保持 2。真实 token IDs/mask 一致；10 次真实视频请求在自动与单解码线程下的 RGB SHA256 和元数据一致。这些是已测样本的等价性验证，不是对全部视频编码格式的穷尽证明。

## 4. IPC：同步传输结果，保留真实错误

代码：[`cosmos_framework/data/generator/dataflow/shared_prefetch.py:109`](../cosmos_framework/data/generator/dataflow/shared_prefetch.py)。

原结果通道使用 `multiprocessing.Queue`。`put()` 返回后，后台 feeder 线程才序列化；如果 tensor 共享或 reducer 此时失败，worker 的异常处理捕获不到，训练端可能只看到目标批次一直没有返回。

现在用 `_ResultQueue` 包装 `SimpleQueue`：序列化和管道写入在 worker 的 `put()` 中完成，失败能进入 worker 的异常回传路径；不再依赖 feeder 线程完成结果提交。接收侧用 pipe poll 实现超时，继续使用原全局 credit，不新增一层预取预算。后台 pin 线程接收或反序列化异常也会包装后交给训练主线程，位置为 `shared_prefetch.py:226`。

同时，在 `_ResultQueue.put()` 期间暂时屏蔽本线程的 SIGUSR1，并在 `finally` 中恢复原屏蔽状态，位置为 `shared_prefetch.py:127`。这是对已复现问题的修复：诊断信号可能打断 PyTorch 共享内存路径的 `posix_fallocate`，当前构建将 EINTR 错报为 `Success (0)`。读取和解码期间仍可采集诊断栈。

验证包含不可序列化结果直接报错、pin 异常传播、信号屏蔽成功/失败路径恢复，以及 512 MiB 真实 tensor IPC 连续信号测试。该修改没有升级共享 Python 环境或修改 PyTorch 安装。

## 5. 媒体容错：启用已有策略，跳过损坏窗口并补批

这是**本配方开启已有能力**，不是本次新增一套跳过算法：

```toml
[dataloader_train.media]
action_skip_media_errors = true
action_max_consecutive_skips = 100
```

配方位置：`examples/toml/sft_config/action128_pretrain_state_h1_52src_192gpu.toml:83`；策略实现：[`cosmos_framework/data/generator/action/media_failure_policy.py:26`](../cosmos_framework/data/generator/action/media_failure_policy.py)。

损坏窗口记录到 `$OUTPUT_ROOT/media_failures/`，从后续候选样本补齐批次。候选 cursor 被保留，提交并保存状态后，恢复时不会重新消费此前已跳过的窗口。连续失败达到 100 次仍终止；结构性和编程错误不会一律当成坏视频吞掉。

本轮另有一项错误分类修改：相机引用时间范围不能覆盖当前窗口时，改用 `MediaDecodeError`，沿用上述显式开启的跳过策略；记录相机、路径和请求起止时间，并清理解码器。只改变这一项覆盖检查的错误分类，其他 `CatalogSchemaError` 继续报错；不截断请求或伪造对齐。见 `unified_action128.py:956`。定向回归 2 项、既有媒体策略回归 13 项及 lint/format 通过；24 节点加载已核对，且 rank 154 / legacy 已实际记录并跳过一个覆盖不足窗口；恢复任务完成 704 步，越过原 microbatch 2110 所在训练步，证据见故障文档。

线上已对原损坏的 M1 file-418.mp4 记录跳过，rank 74 后续实际消费了替代批次。**源视频没有被修好，跳过也会改变有效样本分布**；后续仍需清洗上游媒体。

## 6. 启动：配置之外，shell 也有改动

仅调整 worker/prefetch 数值可改 TOML；部署整套修复还依赖 launcher 和共享 Python 源码。

| 文件 | 当前行为 |
|---|---|
| `examples/launch_pretrain_state_h1_52src_192gpu.sh:4` | 关闭 tokenizer 内部并行 |
| 同文件 `:6` | 若 fd soft limit 低于 65536，则提高到 65536，减少 tensor IPC/fd 压力 |
| 同文件 `:11` | 选择 52 源配方、输出根目录和诊断目录，默认 24 节点 |
| 同文件 `:17` | 先复制到节点本地目录，再执行训练入口 |
| 同文件 `:20` | 打印实际本地 shared_prefetch、unified_action128、media_decode、dcp 四份源码 SHA256 |
| `examples/_stage_training_code.sh:12` | 用 Git 文件列表复制已跟踪和未忽略的未跟踪文件，包含未提交修改；每次使用新目录 |

本地目录默认位于 `/opt/cosmos-framework-pwm/run.*`，避免训练依赖共享目录作为工作目录。模型、输出和数据并不因此全部复制到本地。52 源 launcher 随后复用 `launch_pretrain_state_h1_96gpu.sh` 的通用启动链，文件名中的 96 不决定实际 GPU 数；NNODES/NPROC_PER_NODE 和本轮配方才决定实际设置。

共享绝对路径的 TOML 与节点本地源码不是原子部署快照：启动过程中改文件，仍可能发生节点版本不一致。因此必须核对 24 节点的指纹、真实保存配置及启动时间。**修改共享文件不会热更新已运行进程**，由平台后续自动重启加载；不需要先 commit 才能复制工作树修改。

| 要应用的改动 | 需要加载的文件 | 生效方式 |
|---|---|---|
| 仅 worker / prefetch 数值 | 配方 TOML | 下一次训练启动读取；shell 无需为数值再重复设置 |
| 缓存、线程、IPC、媒体错误分类、恢复诊断 | 对应 Python 源码 | 下一次启动复制到节点本地并导入；核对源码指纹 |
| fd limit、tokenizer 父进程环境、节点本地部署 | launcher 及 staging shell | 下一次执行 launcher |
| 告警解析、日志归档 | 输出目录中的监控脚本 | 重载监控进程即可，无需重启训练 |

## 7. 诊断和监控：为后续故障保留现场

这些改动用于定位和告警，不等同于数据吞吐优化。

- `unified_action128.py:312`：单次 Parquet 读取超过 30 秒，打印路径、row group、未压缩字节数和列。
- `shared_prefetch.py:486`：等待信息包含 source、逻辑/物理 worker、PID、当前/最近提交的 microbatch 和 ready 列表；周期性采集目标 worker 与 pin 线程栈。
- `shared_prefetch.py:570`：补充已结束但耗时至少 10 秒的 receive 日志，覆盖短于 60 秒栈诊断间隔的等待。计时包含结果接收与后台 pin，不能直接视为磁盘读取耗时。该诊断补丁已在后续 3c、93 任务加载；原 e6 故障时尚未加载。测试结果见 `pressure_step813_20261030/receive_wait_regression.log`。
- `cosmos_framework/tools/launch_diagnostics.py:37`：补充 cgroup CPU 配额/节流、CPU/内存/I/O PSI、memory.high/low、pids 和 `/dev/shm` 容量。
- `outputs/dataloader_fix_20261030/watch_training.py`：每 30 秒只读采样共享日志，识别每轮启动/恢复前 50 步的逐 rank 日志及其后的 rank0 日志格式，发现原始异常/NCCL watchdog 后归档对应 job 现场。
- 同目录 `watch_training.py` 的 tail 读取跳过字节窗口开头的不完整行，保留完整边界行，防止历史告警被截断后作为新文本重复报警；6 项隔离检查通过，含真实日志复现。该修复只需重载监控。
- 同目录 `observe_training.py`：按偏移读取所有新增告警，致命告警保持 pending，复核后显式确认，避免只看最新快照漏掉中间一轮。 消费等待另设独立计数和摘要，避免它排在两条 Parquet 慢读之后时被示例上限遮住；7 项隔离回放检查通过，包含部分行续读、致命告警保留和确认时的新告警。此项立即在监控端生效，无需重启训练。

后两项是实验输出目录中的运维脚本，不属于训练数据管线源码。每轮启动/恢复前 50 步后，日志仅报告 rank0 步时；本次从 800 恢复，逐 rank 记录到 850，不能把其他 rank 最后一次 850 步记录解释成停住。切换后的首步 851 通常只初始化计时、不打印 IterSpeed，852 起恢复正常打印，见 `cosmos_framework/callbacks/iter_speed.py:101`。观察超时或等待告警也不等于进程已经退出。

## 8. 验证结果：主要改善长尾，不能当成单项因果实验

### 本地资源对照

相同 rank 40、OMP=1、64/64、120 个真实批次、后台 pin：

| 版本 | 峰值 PSS（GiB/rank） | 峰值线程 | 耗时（秒） |
|---|---:|---:|---:|
| 对照版本 | 139.57 | 2902 | 260.64 |
| episode 缓存 + tokenizer 串行 | 138.99 | 183 | 261.41 |
| 再加 Arrow 空闲页回收 | 78.71 | 182 | 268.99 |

完整修复的峰值 PSS 降约 43.6%；单次回放耗时约增加 3.2%，未做统计显著性推断。这说明不能只删除 Python 缓存引用，还需处理原生分配器保留页；也不说明单机数据读取速度提高一倍。

补充验证：停止消费后确实填满 ready=inflight=limit=64，峰值 79.58 GiB、182 线程。该测试包含主动等待，不能用总耗时比较吞吐。120 批 source/cursor 与对照一致，四个真实 episode 的 action/state 表一致；这些检查不证明整轮 GPU 训练逐位一致。

相关回归：缓存/回收等 11 项通过；队列、信号、跳过相关回归 57 项通过，1 项因缺外部 DROID fixture 排除；启动诊断 7 项通过。Python lint/format 和相关 shell 的 `bash -n` 已通过。这里记录已有验证，本次写文档未重新执行训练测试。

### 线上观察

整套资源修复已在 `job-29fe65de-b52b-4013-8124-c1f72624a59f` 全部 24 节点核对加载，实际配置为 64/64。全部 192 ranks 在前 50 步的直接日志中，已越过历史 microbatch 46/108/117/125。

| 指标 | 前轮旧资源实现 | 29fe 修复任务 |
|---|---:|---:|
| 第 5–30 步累计时间 | 1715.62 秒（28.6 分钟） | 890.77 秒（14.8 分钟） |
| 同区间单步中位数 | 35.28 秒 | 34.49 秒 |
| 同区间最长单步 | 196.68 秒 | 36.20 秒 |

典型单步耗时变化不大，累计收益主要来自减少长尾等待。两轮外部存储负载可能不同，不能分离各补丁的独立贡献，也不能宣称所有数据超时永久消除。该 29fe 任务第 202–300 步 rank 0 中位 33.86 秒，范围 31.27–35.93 秒。

内存仍需长期观察：第 300→326 步，24 节点 memory.current 中位增加约 26.8 GiB，其中 anon 中位增加约 13.0 GiB，shmem 基本持平。这些分类不能直接定位分配者；shmem 属于 file 子项，不可重复相加。没有据此宣布内存增长问题已经完全解决。

### 为什么早期慢，后面恢复正常

需要分开看三种现象：

1. **启动成本**：compile、tokenizer AOT、数据冷读与预取池建立会叠加。29fe 的第 1 步约 357.7 秒，第 4 步约 170.9 秒且有 AOT，第 5 步已到 36.2 秒。后续 e6 从 600 步恢复，首个 601 步为 533.85 秒；192 ranks 的 tokenizer AOT 各约 199.82–228.54 秒。这些计时存在重叠，不能简单相加，也不能将整步时间全部归给 compile。
2. **运行中的数据长尾**：历史任务已经进入训练后，仍出现 Parquet 读取、结果提交或媒体错误导致的等待。一个 rank 没有目标批次，就可能让其他 ranks 在同步通信处等待；增加预取不能保证掩盖任意长的读取。资源修复主要减少这类长尾，而非大幅缩短模型本身的计算时间。
3. **W&B 曲线拼接**：复用旧 run、重启后 step 回退，可能使早期新记录未写入。图上“几十步后变快”不一定是同一次进程连续 warmup 的结果。

恢复后的 e6 第 701–750 步共 50 条 rank 0 步时，中位 34.10 秒、范围 32.31–35.41 秒；期间仍有后台 Parquet 慢读告警。**慢读告警表示预取中有慢任务，不直接证明训练消费端已等待或 GPU 已空转**；这 50 步也不足以排除更短的局部等待。对比应同时看目标批次等待、rank 步时和通信日志。

## 9. checkpoint：增加读取诊断，单独处置已有文件可见性

这一部分包含两种不同性质的操作：

| 类型 | 实际改动 | 验证与边界 |
|---|---|---|
| 保留的代码修改 | scheduler 的 `dcp.load` 前显式调用 `storage_reader.read_metadata()`；launcher 加印 `dcp.py` 指纹 | 原先泛化的 `metadata is None` 被定位为训练节点打开 `scheduler/.metadata` 的 `FileNotFoundError`；不改变恢复状态的选择规则 |
| 针对已有文件的处置 | 分别备份 600、800 步 checkpoint 的 scheduler/trainer，每个 checkpoint 共 195 个非空小文件（各约 4.99 MB），写临时文件、fsync 后原子替换，逐文件核对 SHA256 前后一致 | 同一路径在当前共享挂载上原本可读，重新发布后下一轮实际恢复成功；没有证明底层挂载或缓存机制，也没有据此断言永久修复 |

代码位置：[`cosmos_framework/checkpoint/dcp.py:1010`](../cosmos_framework/checkpoint/dcp.py)。模型、优化器文件及 latest 指针未参与这次重新发布；没有跳过 scheduler/trainer 恢复或伪造新训练状态。

验证不仅检查文件存在：本地实际读取 metadata 并加载 scheduler 状态；随后 e6 任务全部 24 节点核对四份源码指纹，线上成功从 600 步恢复数据游标 1800、模型、优化器、scheduler 和 trainer，加载耗时 33.39 秒。全部 192 ranks 完成恢复后的首步，继续训练已验证；未做逐位轨迹等价测试。 后续 800 步 trainer metadata 可见性故障按同样方式处理后，93 任务也在 15:49:15 完成完整状态加载，耗时 76.07 秒，游标恢复 2400；恢复后的首步已在 192 ranks 核对完成，rank0 继续到 803 步，原第 831 步已由全部 192 ranks 通过，长期验证仍在进行。

观察保存后内存时，还需计入原有异步保存机制：`cosmos_framework/checkpoint/dcp.py:1152` 首次创建并保留 pinned/shared CPU staging 状态，后续保存复用它。完成写盘不意味着这部分缓冲立即释放。e6 首次恢复后保存为 800 步，775→803 步的节点 shmem 中位约增加 60.6 GiB，与该机制相符，但分类计数本身不足以精确归因；需将这个分配台阶和之后的持续增长分别观察。这是已有机制的解释，本轮未修改其内存管理。

## 10. 保持原有行为的部分

- **模型与训练目标**：资源补丁没有修改模型结构、loss、批次 token 上限、52 源比例、seed 或逻辑分片。坏媒体跳过会影响实际接受的样本；旧/新线上 loss 也并非逐步完全相同。
- **compile**：仍按配方启用 language 区域编译，见 `examples/toml/sft_config/action128_pretrain_state_h1_52src_192gpu.toml:39`；没有用关闭编译换取上述数据供给改进。
- **W&B / MFU**：没有新增 MFU 算法，也没有修改 run ID。29fe 越过当时旧 run 的第 41 步后恢复写入是已核对的历史现象，不代表后续重启仍以 41 为边界。实现见 `cosmos_framework/callbacks/mfu.py:85`。
- **恢复选择规则**：同目录 latest 优先于外部 warm-start，见 `cosmos_framework/checkpoint/dcp.py:789`。显式 metadata 读取只补充错误诊断；上述小文件重新发布也不改变状态内容。

## 证据入口

以下路径均位于 `outputs/dataloader_fix_20261030/`，属于本次实验产物，可能受输出目录保留策略影响：

| 内容 | 路径 |
|---|---|
| 本地内存回收、缓存和满预取 | `failure_micro108_20261030/REPORT.md`、`episode_purge_validation.json`、`full_buffer_validation.json`（后两项在同子目录） |
| 信号中断与真实 tensor IPC | `failure_shm_signal_20261030/torch_signal_validation.json`、`regression_verified.log`（同子目录） |
| 视频单线程等价性 | `decode_parity_validation.json` |
| 真实坏视频跳过 | `failure_step38_20261030/skip_real_video_validation.json` |
| 29fe 部署与线上步时对照 | `failure_micro125_20261030/new_run_deployment.json`、`online_before_after_steps5_to30.json`（同子目录） |
| 29fe 资源增长样本 | `failure_micro125_20261030/resource_growth_300_to326.json` |
| 保存完成与数据游标 | `checkpoint200_verified.json`、`checkpoint200_cursor_inspection.json` |
| 相机覆盖不足的窗口跳过 | `failure_camera_coverage_20261030/summary.json`、`new_run_deployment.json`（同子目录） |
| 恢复诊断及同内容重新发布 | `failure_resume_metadata_20261030/summary.json`、`small_state_republish.json`、`new_rune6_resume_progress.json`（同子目录） |
| e6 实际部署、AOT 和稳态步时 | `failure_resume_metadata_20261030/new_rune6_deployment.json`、`new_rune6_aot_progress.json`、`new_rune6_steps701_to750_timing.json`（同子目录） |
| 分批读取修复及本地验证 | `failure_micro2490_20261030/streaming_patch_report.json`、`published_streaming_regression.log`、`new_run93_deployment.json`（同子目录）；原始对照在 `pressure_step813_20261030/streaming_probe_*real.json` |
| 800 步小文件重新发布 | `failure_resume800_metadata_20261030/small_state_republish.json`、`local_metadata_read_after_republish.json`（同子目录） |
| 监控告警覆盖 | `monitor_nccl_alert_validation.json`、`observer_alert_delivery_validation.json`、`observer_consumer_wait_validation.json` |

本文件维护最终改动和验证边界；新的故障和逐次观察继续记入 [故障记录](pretrain52src_incidents.md)，避免把此处变成第二份时间线。
