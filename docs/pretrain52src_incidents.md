# Cosmos 52 源预训练：故障与修复记录

按文件和行为整理的改动说明、验证结果及部署方式，见 [pretrain52src_changes.md](pretrain52src_changes.md)。本文保留故障现场和处理时间线。

更新：2026-10-30 20:50，北京时间。93 任务至 1235 步，100.01 秒；主节点与 worker2 内存压力再次升高，仍无新退出、OOM 或消费超时。1000/1200 步保存及 192 份游标检查通过，继续监控。

## 当前状态

当前任务为 `job-93f37403-b199-413b-a201-25c441e1a976`，24 节点四份源码指纹一致，已加载大 row group 分批读取和短时 receive 诊断，worker / 全局预取仍为 64/64。15:49:15 完成 800 步 dataloader/model/optim/scheduler/trainer 加载，游标 2400，耗时 76.07 秒；首个恢复步 801 已由全部 192 ranks 完成，rank0 进一步到 803 步；旧失败步 831 已在全部 192 ranks 核对完成，rank54 耗时 33.87 秒。部署与恢复证据见 `failure_micro2490_20261030/new_run93_deployment.json`、`new_run93_resume_progress.json`。训练目标与监控继续。

旧 c5 任务中 `shared_prefetch.py` 的信号补丁 SHA256：`76b7ebff5e7230dc2018db49950bda42e59f8082447c49b637ffb1e4e84fd1ed`。部署证据：`outputs/dataloader_fix_20261030/failure_shm_signal_20261030/deployment.json`。

前轮现场（05:14）：全部 192 ranks 完成第 41 步后，第 42 步等待 microbatch 125，来源 `InternData-A1-eef--other_unclassified:8`。node3 的 rank 104–111 多个 worker 在 `read_row_group`，该节点 `memory.pressure full avg10` 约 94%，尚未观察到 OOM、退出或最终 TimeoutError。慢读首次告警早于消费等待约一小时，文件元数据大小约 0.54–0.91 GB。六份待加载文件的 SHA 与本地已测版本全部相符，证据在 `outputs/dataloader_fix_20261030/failure_micro125_20261030/`；内存压力不等同于共享存储磁盘已满，尚不能确定唯一根因。

05:15 最终结果：上述等待已触发 `TimeoutError: Shared dataloader timed out after 450.0s waiting for global microbatch 125`。该轮完成 41 步，第 42 步失败；现场已归档。当时共享补丁已就绪；后续 29fe 任务已完成 24 节点部署和历史故障位置验证，见上方当前状态。详见 `outputs/dataloader_fix_20261030/failure_micro125_20261030/REPORT.md:1`。

补充时序：本轮其他 ranks 已于 05:14:38 报 NCCL ALLREDUCE watchdog 600 秒超时，早于 node3 的最终数据 TimeoutError。两者计时起点不同，450 秒数据超时并不保证先于所有通信超时报出；长时间读取等待的证据早于这两者。详见本次报告的 `nccl_secondary_timeouts.json`。

## 实验入口与配置

仓库：`/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm`。启动入口为 `examples/launch_pretrain_state_h1_52src_192gpu.sh`，配方为 `examples/toml/sft_config/action128_pretrain_state_h1_52src_192gpu.toml`。

| 项目 | 当前设置 |
|---|---|
| 节点 / GPU | 24 节点 × 8 GPU，192 ranks |
| DP shard / 梯度累积 | 16 / 3 |
| 数据 | 52 个 catalog，State H1，seed 42 |
| 物理 worker / 全局预取 | 64 / 64，每个 rank 分别计数 |
| 逻辑流 | 每个源一条，合计 52 条；调物理 worker 时不改变逻辑分片 |
| 数据等待 / 视频解码超时 | 450 秒 / 300 秒 |
| 媒体错误 | 跳过失败窗口并记录；连续 100 个失败仍终止 |
| 文件描述符 | launcher 将 soft limit 至少提高到 65536 |

最初提交的命令包含 `NNODES=23`。若该值与 8 GPU/节点、DP shard 16 一起生效，184 不能整除 16，属于启动配置风险；当前已按用户重启改为 24 节点。现存证据不足以将此风险断言为最初一次失败的首因。

输出根目录：`outputs/pretrain52src_state_h1_nano_192gpu_ga3_from17200_run01/`。`logs/action128_node*.log` 随重启覆盖，`logs/entrypoint_node*.log` 追加；每个 job 的诊断文件在 `logs/diagnostics/`。原始故障日志应在覆盖前归档。

## 关键任务与时间线

| 任务 | 观察到的进度 / 事件 | 结论或处置 |
|---|---|---|
| `job-b9ee1270…` | 第 9 步约 238 秒，多 rank 等待 microbatch 26 | 发现跨 rank 数据长尾与 MFU 明显下降的关联 |
| `job-4c397b38…` | 完成 15 步；第 16 步卡在 microbatch 46 | worker 数据读取 / 结果传输未完成，其他 rank 随后 NCCL 超时 |
| `job-6073c1e4…` | 完成 37 步；第 38 步收到预取 microbatch 117 的异常 | rank 74 遇到损坏的 InternData-M1 AV1 视频 |
| `job-59c2bd3d…` | 在跳过配置修改前已加载 `skip=false` | 后续重启才会加载新增开关 |
| `job-67cf7718…` | `skip=true` 已生效；首批数据传输失败 | 诊断 SIGUSR1 中断共享内存预分配，错误被误报为 Success (0) |
| `job-5550278d…` | 启动源码早于最新信号补丁 | 不能用这轮验证最新补丁；具体终止原因未完整审计 |
| `job-1052876d…` | 完成 36 步；第 37 步 microbatch 108 读取 Parquet 超时 | 24 节点补丁一致，坏视频跳过有效；新读取阻塞仍在定位 |
| `job-c5b3f738…` | 全部 192 ranks 完成第 41 步；第 42 步等待 microbatch 125，源 8 | 原坏媒体替代批次已消费；约 94% 内存压力，随后 450 秒超时；新内存补丁未加载 |
| `job-29fe65de…` | 24节点补丁和64/64已核对；首个200步保存完成，rank 0至701步 | 历史故障位置通过；119.7 GiB检查点与192份游标齐全，完整恢复未测试；持续监控 |

## 一、数据读取长尾拖住全局训练

`job-4c397b38…` 在 00:01:12 完成第 15 步，随后 rank 108/111 等待 microbatch 46，源为 `InternData-A1-eef--basic_transfer:4`。rank 111 的 worker 栈持续停在 PyArrow `read_row_group`；rank 108 后来显示已提交该批，主进程仍没有收到。其他 rank 最终在 DP-replicate ALLREDUCE 等待超过 600 秒后退出。NCCL 超时是后续症状。

处理：把 worker/prefetch 按要求从 32/32 提至 64/64；Arrow row group 读取改为 `use_threads=False`，由物理 worker 提供并行；worker timeout 从 900 秒改为 450 秒，让源数据错误早于 NCCL 600 秒超时暴露，同时为媒体解码的 300 秒期限留出余量。

已知边界：仅凭 Python 栈不能严格区分共享存储延迟与 fork 后 Arrow 原生线程池问题。原实现本地也能读过 microbatch 46，因此不能宣称线上 I/O 卡住已经被完整复现并消除。 新事件：`job-1052876d…` 于 03:34:00 在第 37 步等待 microbatch 108 超时，rank 40–43 的 physical worker 3 均在 `read_row_group`，后续多批已 ready。此次已启用 `use_threads=False`，说明前一次线程调整不足以消除所有读取阻塞。四个 rank 的首个候选 episode 文件本地串行读取约 3.7–5.6 秒，CUDA 初始化后的 fork 读取也通过；这不是完整故障批次复现，也不能确认线上实际停在首个候选文件。本机探测与 pytest 导入期间还观察到内核 `wait_on_page_bit_common` I/O 等待。已新增超过 30 秒的 Parquet 读取文件路径 / row group / 大小日志，供后续自动启动定位；这只是诊断补充，尚无根治结论。 新增诊断相关的 9 项测试通过，92 项未选；按 52 源配置还原的 rank 40 源 3 首批串行回放约 24.5 秒，CUDA 初始化后的 rank 40–43 四进程 fork 首批分别约 19.4–20.5 秒，均正常完成。单机回放未施加线上 192-rank 的存储压力，仍未复现线上阻塞。证据归档：`outputs/dataloader_fix_20261030/failure_micro108_20261030/` 与 `outputs/dataloader_fix_20261030/incidents/job-1052876d-abf7-4eca-ae01-dd61eae0e3e4/`。 04:04 后的新增诊断显示，慢读节点 `memory.pressure full avg10` 达 70–82%，但 `memory.events` 的 high/max/oom/oom_kill 仍为 0，且总内存低于 memory.max。已复现两项资源放大：每个 worker 的 tokenizer 可额外创建 80 个 Rayon 线程；Arrow 逻辑缓存释放后，mimalloc 仍保留大量空闲页。只保留 episode 并关闭 tokenizer 并行时，完整回放 PSS 仍约 139 GiB；再在大 group 提取后释放临时表并调用 `release_unused()`，120 批峰值降至 78.71 GiB。缓存/回收/限制/fork 检查 11 项通过，队列与跳过策略回归 57 项通过（缺失外部 DROID fixture 的 1 项排除），启动诊断 7 项通过。四个真实 episode 的 action/state 帧表与旧读取完全一致。这证明修复了可复现的额外资源占用，尚不能断言内存回收是线上所有读超时的唯一原因。代码：`cosmos_framework/data/generator/action/unified_action128.py:332`、`:419`。详见该故障目录的 `REPORT.md` 和 `episode_purge_validation.json`。

代码：`cosmos_framework/data/generator/action/unified_action128.py:308`；配方 `:73`。证据：`outputs/dataloader_bench_20261029/stall_micro46_node3_20261030.log:287`、`outputs/dataloader_fix_20261030/REPORT.md`。

线上验证补充：29fe 新任务已在 24 节点加载整套资源修复，并由全部 192 ranks 消费越过历史 microbatch 108/125。第 5–30 步总耗时从前轮约 1715.62 秒降至 890.77 秒，中位步时变化较小，长尾明显减少。这支持整套修复在本轮的有效性，但不能分离 tokenizer、episode 缓存、Arrow 回收各自的贡献，也不能排除两轮外部存储负载不同。证据：`failure_micro125_20261030/online_before_after_steps5_to30.json`。

## 二、后台 Queue 序列化失败会丢结果

标准 `multiprocessing.Queue.put()` 返回后，feeder 线程才序列化结果。若 tensor 共享或其他 reducer 在 feeder 线程报错，worker 已认为 put 成功，主线程只能一直等缺失的批次。这可以解释一种“后面的批次已 ready、目标批次一直不到”的故障模式，但不能据此追溯断言上轮 microbatch 46 一定由此触发。

处理：结果通道采用同步序列化的 `SimpleQueue` 包装，让 worker 自己捕获并回传异常；后台 pin 线程的接收 / 反序列化错误也会传给训练主线程。仍使用原有全局在途 credit，不新增另一层预取窗口。

验证：用不可 pickle 的 lock 作为结果，旧实现两种 pin 模式都只报 feeder 错误并超时，修复后直接报告真实序列化错误且回收 worker。代码：`cosmos_framework/data/generator/dataflow/shared_prefetch.py:109`。证据：`outputs/dataloader_fix_20261030/serialization_before_fix.log`。

## 三、增加 worker 前要处理嵌套线程

PyAV 自动 codec 线程会为缓存的每条视频流创建线程池。本地抽样曾看到一个 decoder helper 有 1153 个线程，一个 rank 的进程树峰值超过 15000 线程。物理 worker 增加后，这种嵌套并行会加重 CPU 争用。

处理：每条视频流设置 `thread_count=1`，Arrow 列读取不再另开线程池；保留用户指定的 64/64。修复后的两个 rank 数据回放峰值线程数为 173–174。10 次真实视频请求的 RGB SHA256 和返回元数据与自动线程完全一致；这不是对所有视频编码格式的穷尽证明。 后续 rank 40 测试发现，即使 OMP=1，tokenizer 批处理仍会将单进程线程数从 2 增至 82；关闭 `TOKENIZERS_PARALLELISM` 后保持 2，token IDs / mask 的 SHA256 完全相同。现已在 52 源 launcher 和物理 worker 初始化中关闭该嵌套并行，新的完整 120 批回放峰值为 182 线程。最初未带 OMP=1 的 5076 线程测试与 torchrun 启动条件不同，不能直接用于推断线上线程数；OMP=1 对照的 2902 线程和逐进程采样才是该次修复基线。代码：`cosmos_framework/data/generator/dataflow/shared_prefetch.py:151`、`examples/launch_pretrain_state_h1_52src_192gpu.sh:4`。

代码：`cosmos_framework/data/generator/media_decode.py:58`。证据：`outputs/dataloader_fix_20261030/decode_parity_validation.json`、`fixed_rank111/summary.json`、`fixed_rank108/summary.json`。

## 四、损坏的 AV1 视频导致第 38 步退出

2026-10-30 01:41:40，rank 74 在等待 microbatch 113 时收到预取 microbatch 117 的媒体异常，源为 `InternData-M1-eef:33`。出错视频为 `InternData-M1/eef/videos/9780fc45d421ce0c/images.rgb.ego_view/chunk-013/file-418.mp4`。

该路径是符号链接，实际指向 `action_processed_r1/InternDataM1-Franka/videos/images.rgb.ego_view/chunk-056/file-056386.mp4`，大小 1,457,524 字节。视频声明 355 帧，libdav1d 自动 / 单线程以及独立 libaom-av1 解码器都在 329 帧、最后输出时间戳约 10.93 秒后失败。libaom 明确报告 `Corrupt frame detected` 和 `Failed to decode tile data`。

处理：在当前 52 源配方 `[dataloader_train.media]` 开启 `action_skip_media_errors=true`、`action_max_consecutive_skips=100`。失败窗口记录到 `$OUTPUT_ROOT/media_failures/`，继续从后续候选补齐批次；保留原始候选 cursor，该状态提交并用于恢复后，不重读此前跳过的窗口。连续 100 次失败仍退出，结构性 / 编程错误不会被当成坏视频吞掉。跳过会改变有效样本分布，上游视频仍需清洗。

验证：13 项现有策略测试通过；真实坏视频触发错误后，成功记录一次跳过并从正常视频补齐两样本批次。窄时间窗口可能没有覆盖损坏帧而仍可解码，验证采用覆盖整段的采样请求。`job-67cf7718…` 的保存配置已确认 skip=true。 线上补充验证：2026-10-30 02:30:27，rank 74 对同一 file-418.mp4 记录 decode_error，episode_index=52449、window_index=5、skipped=1；训练随后继续前进。该错误来自预取，105 任务在这次记录时尚未实际消费该批次。证据保存为 `outputs/dataloader_fix_20261030/live_media_skip_20261030.json`。 后续线上验证：`job-c5b3f738…` 的 rank 74 于 04:01:20 再次跳过同一 episode 52449/window 5；05:00 全部 192 ranks 完成第 40 步。保存配置 GA=3，rank 0 dataloader global_id=120，且捕获 rank 74 第 40 步完成记录，确认 microbatch 117 已实际进入训练，替代批次消费验证通过。该结论不表示源视频已修复，也不是新增内存补丁的部署验证。证据：`outputs/dataloader_fix_20261030/failure_micro108_20261030/milestone_c5_step40.json`。

新内存补丁任务补充验证：29fe 的 rank 74 于 05:45:46 再次跳过原 file-418.mp4（episode 52449/window 5），全部 ranks 后续完成第 40 步；替代批次 microbatch 117 已实际消费。证据：`failure_micro125_20261030/new_run_media_skip.json`、`milestone_29fe_micro117.json`。

代码：配方 `:87`、`cosmos_framework/data/generator/catalog.py:396`、`cosmos_framework/data/generator/action/media_failure_policy.py:26`。证据：`outputs/dataloader_fix_20261030/failure_step38_20261030/REPORT.md`、`independent_decoder_probe.log`、`skip_real_video_validation.json`。

## 五、诊断信号触发 PyTorch 共享内存分配中断

`job-67cf7718…` 的最早错误发生于 02:01:19：首个 microbatch 的结果传输报 `unable to allocate shared memory(shm) … Success (0)`。此前 02:01:12 为等待 worker 发送了 SIGUSR1；归档栈显示目标 worker 正在 `_share_fd_cpu_`。其他 rank 后续才出现 NCCL remote peer 退出。

根因已通过本地实验复现：信号可使 `posix_fallocate` 返回 `EINTR=4`，而 errno 保持 0；当前 PyTorch 路径误读 errno，把中断显示成 Success (0)。5 次底层复现均得到相同结果。PyTorch 上游已有相同错误处理问题的记录：[issue #176069](https://github.com/pytorch/pytorch/issues/176069)。

这次诊断机制本身触发了训练故障。处理是在 `_ResultQueue.put()` 期间暂时屏蔽本线程的 SIGUSR1，并在 finally 中恢复原信号屏蔽状态。读取与解码期间的栈诊断仍保留，不改 64/64，不升级或热改共享 Python 环境。

验证：向实际 512 MiB tensor IPC 传输连续发送诊断信号，旧实现复现同样的 Success (0) 错误，新实现完整传输成功；成功 / 异常两条路径的信号恢复测试通过。相关回归最终为 **57 passed, 1 deselected**；排除项仅为本机缺少外部 DROID 文件的既有测试。

代码：`cosmos_framework/data/generator/dataflow/shared_prefetch.py:127`。证据目录：`outputs/dataloader_fix_20261030/failure_shm_signal_20261030/`，包括 `entrypoint_node14.log:504`、`torch_signal_validation.json`、`regression_verified.log`、`deployment.json`。根因不应被误判为 NFS 磁盘已满。

## 六、重启时配置和代码可能来自不同时间

launcher 会先把工作树复制到节点本地 `/opt/cosmos-framework-pwm/run.*`，随后训练从共享绝对路径读取 TOML。因此启动途中修改文件，可能出现“新配置、旧代码”，甚至不同节点处于不同版本。

实际发生过：`job-6073c1e4…` 在 00:42 开始复制，而结果通道 / Arrow 修复在 00:49、解码线程修复在 00:53 才落盘，最后 23 个节点为旧版本、1 个节点为新版本。另一次 `job-59c2bd3d…` 在 01:50:29 修改跳过开关之前已加载 skip=false，后续保存配置仍是旧值。

处理：每次启动记录三份实际节点本地源码 SHA256；核对全 24 节点、job 启动时间、配置保存时间和真实开关。复制脚本包含工作树已修改 / 未提交文件，无需为了部署先 commit。已运行进程不会因共享文件改变而热更新；修复由后续自动重启加载。

代码：`examples/_stage_training_code.sh:12`、`examples/launch_pretrain_state_h1_52src_192gpu.sh:17`。`job-1052876d…` 的 24 节点信号补丁已核对一致，`job-c5b3f738…` 的 24 节点启动源码也一致；新增 tokenizer/episode/Arrow 回收补丁写入时间晚于这轮复制，不能把当前任务当作其部署验证。

## 七、慢步、compile、MFU 和 GPU 等待的判断

不同阶段需要分别依据日志判断。`job-6073c1e4…` 前三步约为 555、40、45 秒；第 3 步后触发 tokenizer AOT，编译和加载 32 个 variants 合计 148.75 秒，使第 4 步达到 182.65 秒，随后第 5/6 步降为 38.30/35.20 秒。这一段有明确编译证据。

数据长尾也确实存在：早期任务第 9 步约 238 秒，其他 rank 等待 microbatch 26，MFU/H100 从前一步约 37.61% 降至 6.27%。rank 0 的 dataloader host timer 只有约 2–3 毫秒，并不能排除其他 rank 拖住整个集体计算。MFU、Tensor Core mmaact、host timer 含义不同，不能把 100%-MFU 解释成 GPU 空闲比例。

MFU 并非缺少实现：`MFUCallback` 默认跳过最初 5 次计数，随后按 `trainer.logging_iter` 窗口上报；当前 logging_iter=1，指标包括 `mfu/H100`。早期已在在线 W&B 与本地 journal 中观察到指标。复用同一个 W&B run ID、每次训练 iteration 又从零开始时，要结合该次 job 的本地 journal 和日志核对，不能单看旧 summary 的 running 状态判断训练存活。 04:00 前后的只读 W&B API 检查确认 `ybainlp/cosmos3_action/buuwsbms` 中已有 `mfu/H100`，旧 summary 的 `_step` 为 36；新任务本地 `debug.log` 明确记录 `run resumed` 且仍复用该 ID。当前训练又从零开始，这与前段指标因步数回退而不更新一致。04:50 再次只读检查确认原 run 的 state=running、summary_step=37、mfu/H100≈0.06245（约 6.25%），已在超过旧步数后恢复新写入；原 run ID 和训练 iteration 均未改动。证据：`outputs/dataloader_fix_20261030/wandb_remote_resume_probe.json`、`wandb_remote_after38_probe.json`。

06:01 只读检查：29fe 任务越过旧步数后，原 W&B run `buuwsbms` 的 state=running、summary_step=43，`mfu/H100≈0.497357`（约 49.7%），并读到第 42 步约 0.44297 的新历史点。说明当前指标已恢复写入；这仍是 MFU 指标，不是 GPU 忙碌百分比。证据：`outputs/dataloader_fix_20261030/failure_micro125_20261030/wandb_remote_29fe_after42_probe.json`。

第 100 步后只读复核：06:35 W&B state=running、summary_step=105、mfu/H100≈0.44595（44.6%），指标持续写入。rank 0 第 52–100 步中位耗时 33.95 秒，范围 31.56–35.82 秒；第 100 步 33.21 秒，真实 token 吞吐约 5368.9 tokens/s/GPU。这些步时/token 指标不能直接给出各 rank 的取数等待或 GPU idle 百分比。证据：`failure_micro125_20261030/milestone_29fe_step100.json`、`wandb_remote_29fe_step100_probe.json`。

第500步附近复核（10:20）：W&B state=running、summary_step501、mfu/H100≈0.44763（44.76%），指标继续写入。第401–500步rank0中位33.88秒、范围31.22–39.84秒。证据：`failure_micro125_20261030/wandb_remote_29fe_step500_probe.json`、`steps401_to500_timing.json`。

代码：`cosmos_framework/callbacks/mfu.py:85`、`:267`、`:450`；默认 callback 接线为 `cosmos_framework/configs/base/defaults/callbacks.py:102`。历史测量证据：`outputs/dataloader_bench_20261029/REPORT.md`。

## 八、64/64 内存实测与“共享存储满”的区分

`prefetch_factor=64` 是每个 rank 在整个混合数据源中的 64 个在途批次，包含执行、IPC、pinning 和 ready；不是每个 worker 64 批，也不是 64×64。当前训练批次、逻辑流 packer 的 carry/pool、Arrow 缓存、模型与节点其他开销仍在此预算之外。语义见 `docs/training.md:637`。

| 本地测试 | 批数 / 条件 | 峰值 PSS（GiB/rank） | 备注 |
|---|---|---:|---|
| 原实现 rank 111，32/32 | 连续 64 批、真实后台 pin | 87.94 | 峰值 15325 线程 |
| 修复后 rank 111，64/64 | 连续 80 批、真实后台 pin | 128.17 | 峰值 174 线程 |
| 修复后 rank 108，64/64 | 连续 80 批、真实后台 pin | 123.66 | 峰值 173 线程 |
| 修复后 rank 111，64/64 | 暂停消费，确实填满 64/64 ready | 125.63 | 填充阶段约 132.54 秒 |
| rank 40，OMP=1，64/64 基线 | 120 批、后台 pin | 139.57 | 2902 线程，260.64 秒 |
| rank 40，episode 缓存 + tokenizer 串行 | 相同 120 批、64/64 | 138.99 | 183 线程；仅清逻辑缓存尚未明显降 PSS |
| rank 40，再加 Arrow 空闲页回收 | 相同 120 批、64/64 | 78.71 | 182 线程，268.99 秒；120 批 source/cursor 一致 |
| rank 40，完整修复，预取确实填满 | 消费 120 批后 ready=inflight=limit=64 | 79.58 | 182 线程；额外填充 222.18 秒，成功 pin 184 批 |
| rank 104，完整修复，新故障位置回放 | 连续 126 批，实际消费 microbatch 125 | 80.85 | 182 线程；目标批准备 18.27 秒，同一 file-288 读取约 4.30 秒 |

旧版本 rank 111 满 64 批时的 125.63 GiB 简单乘 8 约为 1005 GiB/节点，尚不含训练和其他开销。新版本 rank 40 的 120 批快速回放峰值为 78.71 GiB，比同 rank 的 OMP=1 基线低约 43.6%；单次耗时约多 3.2%，不是统计显著性结论。额外的满预取测试已确认 ready=inflight=limit=64、pin_memory_batches=184，整轮峰值 79.58 GiB、182 线程；其中消费阶段约 261 秒，停止消费后再填满约 222.18 秒，总耗时 483.29 秒，不能把包含主动等待的总吞吐用于与快速回放比较。按这一段 rank 40 观测简单乘 8，数据进程树约 637 GiB/节点，仍需加模型/通信/节点开销并考虑其他 rank 和长时间运行差异，不能当作全实验上界。单文件探测显示只清逻辑缓存后 PSS 仍 3.47 GiB，调用 Arrow 回收空闲页后为 0.56 GiB，有效 episode 保留。测试没有进行 192-rank GPU 训练或施加全局 NFS 压力；11.2 秒 simulation 也不是实测 GPU idle。满预取证据为 `failure_micro108_20261030/full_buffer_validation.json`。

`job-6073c1e4…` 长时间运行时节点峰值实际达到 1478.47/1520 GiB，约 97.3%，但所有节点 OOM / oom_kill 为 0；出错节点的峰值约 1262.84 GiB。本次该任务的首因是损坏视频。后续 Success (0) 的共享内存错误已由信号中断独立复现，也不能据此推断 NFS 存储容量已满。共享文件系统容量、tmpfs 共享内存和 cgroup 总内存需要分别取证。

生产内存口径补充（约第 60 步阶段）：24 节点中位 memory.current≈794 GiB，anon≈227 GiB、file≈554 GiB，其中 shmem≈533 GiB；file 不能直接等同于共享存储文件缓存或磁盘占用。另行 statvfs 检查显示 `/dev/shm` 容量约 760 GiB，最少可用约 753 GiB，与 cgroup shmem 是不同口径，不能互相替代；file_dirty/file_writeback 中位均为 0。仅凭 shmem 分类不能确定各类共享映射的具体消费者。证据：`outputs/dataloader_fix_20261030/failure_micro125_20261030/new_run_step60_resources.json`，原始容量在该 job 的节点诊断中。

输出盘容量补充：06:18 对实验输出路径调用 statvfs，文件系统可用约 138341.78 GiB（约 135.1 TiB）。这不验证用户/目录配额，也不能反推早前故障时的容量；目前查询未显示全盘容量耗尽。证据：`outputs/dataloader_fix_20261030/output_storage_capacity_latest.json`。

第 100 步附近复核（实际采集时 rank 0 为 101）：24 节点中位 memory.current≈859.0 GiB、anon≈253.1 GiB、file≈592.9 GiB，其中 shmem≈540.6 GiB，采样节点峰值约 881.6 GiB。相较约第 60 步，各项中位分别增加 memory≈64.6、anon≈26.3、file≈38.5、shmem≈7.3 GiB（shmem 属于 file 子项，不能重复相加）；两次快照不能证明泄漏或定位具体消费者。file_dirty/file_writeback 中位均为 0，未见明显内存压力、OOM 或退出，继续观察后续增长。证据：`failure_micro125_20261030/new_run_step100_resources.json`、`resources_step60_to100_comparison.json`。

第 150 步复核（07:01）：24 节点内存中位 913.9 GiB、anon262.5、file637.4（其中shmem555.5），anon与第125步附近相比变化不到1 GiB；无明显压力、OOM 或退出。rank 0 第 52–150 步中位 33.95 秒、范围31.56–35.82秒；W&B summary_step=150、mfu/H100≈0.47252（47.3%）。当时首个200步检查点尚未发布，继续核验实际保存，未做恢复测试。证据：`failure_micro125_20261030/new_run_step150_resources.json`、`milestone_29fe_step150.json`、`wandb_remote_29fe_step150_probe.json`。

第200步保存核验：07:32确认latest/.save_complete、四类metadata和192份非空数据加载状态文件到齐；游标内容均为global_id600，GA=3。逻辑总量119.7 GiB（model约30.2、optim约89.4、数据加载状态约122.4 MiB）。第200步42.48秒、第201步44.13秒，随后恢复约32–35秒；未做完整恢复测试。保存后约第208步节点中位内存1041.0 GiB、anon270.9、shmem635.4，未见明显压力或OOM。证据checkpoint200_verified.json、checkpoint200_cursor_inspection.json、failure_micro125_20261030/new_run_step200_resources.json。

第250步附近复核：24节点中位memory.current≈1076.1 GiB、anon≈272.0、file≈786.7（其中shmem≈639.1）。相较保存后约第208步，memory中位增加约35.1 GiB，anon约1.2、shmem约3.8，文件类扣除shmem后的中位约增27.6；这是两次采样比较，不能定位具体分配者或排除后续增长。当前无明显压力、OOM或退出。证据failure_micro125_20261030/new_run_step250_resources.json、resources_step200_to250_comparison.json。

持续运行复核：14:42 已确认 800 步 checkpoint 的 latest 指针、`.save_complete`、model/optim/scheduler/trainer 四份非空 metadata 和全部 192 份游标文件；逐份解包后 global_id 均为 2400，调度身份及 rank/world_size 拓扑校验通过。逻辑大小约 119.7 GiB，未执行 800 步完整恢复，600 步完整恢复是此前独立验证。第 751–800 步共 50 条 rank 0 步时，中位 33.92 秒；去掉保存提交步 800 后，49 条中位 33.90 秒、范围 31.25–36.09 秒。保存步 800 为 43.14 秒，801 为 44.68 秒，803 回到 34.22 秒。后台慢读仍有短时压力，但尚无消费等待。保存结束附近实际 803 步的节点内存中位 1046.8 GiB，其中 anon 277.0、file 758.8、shmem 640.5 GiB（shmem 属于 file，不重复相加）；保存阶段资源不能直接当作稳态。证据为 `checkpoint800_verified.json`、`checkpoint800_cursor_inspection.json`、`checkpoint800_file_inventory.json`，以及 `failure_resume_metadata_20261030/new_rune6_steps751_to800_timing.json`、`new_rune6_step800_saved_resources.json`（后两项同子目录）。继续观察保存后内存是否回落，下一次保存 1000 步。

## 验证清单与证据索引

- 数据 / 媒体相关原有回归：`171 passed, 2 skipped, 6 deselected`；排除项包含已有基准断言失败和缺失 DROID fixture，详见 `outputs/dataloader_fix_20261030/REPORT.md`，没有将排除项冒充通过。
- 两个故障 rank 的 64/64 真实数据回放：rank 111、108 各 80 批；rank 111 前 64 批的 source/cursor 与原实现完全一致。原实现本地也通过，故不能证明线上 I/O 长尾全部消失。
- 10 次实际视频请求的自动 / 单线程 RGB 与元数据一致；坏视频用两个独立解码器复现损坏。
- 跳过策略 13 项通过，真实坏视频跳过与补批通过；信号补丁相关 57 项通过，另 1 项因缺外部数据排除。
- 512 MiB 实际 tensor IPC 的诊断信号压力复现，修复前失败、修复后成功。
- Python lint / format、相关 shell 的 bash -n 检查通过。

主要原始资料：`outputs/dataloader_bench_20261029/`（早期 15 组调参、MFU 与慢步）；`outputs/dataloader_fix_20261030/`（队列 / 解码 / 内存回放）；其下 `failure_step38_20261030/`（坏视频和跳过）；`failure_shm_signal_20261030/`（共享内存信号故障）。

## 后续监控与更新方式

只使用共享日志和本机复现，不依赖远端 SSH。每 30 秒读取进度、cgroup 内存/OOM/退出、内存压力、真实配置和等待告警，捕获 `Action128 Parquet read still running` 的具体路径/row group；慢读或观察超时不等于任务已退出。监控输出为 `outputs/dataloader_fix_20261030/training_watch.jsonl`、`training_alerts.jsonl`、`training_watch_live.log`。故障快照自动归档到 `outputs/dataloader_fix_20261030/incidents/<job>/`，并校验日志头中的 job，避免混入其他轮次或被自动重启覆盖。CPU 配额/节流、CPU/内存/I/O 压力及本地 `/dev/shm` 容量诊断已在 c5b3f738 这轮加载；新增 memory.high/low 和 pids.current/max/events 已在 29fe 的全部 24 节点核对加载。

监控格式问题（06:07 修复）：`IterSpeed` 前 50 步打印所有 rank 的 `Time: … | Warmup: …`，之后只在 rank 0 打印 `Iter speed: … seconds per iteration`，原监控仅识别前一种格式，导致显示停在 50，训练实际已到 56。现支持两种格式、保留最大已观察步数以避免旧日志覆盖新进度，并标注 `rank_progress_scope=rank0_after_warmup`；其他 rank 的最后 50 步记录不代表停滞。已用真实第 56/57 步日志验证，重载的只是只读监控，训练未重启，六份运行源码仍与已测清单一致；监控改为持续每 30 秒采样。代码契约：`cosmos_framework/callbacks/iter_speed.py:84`、`:115`；证据：`outputs/dataloader_fix_20261030/monitor_regular_format_validation.json`。

检查点核验：实际 `save_iter=200`、`keep_last_n=3`。读取 `checkpoints/latest_checkpoint.txt` 后，核对 `.save_complete`、model/optim/scheduler/trainer 四份 `.metadata` 和 192 份非空 `dataloader/rank_<n>.pkl`；先看到目录不代表保存已完成。只读脚本为 `outputs/dataloader_fix_20261030/verify_latest_checkpoint.py`，结果写入 `checkpoint_latest.json`；第200步已按这些项目核对完成。文件核验不加载模型/优化器状态；另已检查数据游标内容，未恢复训练，不能称为恢复测试通过。发布顺序见 `cosmos_framework/checkpoint/dcp.py:1300`、`:1305`，latest 文件写入见 `cosmos_framework/checkpoint/base.py:206`。

告警覆盖补充：直接识别 `Watchdog caught collective operation timeout` 并触发故障归档，避免 NCCL watchdog 已报错但尚无 Python 异常时漏掉现场。用 c5 任务归档的 23 个节点真实超时日志验证；普通 destroy_process_group 退出警告不会因此被当成 watchdog 超时。只重载只读监控，不终止或重启训练；告警本身也不证明进程已退出。证据：`outputs/dataloader_fix_20261030/monitor_nccl_alert_validation.json`。

遇到新错误，先确定最早 rank 和原始异常，再处理后续 NCCL / SIGKILL 清理信息。修复后以本地复现、针对性回归、24 节点源码指纹、实际配置和越过故障点的训练进度共同判断效果；不以单机通过或进程仍活着宣称整轮实验已稳定。

恢复选择逻辑复核（07:50，只读）：当前 load_training_state=false、action128_load_mode=warm_start 用于外部初始化。已有同目录 latest 时，代码先选择该保存目录的 model/optim/scheduler/trainer/dataloader 全部状态；同目录恢复将 warm_start 模式归一为 auto，不沿用外部初始化跳过键。当前 keys_not_to_resume=[]。因此在输出根目录和拓扑不变时，按代码应选择本轮200步检查点，不需把外部base路径手工改成200；这不是完整恢复实测，未重启健康任务。代码：`cosmos_framework/checkpoint/dcp.py:676`、`:789`、`:819`、`:858`、`:864`。证据：`outputs/dataloader_fix_20261030/checkpoint200_resume_selection_review.json`。

告警交付补充（08:12）：监控每30秒采样，观察查询约40秒；只看最新快照可能跨过含异常的那一轮。observe_training.py现按日志偏移读取全部新增告警，致命异常保持pending，复核处理后才用--ack-fatal确认；空的新快照不会清掉未确认异常，新追加未读异常也不会被确认参数清掉。初始化审计本轮185条均为启动等待，未发现致命异常。使用隔离的真实旧故障日志通过6项回放检查，覆盖大快照、半行写入、跨采样持久待确认、显式确认及等待告警分级；未向生产日志注入测试告警，也未重启训练。pending告警不等于进程已退出，仍需核对退出/心跳。证据outputs/dataloader_fix_20261030/observer_alert_delivery_validation.json；当前观察结果observer_last_observation.json，状态observer_alert_state.json。

监控去重修复（11:06）：固定字节 tail 窗口在日志增长后可能从旧告警行的中间开始，截断文本与原完整行的去重键不同，导致05:30的启动 microbatch0等待于11:05再次被识别。已让 tail 跳过开头不完整行，同时保留恰好位于行边界的完整行；6项隔离检查通过，包含真实旧日志复现。只重载只读监控，未触碰训练；新的监控工具会话为88988。证据outputs/dataloader_fix_20261030/monitor_tail_boundary_validation.json。

真实慢读观察（11:08–11:14）：node13的RoboCasa365、FastUMI和Cosmos3-DROID等Parquet读取出现23条超过30秒告警，文件row group元数据未压缩量约0.49–1.81GB。该节点诊断memory.pressure full avg10最高采样23.37%，之后回落到0%；memory.events的max/oom/oom_kill均为0。至少6次先前读取已由同一顺序worker的后续不同文件读取证明返回，但没有精确读取耗时或所有预取批次已消费的证明。训练继续到600步，第501–600步中位33.92秒、最长36.17秒，期间未观察到新的消费端等待或退出。未据此修改训练参数、杀worker或重启任务；继续观察是否再次发生。24节点日志/诊断与进展证据在outputs/dataloader_fix_20261030/slow_reads_step590_20261030/，含snapshot_manifest.json、worker_read_progression.json、pressure_episode.json。

内存上限回收观察（11:38）：worker0节点约1507/1520GiB，memory.events.max=198、oom=oom_kill=0。相较第624步附近采样，pgscan增加2166006、pgsteal增加1963325，证明期间发生了回收；当前memory.pressure full avg10=0。anon约293.4GiB，file约1198.6GiB（包含shmem约655.4GiB），非活跃文件页约318.0GiB。第642步仍为33.49秒，无新增消费等待或退出；不能据此保证后续回收没有开销或不会OOM。未修改cgroup限制、丢弃缓存或重启训练。证据outputs/dataloader_fix_20261030/near_memory_limit_step624_snapshot.json、memory_reclaim_step640_observation.json。

真实慢读复现（11:52–11:54）：node13再现5条超过30秒的Parquet读取告警，采样memory full avg10达4.15%，随后回落到告警阈值1%以下。rank0继续完成第668步34.82秒，无新消费等待、OOM或退出；未因该告警修改训练配置。24份日志及24份诊断已归档至outputs/dataloader_fix_20261030/slow_reads_step667_20261030/，读取返回情况继续核对，不能将没有新告警视作全部预取完成。

11:59复核：rank0第677步33.53秒；慢读累计9条，2组同worker换文件返回证据。node13原始诊断近期full avg10峰值至少16.35%，最新已采样12.93%，memory.current约1285/1520GiB、max/oom/oom_kill均0。压力与读取变慢相关，具体来源未确认；无新消费等待或退出，未修改训练代码。
现场：outputs/dataloader_fix_20261030/slow_reads_step667_20261030/。

相机引用覆盖不足（12:13）：rank154在legacy:44源microbatch2110失败，rank0最后完成703步。unified_action128.py的_video在调用decoder之前发现请求末时间超出camera to_timestamp，原来抛CatalogSchemaError，已有媒体跳过策略只处理MediaDecodeError，因此任务终止。仅将这一项覆盖不足改为MediaDecodeError并放入已有相机上下文/decoder清理路径；其他schema错误仍报错，默认未开启跳过时仍抛错。最新600步检查点已再次确认完整；完整恢复尚未测试，平台后续自动重启需要核对新指纹。

覆盖不足修复验证（12:19）：2项定向回归、13项既有媒体策略回归通过，ruff check/format通过。新任务job-5b628c82-baa6-472b-a91b-35fcb5db7b46已开始启动；当前18个节点出现新job诊断头，尚未获取新源码指纹，不能称24节点已加载或已恢复训练。预期unified_action128 SHA256为b2de259b07b5653cb30bd553b3abbf748d7fc84ef752eb1c2abc3fceaed268fe。

重启加载核对（12:22）：job-5b628c82-baa6-472b-a91b-35fcb5db7b46全部24节点的三份关键源码SHA均匹配当前已测版本，unified_action128为b2de259b07b5653cb30bd553b3abbf748d7fc84ef752eb1c2abc3fceaed268fe。指纹实际位于同一启动段诊断头之前，最初只读取诊断头之后漏掉指纹，已修正核对范围，并要求中间无其他诊断头以防串用前轮指纹。当前仍在初始化，尚无checkpoint恢复完成或训练步完成证据。证据failure_camera_coverage_20261030/new_run_deployment.json。

恢复进展（12:25）：job-5b628c82在主节点12:25:10明确选择同目录iter_000000600，加载keys为dataloader/model/optim/scheduler/trainer，证实优先选择最近完整检查点。这里只观察到恢复入口选择，尚未见完成加载或恢复后训练步，不能将其写作完整恢复通过。证据failure_camera_coverage_20261030/new_run_resume_progress.json。

恢复失败（12:26）：job-5b628c82的数据global_id恢复到1800，模型和优化器加载走过后，在scheduler的DCP加载路径报CheckpointException/metadata is None，多节点退出，完整恢复失败。PyTorch当前实现会吞掉read_metadata原始异常。对同一600检查点，本地scheduler/trainer metadata均可读取，scheduler实际DCP加载也通过（last_epoch600、step_count601），故尚不能断定文件损坏或共享存储满。共享源码补充scheduler加载前显式metadata检查以保留原始异常，launcher增加dcp.py指纹；这是诊断补强，未宣称故障根因已解决。平台已启动job-466efc10-d4d0-4b2e-9402-4f40573f2372，其是否包含这项后加诊断须继续核对。现场incidents/job-5b628c82-baa6-472b-a91b-35fcb5db7b46/，本地证据failure_resume_metadata_20261030/summary.json。

恢复再次失败（12:31–12:32）：466轮仍在scheduler路径报metadata is None，24节点退出；该轮日志未输出后加的dcp.py指纹，不能确认诊断已包含。平台启动job-6687f278-253f-47fd-a5c2-873a4106beb4，继续核对四文件指纹和底层异常。466现场见incidents/job-466efc10-d4d0-4b2e-9402-4f40573f2372/。没有跳过scheduler或重置训练状态。

原始原因及可见性处理（12:36）：668轮24节点已加载dcp.py显式metadata检查，原始异常为FileNotFoundError [Errno2]，目标600/scheduler/.metadata；同路径在当前共享存储端可读且本地实际DCP加载成功。确认是训练节点读取时找不到文件，尚不能单凭此确认具体挂载/缓存故障机制。保留原件后，对scheduler/trainer共195个非空文件（约4.99MB）用相同内容、fsync后原子替换重新发布，所有SHA256前后一致；不改模型、优化器、latest或保存完成标记。此处理旨在刷新目录项可见性，仍需下一轮验证，不能宣布已恢复。证据failure_resume_metadata_20261030/small_state_republish.json、before_republish/；668现场已归档。

恢复加载成功（12:39）：重新发布文件内容后，e6轮通过scheduler和trainer加载，12:39:21明确Loaded checkpoint ... in iteration 600，耗时33.39秒；这次实际加载验证成功，但首个恢复训练步仍待出现。不能断定底层共享文件系统可见性问题永久消除。证据failure_resume_metadata_20261030/new_rune6_resume_progress.json、new_rune6_deployment.json、small_state_republish.json。

恢复后继续训练验证（12:48）：e6已完成601、602、603步，rank0最新603步32.69秒。全部192个rank都有至少601步的完成日志，证实完整恢复后实际继续训练。首步rank0记录533.85秒，包含本轮恢复与冷启动开销，不能当成稳定步时，也不应全部归因于同一环节；tokenizer准备已在192个rank完成，约199.82–228.54秒。当前无新增数据等待、OOM或致命错误。原相机覆盖失败发生于micro2110，对应704步附近，仍需继续验证。证据failure_resume_metadata_20261030/new_rune6_first_steps.json、new_rune6_resume_progress.json、new_rune6_aot_progress.json。

相机覆盖跳过实际生效（13:32记录，13:38复核）：e6的rank154、legacy:44已将sample706517、episode41369/window4（8.0–8.9668秒）的相机引用覆盖不足记录并跳过，camera main请求56.6667–57.6333秒，对应file-331.mp4；accepted188、skipped1、consecutive1。该记录证明线上已处理这一真实报错类型，训练继续到691步；旧轮仅记录了micro2110/source，未记录具体样本身份，不能单凭此断言已消费原失败批次。仍需等待消费越过2110。证据failure_camera_coverage_20261030/new_rune6_real_skip.json，包含当前job日志关联。

旧故障关口通过（13:44:54完成，13:46核对）：e6的rank0完成704步34.60秒。GA3下该训练步消费micro2109/2110/2111，结合rank154/legacy真实覆盖不足窗口已被跳过，说明任务已越过旧轮失败步；此处依据完成日志与训练循环，不伪称已有新的192份游标快照。新一段慢读共有9条、0组先前读取返回证据，已归档slow_reads_step704_20261030/；目前无消费端超时、OOM或致命错误，不因该告警修改训练参数。关口证据failure_camera_coverage_20261030/new_rune6_step704_passage.json。继续监控长期资源和第800步保存。

慢读压力复现（14:00）：e6的node1出现新一批FastUMI等Parquet慢读，13:42以后累计22条；该节点memory full avg10采样峰值17.7%，最新12.55%，memory.current约959.5/1520GiB且max/oom/oom_kill为0。rank0仍继续到730步33.45秒，无消费端超时或退出。已将日志、诊断和压力原始样本追加归档slow_reads_step704_20261030/；继续观察，尚未归因或新增运行时修改。

保存后高压力跟踪：14:52 训练仍在推进，但第 814/815/816/817 步分别为 68.64/62.82/64.43/100.14 秒，明显慢于此前约 34 秒。node14（diagnostic worker13）的 memory full avg10 从约 24% 升至约 77%，同节点多个源、包括未压缩约 0.5 MB 的 row group 都出现慢读；14:50 该节点使用约 1100/1520 GiB、host available 约 922 GiB，memory.events high/max/oom/oom_kill 均为 0，不能据此认定容量耗尽。尚无本阶段持续 60 秒的消费等待日志，短于诊断间隔的等待仍可能发生，不能按步时精确归因。本机对其中 file-383.parquet 的 row group33 实际读取成功，open 0.068 秒、read 0.125 秒；这只排除了该次本机读取失败，未复现线上节点压力。已归档 24 节点共 48 份日志/诊断至 `pressure_step813_20261030/`，后续现场及本机探针另存该目录。异步保存的 CPU staging 缓冲原本保留并复用（`cosmos_framework/checkpoint/dcp.py:1152`），首次保存后的 shmem 台阶不能直接视为泄漏，也不能解释全部压力；观察记录见 `post_save800_20261030/staging_memory_observation.json`。

持续慢步补充：截至 15:12，818/819/820/821/822/823/824/825 步分别为 96.96/111.18/129.33/141.56/117.08/135.84/191.03/228.31 秒，任务仍推进；node14 memory full avg10 最高已观察约 83%，最新约 79%。尚无本阶段 OOM、退出或致命错误。W&B 已核对收到 821 步，MFU 为 11.86%。完整步时及最新现场见 `pressure_step813_20261030/steps814_to825.json`、`resource_growth803_to825.json`、`failure_resume_metadata_20261030/new_rune6_step825_resources.json`。原有消费等待栈每 60 秒打印，缺少较短等待计时，不能由无诊断日志推断没有短时等待，也不能按相关性唯一归因 GPU 等待。已在 `shared_prefetch.py:570` 补充至少 10 秒且已结束的 receive 日志，计时包括 IPC 和后台 pin；12 项多进程回归（包括阈值与原批次/credit 行为）以及 lint/format 通过。当前 e6 仍加载旧源码，新诊断 SHA 为 574bb70e3d1dbdce83cc523040ace7d7ec93856b8a92815a5de0300e3ee2f19e；平台下次重启才加载，未主动重启训练。该补丁是观测补充，不是对当前高压力的根治。验证与指纹见该压力目录的 `receive_wait_patch.json`、`receive_wait_regression.log`。

## 第 831 步数据超时与 800 步恢复故障

e6 最后完成 830 步。node14 / rank54 在 microbatch2490（831 步）等待 `InternData-M1-eef:33`，physical worker33 / pid1222；其栈停在 PyArrow `read_row_group`，pin 线程在等待结果，后续多批已 ready。15:35 采样确认最终 450 秒 TimeoutError，随后出现分布式退出告警。此前 node14 memory full avg10 持续约 80%–88%，未见 OOM 或 memory.max 事件。故障路径明确为数据读取；独立的 OS/共享存储根因仍未证明。未记录到该次调用的确切文件路径，不能把 worker 早先的 file-053 直接认定为当前文件。

修复：大 group 的 episode 提取改用 8192 行的 Arrow record batch 逐批过滤，保留原 metadata 大小限制、行顺序、类型、空值、连续索引和 episode 边界校验。仍扫描所需 group，不跳过合法帧。14 项已发布实现回归及 lint/format 通过。真实 M1 file-053 所选 episode 输出校验值一致，单进程 RSS 4.08→0.40 GiB；不是全局压力复现，不保证所有存储长尾消失。证据：`failure_micro2490_20261030/summary.json`、`streaming_patch_report.json`、`published_streaming_regression.log`。

平台随后自动启动 3c 任务；24 节点已加载短时 receive 诊断，但代码复制早于分批读取发布。该任务在恢复 800 步时打开 `trainer/.metadata` 返回 FileNotFoundError，而当前共享挂载可读取。已备份并同内容原子重新发布 scheduler/trainer 的 195 个非空文件，共 4,991,782 bytes，逐文件 SHA256 前后一致；未改模型、优化器、latest 或完成标记。本机重新读取 scheduler/trainer metadata 分别得到 8/1155 项。具体可见性机制尚未确定。现场及完整 manifest 在 `failure_resume800_metadata_20261030/`，3c 全节点自动归档在 `incidents/job-3c7b5d3e-da56-4c45-a4c0-4fa7534303c9/`。

93 任务 24 节点已加载分批读取 SHA `00e50d5b5fe60284d8c83fd0075ec3eea051d648c2bf73ceefaa310999fc4c81` 和 receive 诊断 SHA `574bb70e3d1dbdce83cc523040ace7d7ec93856b8a92815a5de0300e3ee2f19e`。完整 checkpoint 加载及全部 192 ranks 继续训练已验证；801 步冷启动 574.22 秒，802/803 步为 31.17/33.64 秒。继续观察 831 步与下一次保存 1000 步。未手动重启健康任务，也未把本地验证当作线上故障已彻底解决。

93 运行复核：截至 16:02 的完整 24 节点原始日志中，首批 micro2400 已在全部 192 ranks 接收，耗时 63.47–84.80 秒、中位 75.33 秒；之后 micro>=2403 暂无已结束且至少 10 秒的 receive 日志，不能排除更短等待或 receive 之外的 GPU 停顿。192 ranks 的 tokenizer AOT 均完成，耗时 182.77–204.10 秒。802–810 步 9 条 rank0 步时中位 33.64 秒、范围 31.17–34.48 秒。实际 810 步节点内存中位 697.7 GiB、anon191.1/file496.2/其中 shmem489.0 GiB；仍在预取和缓存填充，不能当成长期稳态内存。证据位于 `failure_micro2490_20261030/new_run93_steps802_to810_progress.json`、`new_run93_step810_resources.json`、`new_run93_initial_receive_waits.json`、`new_run93_aot_progress.json`（均同子目录）。原故障 831 步随后已由全部 192 ranks 通过，证据见 `new_run93_step831_passage.json`。

831 关口复核：16:13:23，旧故障 rank54 直接日志确认第 831 步完成，33.87 秒；全部 192 ranks 都有该步精确完成记录。该步包含原超时的 microbatch2490；未额外采集此时的实时数据游标，不能把预期 committed_global_id2493 当成已读取的游标值。数据读取已越过该编号关口，但恢复起点和节点分配不同，不能单凭这次通过分离修复与重启的贡献或宣称所有 I/O 长尾已消失。证据在 `failure_micro2490_20261030/new_run93_step831_passage.json`。

93 持续运行复核：截至第 850 步，全部 192 ranks 有该步直接完成记录；802–850 步共 49 条 rank0 步时，中位 33.88 秒、范围 31.17–35.73 秒。完整原始日志中，首个冷启动步之后仍暂无已结束且至少 10 秒的 receive 记录；不能排除更短等待或 receive 之外的停顿。实际 850 步节点内存中位 806.8 GiB、anon220.0/file573.7/其中 shmem527.5 GiB；较 825 步中位分别增约 51.0/11.9/39.6/17.0 GiB，仍不能认定内存稳定。报告在 `failure_micro2490_20261030/new_run93_steps802_to850_progress.json`、`new_run93_step850_resources.json`、`new_run93_resource_growth825_to850.json`（同子目录）。恢复后的前 50 步结束，其他 ranks 的步时停在 850 是日志策略；851 的 IterSpeed 初始化空档后，852 已正常打印 34.17 秒。


93 长期运行复核（20:27）：rank0 已到 1207 步，该步 109.74 秒。1101–1200 的 100 条步时中位 34.66 秒、最大 234.01 秒；1164–1180 连续多步较慢，不能沿用最初约 34 秒的短窗口认定长期步速稳定。20:26 全量扫描当前 24 份节点日志，192 条已结束且至少 10 秒的 receive 均是冷启动 micro2400，此后暂无此类日志；192 条持续等待诊断同样属于 micro2400。此观测不能排除短于阈值的等待、receive 以外的加载开销或其他停顿，尚不能把当前慢步直接归因于 worker 供给不足。审计见 `failure_micro2490_20261030/new_run93_longtail_receive_audit_latest.json`。

第 1206 步附近 24 节点 cgroup 内存中位约 1270.2 GiB、anon307.8/file950.9/其中 shmem657.1 GiB；file 与 shmem 不能重复相加。最高节点约 1306/1520 GiB，worker2 的 memory full avg10 一度超过 70%，未见 memory.events 的 OOM/max。共享存储慢读告警仍持续，包含很小的 group；大 group 分批读取降低局部解码峰值，不能据此宣布整机资源问题已解决。资源快照见 `failure_micro2490_20261030/new_run93_step1206_resources.json`。

1000、1200 步 checkpoint 均有保存完成标记和 model/optim/scheduler/trainer 非空 metadata；实际读取全部 192 份游标，global_id 分别为 3000/3600，schedule、execution version 和 DP topology 校验通过。证据为 `checkpoint1000_cursor_inspection.json`、`checkpoint1200_cursor_inspection.json`、`checkpoint1200_verified.json`（均在 `outputs/dataloader_fix_20261030/`）。这不等于已实际恢复 1000/1200；当前任务实际恢复验证仍为 800 步，不主动重启健康任务。W&B API 20:27 核对收到 step1206，MFU/H100=0.322528，证据见 `failure_micro2490_20261030/wandb_remote_93_step1206_probe.json`。


慢步计时进一步定位（20:31）：通过 W&B API 实际读取 1160–1206 范围内同时具备所选字段的 47 条记录。15 个步时小于 40 秒的样本中，rank0 `timer/optimizer_step` 中位约 2.70 秒；24 个大于 60 秒的样本中，该值中位约 75.35 秒。两组 rank0 `timer/dataloader_train` 分别约 0.0042/0.0048 秒，forward/backward 主机计时相近。1177 步总耗时 234.01 秒，其中 optimizer 区间 184.99 秒，worker2 的同期 memory full avg10 峰值 79.46%；1186 步总耗时 31.72 秒，optimizer 区间 1.76 秒，各节点压力已回落。这是同期相关性，不是根因证明。1201 步总耗时 151.31 秒但 optimizer 仅 5.73 秒，说明不能用同一个区间解释全部慢步。

这些 timer 由 rank0 发布，`dataloader_train`/forward/backward 是 GA3 三个 microbatch 的平均主机时间，optimizer 每步一次；不是 CUDA kernel 精确计时，也不是全 rank 最大值。optimizer 区间包含 callbacks、model hook、更新及清梯度（`cosmos_framework/trainer/__init__.py:455`）；例如 SkipNaNStep 在该区间内 all_reduce 并读取 GPU 标量（`cosmos_framework/callbacks/skip_nan_step.py:54`），等待别的 rank 也可能记在这里。因此目前证据更支持优先排查高压力节点和同步长尾，不支持直接认定 rank0 worker 不够，也不能直接认定 Adam 更新计算变慢。原始 API 结果与 24 节点同期压力分析分别见 `failure_micro2490_20261030/wandb_remote_93_longtail_timers.json`、`new_run93_phase_pressure_audit.json`（同子目录）。20:33 复核，1210/1211 步分别为 46.23/38.41 秒，worker2 memory full avg10 降至 1.23%；仍有节点内存约 1315/1520 GiB，压力回落不代表长期内存增长已解决。未见新退出/OOM/致命错误；未改训练参数或主动重启。


内存增长分类复核（20:37）：线上 DeviceMonitor 在 1000/1200 两次保存附近的 192-rank 汇总显示，节点 cgroup current 平均约 1085.63→1260.79 GiB（+175.17）；其中 anon +41.82、file +132.97、file 内的 shmem +15.84，因此 file-shmem 增约 117.14 GiB。这个区间的增长不能全部解释为更多在途批次或 pinned 数据；存在明显的其他文件页增长，但尚未证明具体文件/回收机制。不同来源的均值、节点中位数和进程指标不可混用。同期每 rank 的进程树 PSS 平均约 113.71→120.93 GiB，pinned allocated 约 72.45→74.43 GiB；pinned reserved 指标为 -1（不支持），不能视为 0 或据此排除分配器缓存。DeviceMonitor 是保存附近的瞬时采样，其 GPU util=0 不能代表训练全程 GPU 空闲。原始日志在 action128_node0.log:12864 和 :24489，解析记录见 `failure_micro2490_20261030/new_run93_device_memory1000_to1200.json`。

20:37 训练继续到 1218 步，31.76 秒，暂无新消费超时、OOM 或退出；worker18 的 memory full avg10 仍约 42.75%，说明单次压力采样和步时不总是一一对应。1207–1215 的新 API 计时也验证了慢步后的 optimizer 主机区间从约 75–106 秒回落至 2–3 秒，rank0 取数平均每 microbatch 约 0.004–0.006 秒；结果见 `failure_micro2490_20261030/wandb_remote_93_steps1207_to1225_timers.json`（文件名表示请求范围，本次实际返回仅到 1215）。保持训练配置，继续观察，不把上述分类或相关性视为根治证据。


20:40 定时资源复核：训练已到 1222 步，1212–1222 共 11 步中位 33.53 秒；重新扫描当前全部 24 份节点日志，冷启动之后仍无单次至少 10 秒已结束 receive 或持续 60 秒等待诊断。24 节点 cgroup 内存中位 1284.06 GiB，anon315.55/file959.18/其中 shmem657.30 GiB；比 1215 步中位总量增约 1.46 GiB（7 步），只能说明本次短窗口增幅较小，不能认定长期稳定。最新采样未见 full avg10 至少 1% 的节点，无 OOM/退出/致命错误。证据为 `failure_micro2490_20261030/new_run93_resources_2040.json`、`new_run93_recovery_2040.json`、`new_run93_resource_growth1215_to1222.json`（同子目录）。继续监控，下次资源核对 1250 步或 20:55，下一次 checkpoint 为 1400。


20:46 后压力和慢步再次出现：1233/1234/1235 步分别为 37.62/84.54/100.01 秒。20:49 主节点 memory full avg10 约 74.95%、worker2 约 70.16%；无 OOM/max、退出或消费超时。20:46 的 24 节点资源中位仍约 1284.35 GiB，anon312.61/file957.71/其中 shmem657.06 GiB；总体中位变化小不能排除个别节点的严重压力。已保存相关 4 份节点日志及 3 份诊断日志，共 7 份，见 `failure_micro2490_20261030/pressure_2046/manifest.json`；资源、步时和消费等待复核分别为 `new_run93_pressure2046_resources.json`、`pressure_2046/progress_latest.json`、`pressure_2046/consumer_wait_audit.json`（都在该故障子目录下）。全量日志冷启动后仍无超过阈值的 receive 完成/等待记录，不能排除较短等待及其他环节。

20:50 API 已实际收到 1234 步计时：rank0 dataloader 平均每 microbatch 0.00642 秒，forward19.50 秒、backward6.25 秒，optimizer1.90 秒，整步84.54 秒。该次明显增长在 forward 的主机计时区间，和此前1177步主要增长在optimizer区间不同，不能把所有慢步统一归为优化器更新变慢。GA3下前三项为microbatch平均值，optimizer每步一次，均非CUDA kernel精确计时；内存压力的同期出现也不是唯一根因证明。API返回截至1234，尚不能用它代替1235的分段证据。见 `failure_micro2490_20261030/wandb_remote_93_pressure2050_timers.json`。保持训练参数，不主动重启仍在推进的任务。
