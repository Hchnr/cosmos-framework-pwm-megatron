# Megatron TP/PP 实现与验收

当前状态（2026-11-19）：**TP/PP 为显式选择的实验后端，尚未完成正确性和生产容量验收。标准 FP32 连续三更新测试失败：新输入第三步梯度相对 L2，TP2/PP2 为 0.57724，TP4/PP1 为 0.10917。已完成同 microstep/rank batch 的 TP4/FSDP4 七更新性能对比（前二预热、后五测量）：prepared 计算平均 8.42588 / 22.82432 秒每更新，整机吞吐 3.91650 / 5.78330 样本每秒，TP 为 FSDP 的 67.7%；不含 VAE/noising、媒体解码。含 VAE/noising 的首轮受到并发 CUDA 初始化干扰，只作暂定结果，仍需无干扰重测。真实 12 槽 BF16 严格重启通过不代表 FP32 数值或完整 576 槽容量通过。当前提交保存实现和诊断进度，不能作为验收通过的生产交付。**

范围和最终门槛仍遵循 [v0.0.1 方案](plans/v0.0.1.tp_pp.md)。现有 `megatron` FSDP 后端保持独立。

## 已实现

- `model/generator/megatron_tp_pp/layers.py`：真正的 MCore Column/Row Linear，Q/K/V 与 gate/up 分别切片，三塔各自保留独立 hidden 宽度及局部 Q/KV heads。复用原有三塔 attention、MRoPE、mask 和双向 KV 交换。保持参数名和 UND 冻结。
- `stage.py`：按全局层号构造本 stage 的层；`factory.py` 提供原始 Nano + width_only_1b 的 meta 层工厂。异宽 U/V/A 打包为一个拥有存储的连续 tensor，兼容 MCore 输出释放和反向 P2P。
- `io.py`：首 stage 独占 vocab embedding、输入投影、timestep 和归一化；末 stage 独占最终 norm、vocab/head 和输出投影。复用原生 prepare/decode 数学，原生 dummy head 所需的极小可微 context 一起传输。非空文本 CE 尚未支持，遇到时直接报错。
- `transport.py`：用明确的 JSON/tensor schema 进行 TP owner 广播和 PP metadata 传输，不传 pickle 或 kernel 对象；同 dtype 合并 buffer，按槽位校验，在接收端重建 attention mask。通道挂接 MCore 的公开 P2P communicator 边界，避免 module.forward 内发送与激活接收相互等待。
- `training_backends/megatron_tp_pp/driver.py`：MCore non-interleaved 1F1B、真正 DP group 的 MCore DDP / distributed optimizer；一次调用处理完整更新，仅 step 一次。Q/K norm 的局部 head 梯度单独求和；复制 residual norm 不重复求和。
- `compile.py`：编译时使用可追踪的 functional TP collectives 和原生 MCore 局部 GEMM/autograd；eager 保留原生异步 Column dgrad。窄范围适配阻止未使用梯度被物化为零；payload 反向带每流使用标记，optimizer 在 DP 汇总使用状态后保留原生 `grad=None` 跳过语义，避免错误衰减权重/moments。支持逐 block compile + full AC 的小容量数学对照。
- `batch.py`：逻辑槽位定义和原 micro-step / 模态分母的显式 loss 权重。576 槽位的拓扑排列及非均衡样本/混合模态权重有 CPU 数学测试。
- 原生 `OmniMoTModel._compute_losses` 接受可选的预计算 sample 权重，供逻辑回放绕过物理 WORLD 样本数归约。原路径默认不变，video/action 的损失、维度/时间/空间 mask 及梯度有独立回归对照。
- `optimizer.py`：真实 FP32 master 与 DP-owned Adam fragments 的映射。显式处理 TE/Apex FusedAdam 的参数组 step 与 PyTorch AdamW 的每参数 step，避免续训时重置 bias correction。
- `scheduler.py`：复用原生 LR lambda，按 base LR 身份映射源组并恢复 LambdaLR 位置/计数；不依赖新旧组顺序相同。CPU 测试覆盖分组重排、后续 LR 和运行时状态重启；实际全模型导入也确认 epoch=1200 及 1×/5× LR。
- `checkpoint.py`：同步、完整的 MCore model/optimizer 保存和同拓扑恢复。模型使用 canonical FQN/TP 轴；optimizer 使用安装版本可用的 `dp_reshardable` 格式。全部组件完成后发布 `.save_complete`。跨拓扑完整恢复及异步保存尚未支持。
- 正式 driver 在完整保存并原子发布 latest 后执行与基线一致的 newest-N + 周期 milestone 保留；排除 symlink、未来/未完成 checkpoint 及其他格式。
- 保存复用固定 MCore 版本的 tensor/replica 翻译，将 Torch DCP planning collective 放到 CPU/Gloo；保存前释放未使用 CUDA cache。恢复结束显式清理 DCP 临时引用，防止大容量训练被 setup staging 挤占。四卡机制保存/退出/恢复的 FP32 对照已通过，实际整模型也已完成恢复后的完整保存。
- `tools/convert_cosmos_checkpoint_to_megatron_tp_pp.py`：源 DCP 只读；按 tensor 有界读取 TP 切片，逐项迁移计算权重、真实 master、moments、step、超参，并保留 trainer/scheduler 与全部逻辑 loader/RNG。输出独立 canonical 导入格式及校验 manifest。部分诊断转换不会发布完成标记。
- `bundle.py`：按 stage 读取 canonical 文件，用 mmap 有界导入，严格检查 canonical shape、覆盖和冻结；在 optimizer 建组前保留源 LR/WD 身份，导入真实 DP fragment。Action128 extend 复用原生契约校验，只允许保留旧域 affine 的追加扩展。实际四卡整模型 FP32 model/Adam 导入已通过；完整训练续训尚待验收。
- `logical_loader.py` / `rng.py`：将多个原生逻辑 DP 流复用到一个有界 worker 池，保留各流 cursor、source selection、worker RNG；按逻辑 rank 切换原生 CPU/CUDA/Python/NumPy 训练 RNG。小规模原生样本/增强/游标对照、改变物理 worker 数的重启和独立 CUDA 噪声 RNG 对照已通过；真实 192 流尚待验收。
- 原生 `build_network_config` 和 `prepare_generation_training_inputs` 已提取为共享 helper；`preparation.py` 复用原生 VAE/noising 与 loss，不构造完整 decoder。主训练入口通过 `train.py` 接入独立 update driver；owner VAE 的构造同步和编译 collective 显式使用 owner DP group。
- `planning.py` 用两次有界流式读取计算原 micro-step / 模态分母，再逐槽位校验原始样本/packing/增强 SHA256。仅保留 counts/hash，VAE/noising 只执行一次；额外 CPU 解码成本仍需性能测量。
- 逻辑 loader 每个物理 worker 最多保留两个活跃 packer/media 流；逐流淘汰时使用原生已交付游标与增强 RNG，避免 192×58 个活跃解码缓存。inline/fork/跨物理 worker 数恢复已有回归。

上述代码路径均在 `cosmos_framework/` 下，未向 inference 引入训练依赖。

## 复现诊断

正式配置为 `examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml`。保留原始 BF16、FA3/Flex、packing、逐 block compile/full AC、优化器和 LR；默认 TP2/PP2、逻辑 DP192、576 槽位，外层 GA=1。完整训练尚未验收；默认路径指向当前机器的只读源状态和独立 canonical 导入目录：

```bash
bash examples/launch_pretrain_state_h1_58src_megatron_tp_pp.sh
```

当前driver已接入原生Adam/裁剪算术，并默认每16个pipeline backward回收不可达Python循环，实际执行记录写入job root的 `microbatch_gc.rank*.jsonl`。`COSMOS_TP_PP_MICROBATCH_GC_INTERVAL=0` 可禁用；原始576槽decoder诊断曾在不回收时OOM，生产容量/完整恢复仍需正式运行验证。该策略不调用 `empty_cache`，不卸载激活、不缩小输入。严格重启工具的同名CLI参数须显式设为16（工具默认0），它会检查每rank/更新的实际回收边界。

最近一次原始global576生产运行已完成全部forward/backward及GC边界，但在首个optimizer的梯度norm NCCL all-reduce报CUDA OOM；没有完成update或发布checkpoint。GC16不能据此视为整次更新容量已验收。失败现场及后续collective预热候选见文末。

拓扑与逻辑槽位参数统一位于 `[model.parallelism.megatron]`；物理 DP 由 WORLD/(TP×PP) 推导。运行时在独立 job 目录写入 `initialization.rank*.json` 和 `updates.rank*.jsonl`，保存格式与原 FSDP 隔离。launcher 继承基线 episode metadata cache、大页和 CPU 线程设置。

单遍路径的 `replay_plan.iter*.rank*.json` 在optimizer更新成功后发布，准备完全部spool文件仍不能提前进行完整输入清单比较。日志 `prepared slot N/M` 表示已供给槽位，不等于已完成N次backward；GC日志在schedule返回时记录实际backward数及完成状态。

本机初始化证据为 `artifacts/megatron_tp_pp/entry_initialization_fp32.rank*.json`：四个训练进程正常退出，iteration=1200、stage 层 0–17/18–35，owner 有完整 192 份 loader/RNG，保留原生 52→58-source 扩展 reset。此次 max_iter=1200，没有执行 optimizer update。

真实媒体、原始 packing、BF16/full AC、关闭 block compile 的 12 槽位诊断完成分母预扫描和首微批前向，随后在 stage0 的 VIDEO MLP 反向重计算触发 OOM（请求 700 MiB，PyTorch 已分配 69.92 GiB，设备仅余 513.94 MiB）。该运行没有完成更新。

后续 compile/full AC 也在默认 allocator 下遇到碎片相关 OOM（已分配 68.11 GiB、缓存未分配 5.75 GiB）。仅启用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 后，原始 packing 的真实 12 槽位更新完成：iteration 1200→1201，四 rank grad norm 均为 0.15925516784582952，共 155 样本、696320 有效 VIDEO tokens、6430 ACTION tokens。整次训练更新约 217 秒，其中分母预扫描约 63 秒，峰值 PyTorch 已分配显存 68.83 GiB/owner、67.52 GiB/首 stage TP peer、55.57 GiB/末 stage。这是首次更新，包含 warmup/cache 成本，不能当作稳态吞吐。launcher 现在默认 expandable allocator，显式用户 allocator 环境变量优先。

1201 checkpoint 已完整保存并由新进程恢复。清理 DCP 暂存后，恢复前后实际 CUDA 分配量严格相等：首 stage 40428334080 bytes，末 stage 40070441472 bytes。随后 1202/1203 两个更新完成，完整 1203 checkpoint 也已发布，进程正常退出；报告为 `artifacts/megatron_tp_pp/real12_bf16_resumed_trajectory.rank*.json` 和 `real12_runtime_checkpoint_1203_manifest.json`。其中 VAE AOT 编译关闭；生产 AOT 路径尚待验收。

性能状态（2026-11-18）：上述真实 12 槽作业后续更新为 157.43–169.83 秒，其中分母预扫描 36.35–45.53 秒，约占对应整步的 22%–29%。原始记录为 `outputs/megatron_tp_pp/cosmos3_action/action128/tp_pp_real12_bf16_compile_ac_expandable/updates.rank0.jsonl`；包含两次恢复轨迹，不能当作连续长期稳态样本。预扫描会额外遍历、解码并哈希原始媒体，是已经量到的开销；其后 CPU 准备、GPU 计算、TP/PP 通信及等待的贡献尚未由生产 profile 分解。此前 FSDP2 40.728 秒、Megatron-FSDP 39.078 秒来自另一后端的四卡对照，不能作为 TP/PP 的加速证明，也不能直接组成严格同口径的减速倍数。目前没有 TP/PP 性能收益结论。待完成同输入、相同 VAE/compile/AC 开关和计时范围的稳态对照，以及完整 576 槽更新的 Nsight 时间线，再给出吞吐、通信/气泡、数据准备和显存结论。

当前配方的 `sample_level_loss_averaging` 为 false，rank-level 平均不需要预先收集样本分母。现新增单遍路径：消费时记录完整 raw hash/样本计数，更新结束归约报告计数；sample-level 平均或 `COSMOS_TP_PP_VERIFY_RAW_REPLAY=1` 仍采用双遍预扫描/回放。CPU 回归已验证两种路径的全部输入 hash、计数及最终 loader/增强 RNG 状态相同，连续两个更新、0/2 worker 的四项回放测试已通过，真实 GPU 吞吐收益尚待测量。报告新增 `raw_data_preflight`、`tokenizer_setup_seconds` 和 `owner_host_seconds`，分别记录 VAE cache 设置、读取、哈希、准备 enqueue、验收 trace/capture 的主机耗时；后者包含等待与异步 enqueue，不能当作 GPU kernel 独占时间。数据准备的细分 NVTX 与已有层/通信标记用于后续生产 profile。

旧性能窗口的口径补充：TP/PP 的 157–170 秒来自两次独立 1201→1203 恢复，每次连续两步，共四次 iter 执行；另有首次 1200→1201 的 217 秒。每个 iter 都为全局 12 个 packed microbatch。四卡 FSDP 是 DP4×GA3，TP2/PP2 是 DP1×有效 GA12；TP/PP 的累积由 Megatron schedule 内部完成。两者全局槽数一致。旧 TP/PP 计时步各含 97/100 个实际样本，旧 FSDP 计时的 1204/1205 含 85/124 个样本，均关闭 VAE AOT，但预热和输入窗口不同。逐步计数见 `artifacts/megatron_tp_pp/historical_batch_size_comparison.json`，不能把重复恢复当作四步连续稳态，也不能以 GA 不同推断全局数据量不同。

固定四卡、每步 12 槽和原 microbatch 大小时，GA 差异来自 DP4→DP1 的拓扑变化，不是已验证的显存不足。TP2/PP2 若也设 GA3，要么每步只处理 3 槽，要么每个 microbatch 承载原来 4 槽的数据；后一种合批的显存尚未测试，不能断言放不下。保持原 microbatch、GA3 和全局 12 槽需要 TP2×PP2×DP4，即 16 卡。

新增匹配窗口为双方从 1200 连续五步、前三步预热、后两步测量，均全局 12 槽、BF16、FA3/Flex、compile/full AC、确定性配置相同、VAE AOT 关闭；短程测量不替代完整 576 槽验收。`COSMOS_TP_PP_PROFILE_ITERATION=1203` 可将捕获延迟至预热后的更新，与 `COSMOS_TP_PP_PROFILE_SLOTS` 配合；后者按各自 DP 副本的局部 microbatch 编号解释。`tools/compare_tp_pp_performance.py` 检查每个槽的 raw/完整 prepared/noise hash、source、样本和 token 计数，拒绝重复 iteration，以每步最慢 rank 的耗时计算吞吐；输入不匹配时不输出匹配吞吐比，并固定检查 BF16 loss 相对差不超过 1%。其三项回归已通过，实际结果仍待运行。`checkpoint_timings.rank*.jsonl` 单独记录同步保存时间，不混入 `update_seconds`；forward/backward 和 optimizer+sync 的主机耗时也单独记录，不能当成独占 GPU kernel 时间。

`tools/verify_tp_pp_native_data_replay.py` 另行构造四个独立原生 loader，验证全部 12 槽位实际媒体、packing、增强和 worker RNG 的完整 hash，与逻辑 loader 逐项相等；报告为 `real12_native_raw_data_parity.json`。恢复 1203 后，独立原生 loader 的全部 12 槽位也一致，见 `real12_runtime_native_raw_data_parity.json`。原生 VAE/noising 对照直接恢复各 rank RNG，调用原始 Omni helper，未使用逻辑 RNG 包装器；全部 12 槽位的实际 epsilon/noised tokens/targets/timesteps 及完整 prepared packing 都一致，见 `controlled12_deterministic_native_prepared_parity.json`。这些结果不代表完整 192 流已经验收。

重复 BF16 续训中，raw hash 一致，但 grad norm 仍有明显差异（1202 的两次观测约 0.19855 / 0.14891）。从相同不可变 1203 checkpoint 做确定性 cuDNN 的 A/B 回放，实际 raw/noise hash 和 loss 完全一致，但 Adam moments 仍有参数桶超过 `1e-4` 门槛；报告 `controlled12_runtime_state_comparison.json` 明确失败，没有放宽门槛。继续同时启用 `COSMOS_ATTENTION_DETERMINISTIC=1` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 后，C/D 的全部 12 个 raw、noise、完整 packed-input hash、loss 和 grad norm（0.12382988206477219）完全一致；完整状态比较通过：1226 个 tensor 的逻辑 archive payload 全部精确相等且有限，17 个对象状态也相等，报告为 `controlled12_deterministic_runtime_state_comparison.json`。该证据尚不能唯一归因于某一个 kernel，也不能替代原生与 TP/PP 的数值对照。attention 确定性默认关闭，保持基线默认行为。

`COSMOS_TP_PP_TRACE_PREPARED=1` 的额外 D2H hash 仅用于验收，默认关闭。`tools/compare_tp_pp_runtime_checkpoints.py` 比较全部模型、master、moments、step/scheduler、逻辑 loader/RNG 和模型并行 RNG；只排除单独保存的物理编译/probe RNG。相等的 tensor archive 走流式 SHA256 + finite 检查，不同 payload 走固定精度门槛的数值比较。回归测试确认即使模型相等，改变 moments 或出现相同 NaN/Inf 也不能通过。

全 36 层的真实 source FP32 对照脚本为 `tests/megatron/tp_pp_full_source_math.py`：裁剪真实 prepared 输入、恢复真实 masters/Adam、连续三次更新。预测平方和诊断 loss 下，原生 CPU/CUDA 参考均完成三步，但 CUDA TP2/PP2 比较仍有整参数梯度 relative L2 超过预定 `1e-4`（例如 VIDEO v_proj 约 `1.16e-4`）；TP1/PP2 隔离也有 q_norm 约 `1.22e-4`，尚不能归因于 TP。CPU 对照明确失败，见 `full_source_cpu_reference_observation.json`，未放宽门槛。该旧诊断还把干净首帧/State 条件改为 noisy target，不能代替原始生产目标。新增 `--objective native_rf --deterministic --native-device cuda` 保留真实条件 mask、位置、timestep 和 RF target，使用独立原生 VAE/noising 已验证并捕获的输入重新对照；原生 CUDA 参考已完成三步；TP2/PP2 的输出门槛通过，但首步全模型梯度 relative L2 为 `3.3869e-4`，未通过 `1e-4`。四 rank 完整逐参数分布保存在 `full_source_real_rf_fp32.update0.observation.rank*.json`，汇总为 `full_source_real_rf_observation.json`；TP1/PP2 隔离也未通过（全局 `2.4751e-4`）。随后查明该参考把非持久化 RoPE 频率初始化在 CPU，而原生生产模型与 TP/PP 在 CUDA 初始化；实测 4 个频率值不同，最大绝对差 `9.31e-10`。这使原对照的运行时常量不完全相同，不能仅据此归因于 TP 算法。现已修正参考初始化设备，增加进入梯度比较前的非持久化频率 SHA 精确校验，并在新的不可覆盖目录重建参考；新参考已完成三步，运行时常量 SHA 精确校验通过；TP2/PP2 首步梯度全局 relative L2 降为 `1.01723e-4`，仍超 `1e-4`，且有逐参数分片超界，不能判为通过。完整结果为 `full_source_real_rf_cuda_rope_observation.json` 及对应四 rank observation。未通过完整梯度/状态门槛前不启动 576 槽正式验收。原生 CUDA 参考的 masters/Adam 和 autograd saved tensors 放在 CPU，避免整模型 FP32 optimizer 在一张卡上 OOM。

验收可通过 `COSMOS_TP_PP_CAPTURE_PREPARED_DIR` 保存不可覆盖的逐槽位 prepared 输入，用 `COSMOS_TP_PP_CAPTURE_ITERATION` 限定更新，供原生/FSDP 使用完全相同的输入对照。该磁盘 spool 默认关闭，临时文件完成后才原子发布，不保留整个更新的主机内存副本。

导入结束后只对本作业 immutable canonical tensor 文件执行只读 page-cache advice；实测清理约 128 GB 缓存，避免导入缓存和完整保存一起逼近容器 900 GiB 内存上限。运行时恢复和比较完成后也清理已读 checkpoint 文件的 clean cache，不修改文件内容。

当前 venv 的 torchcodec 需要 FFmpeg 共享库。本机使用 `/opt/cosmos-megatron-ffmpeg8` 中指向现有 av wheel 的 FFmpeg8 库链接；已验证 Torch/torchcodec 及真实 DROID 解码。launcher 可通过 `COSMOS_FFMPEG_LIBRARY_PATH` 显式指定只含所需 FFmpeg 库的目录，避免引入容器中的冲突 PyTorch 库。

环境沿用仓库的 Megatron venv，固定 `megatron-core==0.20.0`。需要四张空闲 GPU：

```bash
bash tools/run_megatron_tp_pp_diagnostics.sh
```

脚本运行 CPU 契约/切片/完整状态转换测试，四卡 TP1/PP1/DP4、TP2/PP1/DP2、TP1/PP2/DP2、TP2/PP2/DP1，以及 TP2/PP2 的 compile + full AC。每次更新有 12 个逻辑槽位，连续 3 次更新，使用非零 moments 和 step，覆盖 padding、非均衡 packed 样本、单微批无 Action 和整个更新无 Action。只对有效 token 计算诊断 loss。

另外用小型可训练输入投影和输出 head 检查首/末 stage 的唯一所有权及跨 P2P 梯度连接；`--train-io --compile --checkpoint-activations` 已通过三次更新。这里的诊断适配器不等于生产 VAE/action 输入处理及 timestep/head 适配。

`tests/megatron/tp_pp_network_smoke.py` 进一步使用实际首末端适配器、vocab TP 和 metadata P2P，对照仓库原生 FusedAdam：12 槽位、4 次更新，覆盖部分微批无 Action、整个更新没有 noisy targets、随后重新出现梯度。eager 和逐 block compile + full AC 均通过，逐参数梯度、全局 norm、master、moments 和 step 均检查。输入为小容量合成 latent，未包含真实 VAE/数据读取。结果位于 `artifacts/megatron_tp_pp/network_io_metadata*.rank*.json`。

`--dynamic-layouts` 额外验证同次 1F1B 更新交替使用两个 UND 传输长度，在途微批各自携带 layout，启用 MCore 的三维动态 shape 协议；已在可训练首末端 + compile + full AC 下通过。生产 short UND 仍需结合真实 metadata 传输及原始容量单独验收。

逐参数和全模型梯度门槛为 relative L2 ≤ `1e-4`，全局 cosine ≥ `0.999999`；近零参数用绝对误差 `2e-6`，理论零梯度另检查精确为零。另比较全模型裁剪范数、更新后的 master/moments、Adam step、冻结权重和 scheduler/LR。保存进程全部退出后，由新进程恢复并继续两个更新。结果和逐 rank 明细写入独立 run 目录；`summary.json` 明确区分诊断通过与未完成的生产门槛。

实际源 checkpoint 的第 0 层还完成了独立的原宽度 FP32 对照：源 step=1200，保留真实 master/moments，TP2/DP2 和 TP4/DP1 均通过 3 次更新，使用合成的多样本 packed 输入。脚本为 `tests/megatron/tp_pp_source_layer_smoke.py --tp 2|4`，最新结果为 `source_layer_tp2_recheck_fp32.rank*.json` / `source_layer_tp4_fp32.rank*.json`；TP4 最大逐参数梯度 relative L2 为 `2.61e-6`。这证明该层的数学与状态桥接，不能代表完整模型、真实样本或原始容量验收。

支持用 `COSMOS_TP_PP_PROFILE_SLOTS=16:20` 只捕获完整更新中一段槽位；不会改变每次更新的槽位数。Nsight 需指定 `--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=none`，避免捕获结束时终止训练。四卡三更新的小容量 profiling smoke 已通过，报告 `/opt/cosmos-tp-pp-profiles/profile_smoke_ok_20261118.nsys-rep` 含 CUDA kernels、TP all-reduce、PP send/recv 和逐层 NVTX；它不能代表生产性能。

`tools/build_tp_pp_tokenizer_cache.py` 可由四卡 torchrun 并行构建原生 VAE AOT cache，再由训练 owner 通过原有 `COSMOS_TOKENIZER_AOT_CACHE_MODE=consume` 加载。实际 8 种生产空间尺寸的 32 个变体已经在四卡全部验证通过，编译约 146 秒；本机 artifact 为 `/opt/cosmos-tp-pp-tokenizer-aot`。缓存包含权重/配置/源码/runtime 身份及逐包 SHA，未改变 VAE 或编译选项。生产训练中的加载、噪声一致性及峰值容量仍需正式运行确认。

## 源状态盘点

`tools/audit_megatron_tp_pp_source.py` 已只读检查原始 `iter_000001200`：1214 个模型 tensor、811 个带 FP32 master/moments 的参数、192 个 rank RNG 及 192 份 loader 状态；消费过的 source/worker 均有 RNG。尚未消费的流保留原始 base seed 和初始游标，不伪造已消费 RNG。完整清单位于 `artifacts/megatron_tp_pp/source_inventory.json`。

第 0 层的 33 个模型参数及所有对应 master/moments 已逐项与原始 canonical tensor 比较，完全相等；报告为 `artifacts/megatron_tp_pp/source_layer_conversion_verification.json`。`tools/verify_cosmos_tp_pp_conversion.py` 会检查精确数值、超参、覆盖与逻辑状态副本，能够拒绝被修改的 moments。

全模型转换已完成，目标为本机独立目录 `/opt/cosmos-tp-pp-canonical-iter1200`。独立 verifier 已对全部 1214 个模型 tensor 及对应 master/moments/超参逐项重建比较，数值完全相等；trainer/scheduler/192 份 loader 副本也通过检查。报告为 `artifacts/megatron_tp_pp/full_conversion_verification.json`。canonical 导入格式与运行时 MCore checkpoint 是不同阶段，转换完成不代表生产续训通过。

`tests/megatron/tp_pp_full_import_smoke.py` 已将真实 36 层 Nano + width_only_1b Action 模型及 Adam 状态装入四卡 TP2/PP2。两个 stage 分别拥有 0–17 / 18–35 层，1210 个源计算 tensor 和全部 811 个参数的 master/moments/step 精确相等；其余 4 个归一化 buffer 按原生 append-only 契约生成 112 域的新配置，保留原 106 域 affine。FP32 导入峰值已分配显存约 37.5–37.8 GiB/卡，尚未包含训练激活/VAE，不能据此声称生产容量 fit。报告为 `artifacts/megatron_tp_pp/full_runtime_import_fp32.rank*.json`。

后续补充了实际源 scheduler 导入：各 rank 恢复 `last_epoch=1200`，LR 保留约 `1e-4` / `5e-4`。报告为 `artifacts/megatron_tp_pp/full_runtime_import_scheduler_fp32.rank*.json`。完整 source trainer/RNG 与真实下一批仍需接入训练循环。

58-source TOML 本身未开启保留源归一化；现有 `examples/launch_pretrain_state_h1_58src_192gpu.sh` 通过 override 开启，并给 Bench2Dex State128 指定旧统计路径。新的生产配置也必须显式继承这两项；契约不一致时导入会拒绝，而不会静默采用当前磁盘统计。

最新四卡机制矩阵结果为 `artifacts/megatron_tp_pp/diagnostics_transport_v2_20261118/summary.json`：36 份 rank 报告全部通过，最大全局梯度 relative L2 为 `3.01e-7`，最小 cosine 为 `0.9999999999999568`，包括保存退出后由新进程恢复继续更新。原生准备/网络 helper 的回归为 98 项通过（19 项 manual 未执行）。这些均不替代 576 槽位和生产容量验收。

首轮组件测试合计 32 项通过，包含 source slice/转换、完整 buffer 契约扩展、输入输出图、loss 权重、metadata schema、逻辑 loader、CUDA 训练 RNG、scheduler 和 owner 准备边界。后续新增 schema、预扫描回放、VAE owner group、retention 和 full→short UND warmup 回归也已分别通过；命令已纳入诊断 shell。

## 剩余交付

1. 原始基线与 TP/PP 的同 prepared-batch 数值对照，解释 BF16 重复差异；真实 12 槽位更新及完整重启保存已通过运行性检查。
2. 验证 metadata、逻辑 loader/RNG 和原 micro-step 权重串联的完整更新，以及有界内存/在途 batch 生命周期。
3. 真实 192 逻辑流的样本/packing/增强/噪声逐槽位对齐及严格恢复；小规模对照不能替代这项。
4. 迁移后的完整模型多更新、scheduler/trainer/逻辑数据串联和退出重启对照。model/Adam 导入已通过，不等于完整续训通过。
5. 原始容量、576 槽位、BF16、VIDEO FA3 / UND+ACTION Flex、short UND 与完整生产 packing 验收。
6. 四卡整次 update 的时间/显存/主机内存/通信画像。192 卡吞吐收益仍为后续验收项。

在这些项目完成前，不声明“四卡功能与完整续训验收通过”，不把小容量 FP32/compile 诊断当作生产性能结果。

### 2026-11-18 五步性能诊断进展

单遍 TP2/PP2 已从 1200 连续完成五步：141.57 / 96.16 / 94.33 / 107.54 / 93.32 秒，实际样本数 155 / 97 / 100 / 85 / 124，均为全局 12 槽。预定后两步均值约 100.43 秒；其中 1204 带 Nsight 捕获，短窗口结果不可作为长期稳态吞吐。完整 1205 checkpoint 已保存。相同条件的原生 FSDP 也已连续完成五步。按所有 rank 的最大耗时计，前三步预热后的两步均值为 TP/PP 100.436 秒、原生 FSDP 46.839 秒；当前 TP/PP 吞吐为参考的 0.466 倍（耗时约 2.14 倍）。两边 1204 均带 Nsight 捕获，计时均排除 checkpoint 写入，不代表长期稳态或 576 槽性能。五步共 60 槽的 source、样本数、tokens、原始媒体 SHA、噪声 SHA 和完整 prepared packing SHA 全部逐槽一致。

TP/PP 捕获报告为 `/opt/cosmos-tp-pp-profiles/matched12_tp_singlepass_20261118.nsys-rep`，SQLite 和 CSV 位于同目录。`tools/analyze_tp_pp_nsys.py` 按 process + correlation ID 关联 CUDA launch，避免多 rank 合并文件中的 correlation ID 冲突。8 槽捕获中，owner 原生准备 CPU 窗口 44.11 秒，其中启动的 GPU kernels 并集 31.28 秒；owner 非 NCCL kernel 活跃 46.85 秒，其他三个 rank 各约 15.4–15.6 秒，另有约 41–48 秒只运行 NCCL kernel。NCCL kernel 时间包含等待，不能解释成等量线缆传输成本。CPU hash 约 4.37 秒，loader 等待约 1 毫秒，说明该窗口的 CPU 预取已供上数据；单 owner 准备和其他卡等待是后续优化重点。该捕获只覆盖更新的一部分，未将不同 rank 的秒数相加为整步耗时，非 kernel 窗口还可能包含 CUDA memops。原始分析为 `artifacts/megatron_tp_pp/matched12_tp_profile_attribution.json`。

五步 BF16 loss 对照发现旧 TP 日志别名问题：`GenerationLoss` 返回的 `value.detach()` 与 loss 共用存储，MCore `forward_step_calc_loss` 对 loss 原地除以 M，也将日志值除以 M。本次 M=12。现已改为 `detach().clone()`，并增加实际 MCore 调度函数的回归检查，证明日志保留未缩放 loss，反向仍只除一次 M。旧日志不改写；显式 `--legacy-tp-loss-scale 12` 的对照保存为 `artifacts/megatron_tp_pp/matched12_5updates_performance_loss_reporting_corrected_20261118.json`，五步 loss 相对差全部小于 1%。未经校正的失败报告也保留。这仅证明 BF16 loss 轨迹门槛，未证明完整梯度/master/moments 或严格续训验收。原生最终数值快照 `outputs/megatron_tp_pp_native_matched12_20261118/math_state_000001205/.math_complete` 已发布，它不是可恢复训练 checkpoint。

原生同 8 槽 profile 中，每卡处理两槽，VAE kernel 并集分别 7.773 / 7.778 / 7.748 / 7.749 秒；TP 单 owner 处理八槽的对应值为 31.247 秒。原生各卡非 NCCL kernel 活跃约 23.5–24.0 秒，NCCL-only 约 0.32–0.79 秒。与 TP 的单 owner VAE 集中和其余 rank 等待形成一致证据；不能把所有 NCCL kernel 持续时间解释成传输时间。原始报告为 `artifacts/megatron_tp_pp/matched12_native_profile_attribution.json` 与 `matched12_tp_profile_attribution_v2.json`。后续优化应首先恢复冻结准备的并行度，并单独验证 loader/RNG 和 prepared 张量一致。

普通 rank-level 单遍训练现在跳过原始媒体全量 SHA；`COSMOS_TP_PP_TRACE_PREPARED=1`、prepared capture 或预扫描/显式回放验证仍强制保留 SHA。普通 manifest 的 `sha256` 为 null，不可作为输入一致的证明；性能比较工具会拒绝两边都为 null 的“匹配”。同样调整应用于原生参考，保持开关对齐。实际 profile 中此前 SHA 开销约 6 秒/12 槽；去除后的整步收益尚未重测，不能从局部耗时直接宣称加速。11 项 planning/loss/performance 测试通过，包含两次更新、worker=0/2 的预扫描/单遍有哈希/单遍无哈希数据元信息与最终 loader/RNG 一致，以及真实 MCore 原地 loss 缩放回归。

完整 36 层 FP32 数学隔离继续使用原始 master/moments 和真实 RF。新增 `--retain-full-text` 回放保留两条真实样本的全部文本条件；仍裁剪视频/动作 token，不能替代生产容量验收。此前精确 CUDA RoPE 的短文本用例全局梯度相对 L2 为 1.01723e-4，超过 1e-4，仍判失败；不通过调整容差覆盖它。完整文本 native 三步参考与 TP 对照独立记录，不覆盖历史失败。

### 并行冻结准备实验（默认关闭）

`COSMOS_TP_PP_PARALLEL_PREPARATION=1` 将每个模型并行组负责的逻辑流，按组内成员顺序分配到所有 TP/PP ranks。每个逻辑流仍使用自己的原生 loader/增强/训练 RNG；VAE、噪声、RF 和 packing 沿用原 helper。各卡先完成各自槽位的冻结准备，逐批以显式 tensor schema 写入共享作业目录的临时 prepared spool，随后首 stage owner 按原 `(update, micro_step, logical_rank)` 顺序读取，执行原 MCore 1F1B。可训练输入模块仍在首 stage，未 detach 或复制其训练。该实现保留一批 CPU staging，不在内存中保留整步原始视频；576 槽时磁盘空间随 prepared 数据量增长，暂存容量和 I/O 开销需要实测。

归一化 spec 从原首 stage owner 显式广播，各准备 rank 只额外构造冻结 tokenizer 和归一化 anchor。保存时在 CPU/Gloo 上收集各准备 rank 的逻辑 loader/RNG，检查无重复、无遗漏，然后合并回旧首 stage owner 的完整 checkpoint 布局；新进程可重新分配这些逻辑流。该开关不改变 TP/PP 参数布局或 topology 字段。每槽数据采用显式 tensor schema，Python 对象通信只在更新/保存/恢复边界发生。正常更新完成后删除其临时 spool；失败作业可能留下临时文件，重新启动使用新的独立目录，不重用未提交的准备结果。

当前 16 项准备、planning、loss、performance 单测通过，其中包括 DP1/2/12 下逻辑流覆盖、旧 owner 状态重建、重复/缺失状态拒绝和 prepared spool 原样回放。真实四卡五步实验已完成，使用相同 source1200、12 槽、BF16/compile/full AC、关闭 VAE AOT 和开启原始/prepared 哈希的设置，逐槽与已完成原生五步结果核对。首次性能比较预先指定未捕获 profiler 的 1205，只有单个测量点，不能视为长期稳态。该路径的五步 GPU 诊断结果见下文；恢复、AOT 和 576 槽验收仍未完成，不将其设为默认。

并行准备接入后，完整 TP/PP CPU 组件套件 `tests/megatron/tp_pp_*_test.py` 已执行：65 passed（29.39 秒）；这不替代该新路径的生产 GPU/恢复验收。旧完整文本原生 CUDA FP32 参考（裁剪 norm 在 CPU 计算）完成三步，norm 为 123.034729 / 20.867571 / 11.728599；其 TP 对照与后续独立 norm 审计见下文。

完整文本 FP32 第一更新的四份梯度/输出 observation 均通过：全局梯度相对 L2=1.00413e-5、余弦=0.9999999999902，所有逐参数梯度及输出通过。但裁剪 norm gate 失败：原生参考 123.034729，MCore 123.170723，相对差约 0.11%。独立扫描保存的全部 811 个原生梯度、以 FP64 累计得到 123.169610；重复 CPU FP32 各参数 norm 得到 123.034731，验证误差来自卸载到 CPU 后的 FP32 norm 累计（部分 50M 元素矩阵低估约 0.7%）。这不是依据 loss 推测的误差。现将参考改为逐参数在 CUDA 上测 norm，再裁剪 CPU 梯度和更新 CPU masters；不更改固定门槛。旧参考及失败报告保留，新 CUDA-clip 三步参考独立输出，当前未验收完整 optimizer 状态。证据为 `full_source_real_rf_full_text_reference_norm_audit_20261118.json` 与 `full_source_real_rf_full_text_observation.json`。

真实四 CPU 进程、非连续模型组 `[0,2]` / `[1,3]` 的 Gloo 状态 gather/恢复分配检查四 rank 全部通过，报告 `parallel_preparation_gloo_state_smoke_20261118.rank*.json`；仍不能替代真实 loader/GPU checkpoint 续训。并行准备真实四卡五步结果及完整保存状态见下文。

并行准备五个更新已全部完成，按最慢 rank 耗时为 115.718 / 50.964 / 52.676 / 51.444 / 49.121 秒。60 槽 raw/noise/packed SHA 全部与原生相同，五步 BF16 loss 最大相对差约 0.0882%。预先指定的无 profiler 更新 1205 为 49.121 秒，原生同更新为 39.835 秒（新 TP/PP 约 1.233 倍耗时）；此前单 owner TP 同更新为 93.327 秒。只作单点诊断，不能宣称长期稳态或 576 槽提速。训练峰值 allocated 为约 70.582 GiB（所有 rank/更新取最大，含准备阶段）；该实验无 Nsight，1205 与参考均无捕获。最终 1205 完整 checkpoint 已保存；此前预保存观察报告为 `matched12_parallel_preparation_pre_checkpoint_observation_20261118.json`。

并行准备五步实验已正常退出（returncode=0），完整 `iter_000001205/.save_complete` 已发布；最终比较报告 `matched12_parallel_preparation_5updates_20261118.json` 确认 60 槽一致和五步 loss 门槛通过。checkpoint 与旧单 owner 同迭代的完整模型/optimizer/逻辑状态比较已启动；该项尚未出结果，不能仅凭 marker 认为保存内容或恢复正确。CUDA-clip FP32 三步参考已在独立目录启动，旧 CPU-clip 参考不改写。

### 真实 checkpoint 的严格恢复命令

`tools/run_tp_pp_restart_acceptance.py` 为四卡 TP2/PP2 提供固定协议：从同一已完成 checkpoint 分叉，A 连续更新两步，B 更新一步并保存、全部进程退出，再以新进程恢复更新第二步。初始 checkpoint 仅通过只读加载的符号链接引用，不复制或覆盖原状态。两分支逐槽比较 raw/noise/packed SHA、检查有限 loss，并比较最终完整模型、master/moments/step、scheduler、逻辑 loader/RNG 和模型并行 RNG。物理 compiler/probe RNG 按原 checkpoint comparer 的既有范围单独排除，结果报告显式记录。每次启动前及全部子进程结束后核对训练源码/TOML 哈希，变更会使本次验收失败，避免不同源码分支被误认为同一次对照。

当前排队命令（等待 CUDA-clip FP32 作业结束再占用四卡）：

```bash
PYTHONPATH=. LD_LIBRARY_PATH='' .venv/bin/python tools/run_tp_pp_restart_acceptance.py \
  --initial-checkpoint outputs/megatron_tp_pp/cosmos3_action/action128/tp_pp_matched12_parallel_prepare_5updates_20261118/checkpoints/iter_000001205 \
  --bundle /opt/cosmos-tp-pp-canonical-iter1200 \
  --output-root outputs/megatron_tp_pp_strict_restart_parallel12_20261118 \
  --parallel-preparation --devices 0,1,2,3
```

默认关闭 VAE AOT，与该 12 槽初始 checkpoint 的实验开关一致；`--compile-tokenizer` 可启用原 callback，须另行验收。该工具的完整 12 槽真实进程恢复结果尚未取得，排队命令不算通过证据；576 槽同样仍是独立未完成项。

并行准备与单 owner 的完整 1205 checkpoint 比较已结束：**1226 项模型/master/moments 张量 payload 全部逐字节相同且有限**，scheduler、逻辑 loader/RNG 和模型并行 RNG 亦无差异。总 gate 仍为 false：仅三个非 owner rank 的 `source_trainer` 从旧空字典变成重复的全局来源信息，报告精确定位为这三个 `.source_trainer.keys`。现已修正保存布局：仍仅原数据 owner 保存该公共信息，后续严格恢复实验将使用修正后的布局；旧 checkpoint 和失败报告不改写。完整报告 `parallel_prepare_vs_single_owner_checkpoint_1205_20261118.json`，摘要 `parallel_prepare_vs_single_owner_checkpoint_1205_summary_20261118.json`。这提供五步并行准备不改变参数/optimizer/逻辑状态的证据，仍不能代替退出重启或 576 槽验收。

修正后的 CUDA-clip 原生参考三步已全部完成并正常退出；norm 为 123.169609 / 20.899004 / 11.720305，manifest 明确 `clipping_norm_device=cuda`。四卡 TP2/PP2 三步对照已启动，不能将原生参考完成当作 TP 状态验收通过。

### 跨步 FP32 参考及下一组状态验收

CUDA-clip、CPU AdamW 参考的第一步梯度、norm 和全部 optimizer 状态通过；第二步梯度全局相对 L2 为 1.10507e-4，超过 1e-4，仍判失败，未通过三步验收。完整观察保存在 `full_source_real_rf_full_text_cuda_clip_summary_20261118.json` 和逐 rank 文件。随后核对到另一参考差异：生产原生 optimizer 为 `cosmos_framework.utils.generator.fused_adam.FusedAdam`，MCore 实际为 TE FusedAdam；旧数学参考使用 CPU AdamW。新增 `--native-optimizer native_cuda` 逐参数搬运 CPU master/moments，通过原生 CUDA FusedAdam 更新，CPU AdamW 仅作为状态及 zero_grad 容器，不执行其 CPU step。

真实 GPU 小测从非零 moments、step1200 连续更新三步，逐参数搬运更新与原生批量 FusedAdam 的 master/moments/step 逐字节相同；原生 capturable FusedAdam 与 MCore 非 capturable TE FusedAdam 的同一三步小测也逐字节相同。证据 `native_cuda_streamed_adam_qualification_20261118.json`、`native_vs_mcore_adam_kernel_20261118.json`。这只验证搬运及实际 optimizer kernel 的小测，不替代完整 36 层三步结果；使用新原生 CUDA optimizer 的完整参考及 TP 对照正在执行，固定门槛不变。

第一组严格恢复工具运行因启动期间 agent 增补只读导出回调，训练源码哈希变化而失去对照资格；已终止该组自身进程，保留 `outputs/megatron_tp_pp_strict_restart_parallel12_20261118/invalidated_source_guard.json` 和全部现场，不改写为成功。新的固定源码运行排在原生 CUDA optimizer 数学作业之后，输出独立目录 `outputs/megatron_tp_pp_strict_restart_parallel12_fixed_source_20261118`；源码 guard 继续生效。

新增 `tools/export_tp_pp_runtime_numerical_state.py`：使用原生产 `train.run` 恢复完整 checkpoint，在初始化回调中导出逐参数实际 model/FP32 master/moments/step/hyperparameters，且 `max_iter` 等于恢复 iteration，不执行训练更新。作业配置和链接位于独立目录，源 checkpoint 只读。导出逐参数分批转 CPU，检查全部 canonical 参数及 TP/DP 片段覆盖，写入并 fsync 后发布 `.numerical_complete`；它不是可恢复 checkpoint。`tools/compare_tp_pp_native_numerical_state.py` 与已完成的原生 FSDP 数值快照比较，保留逐参数固定 1e-4/2e-6 观察，检查实际五步输入 lineage 和 loss，另检查全部冻结模型；不宣称已比较 production 梯度。真实四卡导出与原生数值比较排在严格恢复之后，结果尚未取得。

当前 CPU 组件套件为 68 passed（31.06 秒）；新增片段覆盖检查覆盖 TP/DP 双维完整性、缺片/重叠拒绝，以及 torch payload tuple 与 JSON list 的身份校验。Ruff 检查及 68 个相关文件格式检查通过。这些检查不替代上述 GPU/恢复/576 槽验收。

### 严格恢复的实际轨迹差异（待修正）

固定源码的严格恢复实验中，A/B 的 1206、1207 共 24 槽 raw/noise/packed SHA 全部相同；1206 loss/norm 相同。1207 loss 仅有小差异，但梯度 norm 为 A=0.1536827602、B=0.1377537110，不能判恢复正确。metadata bytes 相差 6,868,998；`(11776-5120) * (128*4*2+8) = 6,868,992` 恰为额外 UND RoPE/ID 张量大小（其余少量 descriptor 字节尚未逐项核对）。代码 `build_stage` 每个新进程将 `_warm_full_und` 置 True，首个 forward 将它清为 False，而 checkpoint 未记录该优化状态。当前将它列为首要原因，仍需保存/恢复该状态后的干预实验确认；不将相关性直接当作已修复证据。完整状态比较还在执行，保持训练源码冻结，不覆盖本次有效失败现场。观察记录 `strict_restart_1207_und_bucket_discrepancy_20261118.json`。

原生 CUDA optimizer 参考下，第二步全局梯度相对 L2 为 8.42267e-5，已通过全局阈值，但 q/k norm 等局部参数仍超界，三步 gate 未通过。新增 `--observe-all-updates` 仅用于继续收集全部有限更新和 compute/master 一致性；所有固定 gate 保持，任何失败都会在最终报告保留 `passed=false` 并非零退出。该观察作业排在实际数值导出之后，不作为通过的捷径。

### 2026-11-18 严格恢复及三步观察的后续结论

固定源码严格恢复的完整状态比较已结束：1226 项张量中 590 项未通过，17 项对象状态均通过，不能验收恢复正确性。随后从相同 1206 checkpoint 做单独干预，仅将 `_warm_full_und` 设为 False，12 槽输入仍完全一致，metadata bytes 恢复为连续运行的 2,220,676,440，但梯度 norm 为 0.1375958916，仍偏离连续运行的 0.1536827602。因此 UND 预热状态解释了 metadata 差异，尚不能解释主要梯度偏差；上一节的首要原因假设未获干预支持，不据此提交恢复修复。证据为严格恢复目录的 `state_comparison.json` 和 `artifacts/megatron_tp_pp/und_resume_bucket_intervention_20261118.json`。

原生 CUDA FusedAdam 参考下的完整三次 FP32 更新观察已结束，四份最终报告均为 false，进程按失败 gate 非零退出。三步全局梯度相对 L2 依次为 1.00413e-5、8.42267e-5、3.61974e-4；第三步全局阈值也失败，第二、三步局部梯度及部分 Adam moments 存在超界。所有 4866 项逐步 compute/master 检查均完全一致，未发现此 FP32 路径的模型参数同步丢失。摘要 `artifacts/megatron_tp_pp/full_source_native_adam_three_update_observation_summary_20261118.json`，逐 rank 明细 `full_source_real_rf_full_text_native_adam_observe_fp32.rank*.json`。该结果没有放宽数值门槛，亦不能代替完整生产 BF16 参数与 master 恢复检查。

对实际生产 1205 checkpoint 恢复后的 BF16 数值导出新增 `tools/audit_tp_pp_model_master.py` 审计：1622 个片段中有 31 项 compute 不等于本 rank FP32 master 的 BF16 转换，全部为 TP rank1 的复制参数。配对检查这 31 项，恢复后的 TP 两份 compute 均完全相等，但每项的两份 FP32 master 都不同；TP rank0 的 compute 全部等于自己的 master 转换，TP rank1 全部不等。模型 checkpoint 去重复制参数，而 optimizer 保留各 TP rank master，恢复代码原先分别加载两者、没有从各自 master 重建 compute。这是已测出的恢复不一致，尚需干预证明其对严格重启轨迹的影响；保留 `restored_1205_compute_master_audit_20261118.json` 和 `restored_1205_replicated_master_diagnosis_20261118.json`，不改写旧数值导出。

单步定位工具新增 `--disable-compile` 和 DP1 专用 `--restore-compute-from-master`，前者保留 full AC，后者仅在恢复回调中从实际本地 master 重建 compute；均关闭 checkpoint 写入并使用独立输出目录。它们用于隔离因素，不能作为严格恢复通过证据。最终仍需正式恢复路径的连续/退出重启对照。

独立核对实际导出与源 runtime checkpoint：全部 1622 个 compute 片段逐字节匹配源模型；同样 31 项仍不等于本 rank master 转换，排除了读取错误模型张量的解释。证据 `restored_1205_compute_source_master_audit_20261118.json`。关闭 compile 的原容量单步定位实验因 NCCL CUDA calloc 失败退出，没有获得梯度结论；不将其计为数值失败或通过，也不据此推断 GA3 的显存需求。

实际生产五步 BF16 数值对照也已完成：60 槽输入一致、loss 轨迹通过 1% 门槛、1622 项 step/超参数比较和 403 项冻结模型比较全部通过。6488 项 active model/master/moments 观察中，按预设 1e-4/2e-6 检查有 5281 项未通过，整体报告为 false；该观察没有生产梯度对照，不以 loss 接近替代状态正确性。报告 `native_vs_tp_pp_actual_5updates_numerical_state_20261118.json`。BF16 误差分布保留，FP32 三步数学门槛仍须单独解决。

严格恢复工具的后续默认窗口已扩大为连续三次更新，对照另一分支第一次更新后退出、启动新进程再连续两次更新，满足计划要求的至少两次重启后更新；可通过 `--updates-after-restart` 增加窗口，不允许降到两次以下。旧两更新实验仅提供失败定位证据，不改写其范围。

### 从各 rank master 重建 compute 的恢复修复

单步干预完成：在已经关闭 UND 重复预热的对照上，仅增加本 rank master→compute 重建，12 槽 raw/noise/packed 输入全部相同，1207 loss 和梯度 norm 与不中断分支完全相同；norm 恢复为 0.15368276020317817。两边所有 rank metadata bytes 总量也相同（4,441,352,880）。证据 `artifacts/megatron_tp_pp/master_resume_intervention_20261118.json`。该结果支持上述加载不一致是实际轨迹偏差原因，但干预工具关闭保存，不能替代最终完整 checkpoint 比较。

正式 `checkpoint.load` 现使用已恢复 optimizer 的 `_copy_main_params_to_model_params`，再由 DDP `start_param_sync(force_sync=True)` 完成 DP 片段 gather 与 compute 更新；不从 BF16 模型反向覆盖真实 FP32 masters。普通 FP32 optimizer 没有该复制方法时保留原模型加载结果。训练 rank state 同时保存/恢复 `input_adapter_warm_full_und`；旧 runtime checkpoint 仅在成功更新后保存，缺失该字段按已经完成预热处理。

新增真实四卡 BF16 回归 `tests/megatron/tp_pp_master_restore_smoke.py`：有意使 TP 复制参数的非零 masters 跨越 BF16 舍入边界，保存后破坏 compute/master/moments，再恢复并逐片段比较原 compute、FP32 master、两项 moments、step 和 rank state。旧代码在 TP rank1 复制 norm 上复现失败；修复后 TP2/PP2/DP1 和 TP2/PP1/DP2 均四 rank 全部通过。报告 `master_restore_after_fix_{pp2,dp2}_20261118.rank*.json`。完整真实三更新严格恢复仍须单独运行，FP32 三步数学误差和完整 576 槽验收尚未解决。

修复后的 CPU 组件检查为 68 passed（29.88 秒），相关六份 Python 文件 Ruff 检查和格式检查通过。新的真实三更新严格恢复已实际启动四个训练 rank，输出 `outputs/megatron_tp_pp_strict_restart_master_restore_three_updates_20261118`：A 连续 1206–1208，B 1206 保存退出后新进程执行 1207–1208。该组启用源码 hash guard，运行期间保持训练和比较源码冻结；当前启动记录不代表最终验收通过。

后续实际进展：不中断 A 分支已完成三次更新并发布完整 1208 checkpoint，B 分支已启动；中途源码 hash 核对没有变化，最终比较仍待完成。完整 36 层、真实 RF 回放、非零 Adam 状态的 TP1/PP2/DP2 三步 FP32 对照已排队，在严格恢复的最终训练 rank 退出后执行，固定数值门槛和全部 observation 保持，用于隔离 TP 矩阵切分与 PP/DP/参考路径的影响；它不能替代必须通过的 TP2/PP2。

只读复核 VAE AOT cache 的身份 fingerprint、当前 tokenizer 源码 SHA、READY 标记及全部 32 个 package 大小/SHA，均通过，报告 `artifacts/megatron_tp_pp/tokenizer_aot_pre_training_integrity_20261118.json`；`actual_training_consumed=false`。已排队在 TP1 完整三步观察结束后，使用 `consume` 模式、正式 callback、并行准备和实际模型执行独立的三更新严格恢复，输出预定为 `outputs/megatron_tp_pp_strict_restart_aot_consume_three_updates_20261118`。当前只是排队，未宣称完整模型 AOT 已通过。完整 576 槽作业尚未启动。

新增 `tools/audit_tp_pp_tokenizer_coverage.py` 只读核对实际 58 个 catalog：全部使用 `unified_action128` adapter。当前 `view_layout` 的三种宽高比族、每族 1–3 个视角，共产生 8 个不同像素画布，全部在 TOML 的 `spatial_sizes` 中。按 cache 声明的 chunk/exact-duration 配置推导的 32 个时序/空间变体也全部存在，源码 SHA、cache 身份与 recipe 画布一致，无缺失项。报告 `artifacts/megatron_tp_pp/tokenizer_aot_58_source_structural_coverage_20261118.json`。这补齐了 58-source 的结构覆盖证据，不解码媒体、不消费采样状态，不证明实际完整模型的显存或数值结果；AOT 实际训练验收继续保留。

### Nsight 归因补充：拷贝与无 CUDA 活动时间

`tools/analyze_tp_pp_nsys.py` 现以只读模式打开 SQLite，支持 `--phase-prefix` 细分 NVTX 阶段，继续按进程和 correlation ID 联合匹配 kernel；按线程排序 launch 并二分定位区间，避免每个阶段扫描全部 kernel。新增 memcpy/memset 与 kernel 的时间并集，分别报告仅内存操作、无 CUDA 活动的时间，阶段内区分非 NCCL 与仅 NCCL kernel。嵌套阶段互相重叠，不能相加当成 update wall time，NCCL kernel 仍包含等待。

对已有单 owner TP 的部分窗口，owner 原先 13.960 秒“无 kernel”中有 2.764 秒仅 memcpy/memset，实际无 CUDA 活动为 11.196 秒；三个 peer 对应为 0.225–0.261 秒，其大量 NCCL 时间仍主要包含等待。原生对应部分窗口每 rank 有 0.699–0.756 秒仅内存操作、2.658–2.708 秒无 CUDA 活动。完整细分为 `matched12_tp_detailed_attribution_20261118.json`（87 个 phase/process 行）和 `matched12_native_detailed_attribution_20261118.json`（304 行）。这些是旧单 owner profile，不将它们冒充并行准备后 49 秒路径的新 profile，也不将部分窗口差异直接换算成整步收益。新增合成 SQLite 回归使用两个进程重复 correlation ID 和交叠 memcpy/memset，验证不会串进程归因或重复累计内存时间，1 passed（2.77 秒）。

三更新严格恢复的首个实际重启后更新 1207 已完成：A/B 共 24 槽（1206、1207）raw/noise/packed 身份全部一致，四 rank 梯度 norm 完全一致，报告 loss 的末 stage 两 rank 也完全一致；1207 norm 为 0.11519503325968233。中间观察 `strict_restart_master_restore_prefix_1207_20261118.json` 明确 `strict_restart_accepted=false`，仍等待第二个重启后更新 1208 及其最终完整状态比较。

后续 1208 更新也已完成并核对：三更新共 36 槽 raw/noise/packed 全部相同，四 rank 三步 grad norm 和实际报告的 loss 均与不中断分支完全一致。最终 1208 checkpoint 正在保存，完整状态比较尚未完成，不提前将严格恢复标为通过。

最终 checkpoint 已发布、所有实际训练 rank 已退出，完整状态比较正在 CPU 执行；排队的 TP1/PP2/DP2 完整源模型隔离已实际启动。第一步四 rank 的全局梯度相对 L2 为 1.68227e-6、余弦 0.999999999999403，逐参数梯度及输出均通过，norm 为 123.1694518425；该结果只是第一步观察，后两步和 optimizer 状态尚待最终报告。

完整源模型数学 helper 新增 `--replay-native-state` 诊断模式，在第二、三步前恢复原生参考上一更新的 master/moments/step，用来区分固定参数状态下的梯度误差与跨步误差放大。该模式尚未执行 GPU 验证；最终始终 `continuous_trajectory=false`、`passed=false` 并非零退出，即使逐点 gate 通过也不能替代三步连续更新验收。默认连续路径不启用该选项；目前运行的 TP1 作业在此新增模式前已加载原 helper，不受其影响。

### 真实严格重启通过与 Adam 算术隔离

`outputs/megatron_tp_pp_strict_restart_master_restore_three_updates_20261118/summary.json` 已完成并 `passed=true`：A 连续 1206–1208，B 1206 保存退出后新进程连续 1207–1208，36 槽 raw/noise/packed 全部相同，loss 最大相对差为 0。`state_comparison.json` 的全部 1226 项张量均为 `exact_payload_equal=true`（流式检查同时确认有限），17 项对象均通过，含 optimizer step/超参数、scheduler、逻辑 loader/RNG 和模型并行 RNG；比较器按既定规则排除物理/compiler RNG。该验收为 12 槽、关闭 tokenizer compile、开启并行准备，不代表 576 槽或 AOT 验收。

TP1/PP2/DP2 三步隔离已结束，四份最终报告均失败：全局梯度相对 L2 为 1.68227e-6、7.86606e-5、9.27541e-4。第三步同样超界，说明不能仅将问题归因于 TP 矩阵切分；原失败报告保留为 `full_source_real_rf_full_text_native_adam_tp1_pp2_fp32.rank*.json`。

新增 `tools/qualify_tp_pp_adam_real_values.py` 从实际 source master/moments/step 和三步原生梯度值分布采样，覆盖全部 811 个参数、189786 个元素，使用相同梯度分别更新原生 streamed、原生 batched、TE 非 capturable 和 TE capturable。原生 streamed/batched 三步逐位相同；TE 非 capturable 的 moments 相同，但 master 分别有 392/530/577 项采样张量不同，最大绝对差为 5.96e-8/5.96e-8/1.19e-7。TE capturable 三步 master/moments 全部与原生 streamed 精确相同。报告 `native_vs_mcore_adam_actual_value_samples_20261118.json`、`native_vs_mcore_adam_capturable_actual_values_20261118.json`；它们保留非 capturable 负对照，整体 `passed=false`，应读取各 comparison，不能将全模型状态标为通过。该真实值检查推翻了此前随机小样本“两个 kernel 处处逐位相同”的泛化，尚未证明它解释完整模型的全部跨步偏差。

AOT consume 的不中断分支已实际完成三个原容量更新并保存最终 checkpoint，cache 初始加载/验证约 28 秒，后续两个更新约 45.95/45.81 秒。这是短诊断窗口，且同一作业期间有上述小型 kernel 资格检查，不作为稳态性能验收。AOT 严格恢复的当前子分支继续运行至保存退出，其父协调器暂缓启动下一分支，以在 GPU 释放后插入完整模型 `--native-adam-arithmetic` 三步候选验证；包装脚本通过 `finally` 恢复父协调器。候选仅在数学 helper 的进程中切换 TE capturable kernel、规范 lr/step tensor，不开启图捕获，不修改当前受 hash guard 保护的生产默认。候选完整三步结果尚未取得，固定 gate 不变。

### 真实 DP2 组合的 bundle 准备

新增 `tools/repartition_tp_pp_bundle.py`，在 TP 不变时重新分配 canonical 参数的 PP owner，并硬链接不可变 payload 与全部逻辑状态，不重写 FP32 masters/moments/step，也不消费 RNG。目标必须是独立且不存在的目录；发布前核对源 manifest 和 payload stat 未改变，拒绝越界 payload 和逻辑状态符号链接。源和目标均须保持不可变，训练输出仍写入独立 job。

实际 TP2/PP1 bundle 已建立于 `/opt/cosmos-tp-pp-canonical-iter1200-tp2-pp1`，覆盖 1214 个参数、2550 个硬链接文件（含全部原逻辑状态），报告 `full_bundle_tp2_pp1_repartition_20261118.json`。独立 `verify_cosmos_tp_pp_conversion.py` 正在逐项对照原始 native checkpoint，不将硬链接创建成功视为独立验证通过。后续用于四卡 TP2/PP1/DP2 的真实状态更新与重启检查，尚未启动该训练。

TE capturable 算术候选的完整源模型三步验证已实际启动；第一更新的四 rank 梯度/输出 gate 通过，全局梯度相对 L2 为 1.00413e-5，与旧路径第一步相同（optimizer 尚未影响这一步梯度）。后续梯度与状态结果待完成；AOT 父协调器继续等待，候选包装脚本负责自动恢复它。

候选三步观察现已全部完成，四份最终报告均失败：全局梯度相对 L2 为 1.00413e-5、9.02485e-5、6.27623e-4，第二步存在局部梯度/状态超界，第三步全局阈值也失败；所有 compute/master 检查仍完全一致。摘要 `full_source_capturable_adam_candidate_summary_20261118.json`。因此真实值 Adam kernel 算术差异虽已测出，对齐 capturable kernel 尚不能解释或消除完整回放的全部跨步偏差；生产默认没有切换。

候选包装脚本已正常执行 `finally`，确认 AOT 父协调器恢复运行，并实际启动重启后的 1207–1208 分支。下一项 `--replay-native-state --native-adam-arithmetic` 逐点诊断已排队，等待 AOT 最终 checkpoint 发布且训练 rank 全部退出后执行；不再暂缓 AOT 父协调器。该诊断不作为连续三步通过证据。


### 最新验收结果与参考实现核验（2026-11-18）

VAE AOT consume 的三更新严格重启已结束并通过，见 `outputs/megatron_tp_pp_strict_restart_aot_consume_three_updates_20261118/summary.json`：A 连续 1206–1208，B 保存退出后恢复两步，共 36 槽 raw/noise/packed 全部一致，loss 最大相对差为 0；最终 1226 个张量 payload 全部精确相等且有限，17 项对象状态全部通过。源码 hash guard 全程保持，实际训练消费正式 cache，禁止 fallback。本项覆盖原容量 12 槽更新，不代表 576 槽或稳态吞吐验收；上文“排队/运行中”的记录为当时的过程状态。

TP2/PP1 canonical bundle 的独立原始源核验也已完成，`artifacts/megatron_tp_pp/full_conversion_tp2_pp1_verification_20261118.json` 为 `passed=true`，覆盖完整模型与源优化器/逻辑状态。实际 DP2 完整模型训练仍未验收。

逐点恢复原生上一更新状态的三步诊断已完成：全局梯度相对 L2 为 1.00413e-5、6.22677e-5、8.54221e-6，但第二更新 rank0/1 分别有 70/82 项局部梯度超界。因此 `pointwise_gates_passed=false`，且该模式始终不是连续轨迹验收。报告 `full_source_real_rf_full_text_native_state_replay_fp32.rank*.json`，失败证据保留。

新增 `tests/megatron/tp_pp_gpu_leaf_reference.py` 独立参考辅助：原生可训练 FP32 Parameter 保留在 GPU，冻结层流式搬入，保存的反向 tensor 卸载至 CPU，Adam moments 在 CPU 持有并逐参数调用原生 CUDA FusedAdam。混合冻结/可训练小模型、非零 step1200、每更新两次梯度累积的三更新核验中，输出、梯度、norm、master/moments/step 均与全 GPU 原生实现逐位一致，见 `gpu_leaf_reference_offload_qualification_20261118.json`。完整 36 层参考已实际启动；它用于核验此前 CPU 叶子参数参考的保真度，尚未取得完整模型通过结论，固定门槛不变。

严格重启 runner 现支持直接从 canonical bundle 开始，并根据 bundle 明确覆盖 TP/PP；canonical 起点必须显式提供逻辑 DP 和每更新全局槽数。已有 runtime checkpoint 起点严格拒绝改变拓扑或更新数据量。这使 TP2/PP1/DP2 与原始 576 槽验收可以复用相同的 A/B 保存退出协议，无需伪造初始 runtime checkpoint；功能支持不代表这些训练已通过。新增 topology 契约检查 9 passed（4.38 秒），相关脚本与测试 Ruff lint/format 均通过。


### 原始 192 流 / 576 槽的数据验收

新增 `tools/build_tp_pp_raw_replay_plan.py`，直接复用 canonical 源状态导入、原生 52→58-source extend 校验、逻辑 loader、原始 packing 和 batch hash；不构建 decoder 或 VAE。生成第一更新的完整 576 槽清单及完成后的逻辑 loader/RNG 快照，保留原始每槽容量，只限制物理 CPU worker 数。该原始数据读取本身不能证明噪声、训练显存、数学结果或性能通过。

实际作业已启动：协调器 `/tmp/run_tp_pp_raw576_native_audit_20261118.py`，输出 `outputs/megatron_tp_pp_raw576_native_audit_20261118`。先用 4 个物理 worker 生成完整清单，成功后自动调用 `verify_tp_pp_native_data_replay.py --logical-dp 192`，以独立、未虚拟化的原生 loader 逐流核对全部槽位。已进入实际媒体读取并交付首槽，尚未产生完整原始数据验收通过结论。新工具 Ruff lint/format 与 Python 语法检查通过。


GPU 原生叶子参数参考现已完成完整 36 层、实际非零 Adam 状态的三更新，norm 为 123.1696090698、20.8993549347、11.7213191986，峰值已分配 CUDA 为 66507893760 bytes（约 61.94 GiB）。参考保留完整文本，但 VIDEO/Action 仍为既定裁剪诊断输入，不是原始 packing 容量。生成记录 `full_source_gpu_leaf_reference_generation_20261118.json`；其完成不代表 TP/PP 比较通过，四卡连续三步对照仍待执行结束。


四卡 TP2/PP1/DP2 的真实 12 槽完整模型严格重启已排队，协调器 `/tmp/run_tp_pp_true_dp2_after_gpu_leaf_math_20261118.py` 等待已确认存活的 GPU 三步数学对照退出后启动。使用已独立验证的 TP2/PP1 canonical bundle、原始每槽 packing、逻辑 DP4/global12、并行准备和关闭 VAE AOT；A 从 1200 连续三步，B 一步保存退出后新进程连续两步。输出预定为 `outputs/megatron_tp_pp_true_dp2_real12_strict_restart_20261118`。当前尚未启动 GPU 训练，不将排队视为 DP2 已通过，也未据此断言显存可容纳。


补充定位证据：CPU 叶子参考与 GPU 叶子参考在全部 811 个参数、每张量最多 32 个等距元素的抽样中，前两更新的 raw gradients、master、两项 moments 全部相同；第三更新 raw gradients 抽样仍完全相同，裁剪/更新后的部分状态有极小差别。报告 `native_cpu_vs_gpu_leaf_reference_samples_20261118.json` 明确仅为抽样，不能代替完整张量比较。新 GPU 参考的 TP2/PP2 第二更新 observation 已完成，全局梯度相对 L2=9.02485e-5，但局部梯度 gate 失败，与旧参考观察一致，尚未解决连续三步问题。

新增数学隔离开关 `--native-linear-autograd`（仅测试 helper、默认关闭），保持 MCore 参数切片、TP 通信、PP、DDP 和优化器，单独改用原生 `F.linear` 的局部 autograd，拒绝 SP/累积融合/混合输出 dtype 等未覆盖选项。TP1/PP2/DP2 完整三步隔离排在真实 DP2 重启之后、完整 576 槽之前；源输入与固定门槛均不变。包装脚本 `/tmp/run_tp_pp_native_linear_isolation_before_full576_20261118.py` 暂缓 full576 协调器，并通过 `finally` 自动恢复它。该隔离尚未执行，不改变生产默认，不代表验收通过。相关两份 Python 文件 Ruff lint/format 与语法检查通过。

完整 576 槽严格恢复与四卡原生 FSDP 三更新参考也已建立串行执行队列，协调器 `/tmp/run_tp_pp_full576_and_native_after_dp2_20261118.py`；保留逻辑 DP192、原始容量、AOT 与 compile/full AC，依次输出 `outputs/megatron_tp_pp_full576_strict_restart_20261118` 和 `outputs/megatron_tp_pp_native_full576_aot_three_updates_20261118`。前述局部 autograd 隔离结束后才恢复启动。排队不是已经运行或通过；与并行原始数据核验重叠的窗口也不能自动作为无干扰稳态性能结论。


GPU 叶子参考的 TP2/PP2 三步梯度 observation 已全部取得：全局相对 L2=1.00413e-5、9.02485e-5、6.27623e-4，与旧 CPU 叶子参考的 capturable 候选观察相同。首更新四 rank 通过，后两更新四 rank 均有 gate 失败，第三更新全局梯度也超界；因此将原生参考保留在 GPU 没有消除既有三步误差。汇总 `full_source_gpu_leaf_three_update_gradient_observations_20261118.json` 覆盖 12 份 observation 的 SHA 和结果，明确不替代仍在完成的最终 optimizer/state 比较；不调整固定门槛、不宣称数学验收通过。


GPU 叶子参考的完整 TP2/PP2 三更新对照已终止，返回码 1，四份最终报告均失败；全部 4866 项 compute/master 检查仍精确一致，但梯度与 optimizer 状态存在超界。最终汇总 `full_source_gpu_leaf_three_update_final_summary_20261118.json`。CPU→GPU 参考放置并未修复该误差，失败保留，不把 observer 第一更新通过当作三步通过。

四卡 TP2/PP1/DP2 真实原容量 12 槽重启验收现已自动启动 A 分支，命令明确覆盖 TP2/PP1、逻辑 DP4/global12，目标连续 1201–1203。仍待训练更新、保存退出、恢复两步和最终状态比较，当前不宣称 DP2 容量或完整恢复已经通过。


TP2/PP1/DP2 原容量作业已结束，首更新未完成即 OOM，协调器返回码 1；rank0/1/2 均留下 `oom.iter000001200.rank*.json`。rank1 当时 PyTorch 已分配 75.10 GiB、未分配缓存仅约 397 MiB，在申请 700 MiB 的 `(59648, 6144)` BF16 中间结果时失败，属于实际容量不足证据，不能外推 TP2/PP2 的 GA3 显存需求。汇总 `true_dp2_real12_original_capacity_oom_20261118.json`；原始日志与 OOM 文件保留，未缩小输入后冒充通过。

新增仅诊断入口 `tools/run_tp_pp_with_activation_offload.py`，在现有 driver 的 forward/backward 外使用 PyTorch saved-tensor CPU offload，保留输入容量、计算模型、MCore 调度和优化器。严格恢复 runner 的 `--activation-cpu-offload` 显式选择该入口并把其源码纳入 hash guard，summary 明确记录诊断开关，默认关闭。该路径不能作为 baseline 对齐性能结果。原容量 DP2 重试已排在完整 576 槽 TP/PP 与原生参考之后，协调器 `/tmp/run_tp_pp_true_dp2_offload_after_full576_20261118.py`；尚未实际运行，不能宣称已解决 OOM。相关 lint/语法检查通过；重启 topology 契约测试仍为 9 passed（3.53 秒）。


### 真实运行、clipping 后端与参考卸载保真度

TP1/PP2/DP2 原生局部 Linear autograd 隔离已完成并失败，三更新全局梯度相对 L2=1.68227e-6、9.47642e-5、1.00083e-3。四份最终报告明确 `native_linear_autograd=true`，未改变固定门槛。汇总 `full_source_native_linear_tp1_three_update_summary_20261118.json`。因此不能仅将既有偏差归因于 MCore 局部 Linear backward；包装脚本已自动恢复 full576 协调器，不存在仍暂停的 full576 父进程。

完整 576 槽 TP2/PP2 A 分支现已实际启动四个训练 rank，完成源状态导入后进入原始媒体/VAE 准备；启用正式 callback 的运行时 AOT 构建（当前队列没有设置 consume cache 环境），不将它表述成消费预构建 package。后续原生参考也使用相同默认 AOT 模式。当前尚未完成一整个 576 槽更新，不能声明容量、重启或稳态性能通过。

发现当前 TE 2.12+cu130.torch210 已有 `multi_tensor_applier`、`multi_tensor_l2norm`、`multi_tensor_scale`，只缺 `multi_tensor_scale_tensor`；MCore0.20 一次导入四项导致整体失败，实际 norm 后端为 `megatron.core.utils.local_multi_tensor_l2_norm`，会逐参数执行 CUDA norm 再在 CPU 汇总。并非 TE 未安装。新增 `tools/qualify_tp_pp_clip_backend.py` 使用原先真实梯度 fixture 的全部 811 参数、189786 元素、三更新分布核验：已有 TE scale 与原生 foreach 均逐位相同，TE norm 与原生 foreach 的样本结果相同，对 FP64 样本 oracle 的相对误差低于 1e-6。样本 norm 的中位 wall time 为 fallback 17.24–17.50 ms、TE 0.55 ms、原生 foreach 2.32–2.40 ms。报告 `clip_backend_actual_samples_20261118.json` 明确这是并行作业下的小型 kernel 诊断，不能外推整步收益；生产未切换后端。

新增 `tests/megatron/tp_pp_activation_offload_smoke.py` 独立对照未卸载 GPU 基线，覆盖 compiled block/full AC、混合冻结参数、每更新两 microbatch、step1200 非零原生 Adam 三更新。默认 `save_on_cpu(pin_memory=True)` 严格逐位对照失败：首更新两个已保存张量布局改变，首参数梯度相对差 1.0046e-7、最大绝对差 3.73e-9；该差异仍满足原有 1e-4/2e-6 数值门槛，但不能宣称逐位相同。失败保留为 `activation_offload_default_layout_failure_observation_20261118.json`。源码检查确认默认 pinned 分配不保留转置 stride。

候选 `tools/tp_pp_activation_offload.py` 用 `empty_like(..., memory_format=preserve_format)` 保存/恢复 dense 布局；相同独立检查的三更新输出、raw gradients、norm、参数、moments、step 全部逐位一致，实际捕获 24 次 CUDA saved tensor、7584 bytes，布局变化为 0。报告 `activation_offload_preserve_layout_compile_full_ac_qualification_20261118.json`。非 dense view 仍可能改变布局，整模型未验收。候选未接入当前受 source guard 冻结的训练入口，不改变原容量正式作业。

纠正旧 GPU 叶子小资格测试的覆盖表述：其 baseline forward 也位于同一个 saved-tensor offload 上下文，因此它验证了 GPU 叶子/冻结流式参数及 Adam，但不能证明未卸载 GPU backward 的原样数值。此次新增的未卸载基线对照补上了该项，保留旧报告而不将其当作完整保真证据。

完整数学 helper 新增默认关闭的 `--preserve-saved-layout`，只用于原生 CUDA 参考，保留原参数、输入 SHA、常量 SHA、RF loss 与全部固定门槛。独立新参考与 TP1/native Linear、TP2 默认、TP2/native Adam 三个连续三更新对照已排在当前完整正式作业之后，协调器 `/tmp/run_tp_pp_preserved_layout_math_after_full_runs_20261118.py`，输出预定 `outputs/megatron_tp_pp_preserved_layout_math_sequence_20261118`。旧参考和失败不覆盖；小模型保布局通过不能代替整模型通过。


### 首个完整 576 槽作业的实际失败与显存隔离

完整 TP2/PP2 首更新已准备全部 576 槽，spool 总计 65338371264 bytes（约 60.85 GiB）；实际训练日志供给至少 112 槽后，在 backward 的 `(59648,6144)` BF16 分配上 OOM，需要 700 MiB、设备仅剩约 640 MiB，rank0 PyTorch 已分配 72.14 GiB、reserved 未分配约 1.33 GiB。该作业返回码 1，未完成一个 update，后续原生参考没有启动。失败记录 `full576_original_capacity_backward_oom_20261118.json` 保留原始 allocator 数值；“已供给 112 槽”不等于已经完成 112 次 backward，确切失败槽位未知。没有将完整准备成功当成训练容量通过。

只读扫描全部 prepared headers 后，首个文本超过 short-UND 5120 的槽为全局 index385（original micro2/rank1）；前面发生 OOM 的供给窗口仍为短 UND。完整 576 槽中有 80 个 long-UND 槽，尚未训练到它们；不能将此次 OOM 归因于第一次 long-UND 切换，也不能宣称其容量已验证。证据 `full576_prepared_text_capacity_audit_20261118.json`。

新增 `tools/replay_tp_pp_prepared_memory.py` 隔离：使用生产导入/compile/full AC/MCore 调度/原生 loss/优化器，逐个回放这同一份完整 576 槽已准备输入，不构造驻留 VAE；记录每次 forward/backward 的分配/保留/峰值显存和输入计数。完成原始 optimizer update 后立即写诊断结果并退出，不推进逻辑 loader/RNG、不发布 checkpoint、不冒充完整续训。它已排在保布局完整数学对照之后，协调器 `/tmp/run_tp_pp_prepared_memory_after_layout_math_20261118.py`。当前只是排队，尚未确定 OOM 是驻留准备模块、峰值差异还是跨微批保留所致。

实际 CPU-only spool I/O 诊断也已完成，真实 encoded payload 101321049 bytes，shared buffered write/read 中位数 0.2210/0.0594 秒、local 为 0.0822/0.0520 秒，全部 readback payload SHA 相同，源文件 stat 不变且 CUDA 未初始化。报告 `prepared_spool_shared_vs_local_io_20261118.json`。该测试不含 CUDA 同步、编码和 H2D，也不保证冷 cache；不能把并行 writer 的时间相加为 update wall time，或把此前 spool write 计时全部解释为 NFS。生产 spool 路径未改。


DP2 saved-tensor CPU offload 诊断已实际连续完成 1201–1203，rank0 update wall time 为 168.35、62.61、60.62 秒（含首次预热，非独立性能结论）；B 分支已完成 1201 并进入保存/恢复流程，loss 与 A 的 1201 完全相同。仍等待两次恢复更新及最终完整状态比较，不能将这段前缀称为重启已通过。

原生 FSDP 的完整 576 槽三更新已独立排队，不再由 TP/PP 完整作业成功与否决定是否运行。它在 decoder 显存回放结束后执行，使用已验证的 AOT consume cache 和原始容量，输出预定 `outputs/megatron_tp_pp_native_full576_consume_three_updates_20261118`；协调器 `/tmp/run_tp_pp_native_full576_consume_after_memory_replay_20261118.py`。当前尚未启动，不构成新的基线吞吐结果。

本次失败还暴露日志发布时机不足：rank-level 路径的完整 replay plan 目前只在 update 成功后落盘，因此 OOM 前虽已计算 raw/noise/packed 记录，但没有发布完整记录文件。不能用另一次 raw-plan 读取来补造该次 TP 运行的 raw SHA 证据；后续应在并行准备结束时立即发布已经计算的记录。该行为修复尚未写入受 source guard 保护的生产源码。

活动作业导航为 `artifacts/megatron_tp_pp/active_job_handles_20261118.json`，记录协调器 PID/create_time、脚本与依赖；文件只是定位信息，观察或操作前必须重新检查真实进程，不能把文件本身视为存活或完成证据。


DP2 原容量 saved-tensor offload 的 A/B 三更新现在均已完成，三更新共 36 槽 raw/noise/packed/source/count 全部相同；loss 和 grad norm 的逐 rank 结果并非全部逐位相同，具体误差见前缀报告，不据此提前判断通过。前缀报告 `true_dp2_offload_three_update_prefix_20261118.json` 明确尚不接受严格重启，仍等待 CPU 完整模型/master/moments/step/scheduler/逻辑状态比较。当前 GPU 训练 rank 已退出，不能把协调器 CPU 比较阶段误判为作业丢失。


补充 DP2 offload 前缀误差：三更新 mean loss 的 A/B 相对差分别为 0、9.40405e-5、2.51950e-4，但逐 rank grad norm 的最大相对差分别为 3.34256e-5、8.09196e-2、3.51105e-2。即保存前的第一更新也不是 backward/norm 全部逐位一致，不能仅凭后两步 loss 在 1% 内就判定恢复正确。完整状态比较仍须通过；也需隔离 BF16 backward 确定性与真实 DP 行为，当前不将问题预先归因于 checkpoint 单点。

DP2 offload 最终完整状态比较已结束，`outputs/megatron_tp_pp_true_dp2_real12_activation_offload_strict_restart_20261118/summary.json` 为 `passed=false`。输入全部一致，完整模型/master/moments 仍未满足固定门槛；该候选不能作为真实 DP2 严格恢复已经通过的证据。

### 保布局参考后的裁剪算术定位

完整 36 层、全部源参数和非零 Adam 状态的新原生 GPU 叶子参考已完成，保存 dense tensor 布局，三更新 norm 为 123.16944885253906、20.898557662963867、11.720803260803223。TP1/PP2/DP2 native Linear/native Adam 对照的首更新 raw gradient 全局相对 L2 为 0、余弦为 1；后两更新分别为 1.65322e-5、4.22502e-4，完整三更新固定门槛仍失败。完整文本保留，但 VIDEO/Action 仍为数值诊断裁剪，不能代表生产容量。报告位于 `outputs/megatron_tp_pp_preserved_layout_math_sequence_20261118`。

首更新原始梯度相同后，原生 FP32 norm 为 123.16944885253906，MCore 返回的 Python float norm 为 123.16945184252262。裁剪系数分别为 0.008118896745145321 和 0.008118895813822746，相差一个 FP32 ULP。新增 `tools/tp_pp_native_clip_arithmetic.py` 保留 MCore 的梯度所有权及 norm 归约，只将 norm、epsilon 加法和倒数缩放改为原生 FP32 张量算术。实际 811 参数、三更新梯度样本检查中，候选与原生裁剪全部逐位一致；旧算术在前两组各有 810 个参数样本不同。证据 `native_clip_arithmetic_actual_samples_20261118.json`，只证明样本裁剪行为，不证明完整训练轨迹。

新增独立诊断入口 `tools/run_tp_pp_math_with_native_clip.py`，在不修改当前受 guard 保护的完整模型 helper 的情况下安装候选，同时记录入口/helper hash。TP1 native Linear/native Adam、TP2 native Adam、TP2 native Linear/native Adam 三组连续三更新已调整到当前数学对照之后、显存回放之前执行，协调器 `/tmp/run_tp_pp_native_clip_before_memory_20261118.py`；原始输入、参考、固定门槛保持不变，不使用逐步重置到参考状态。当前尚未执行，不将排队视为通过。生产默认裁剪行为未改。

上述重排只暂时停止没有训练子进程的显存回放协调器 PID1063649，原生完整容量和 selective offload 协调器继续等待该进程，当前数学训练不受影响。候选协调器及其外部 supervisor 均在结束路径恢复 PID1063649，并核验 create_time 防止 PID 复用；旧末尾排队的裁剪协调器 PID1113151 已结束，避免重复运行。

TP2 默认算术相对新保布局参考的三更新 raw-gradient observation 已落盘，rank0 全局相对 L2 分别为 1.11385649e-5、1.37614132e-4、7.82109438e-4；后两更新 observation 的 `passed=false`。证据为 `artifacts/megatron_tp_pp/full_source_real_rf_preserved_layout_tp2_default_fp32.update{0,1,2}.observation.rank0.json`。该结果进一步确认首步梯度接近不能代替连续三更新验收；完整各 rank optimizer 状态汇总仍由实际运行产生，不据此预判裁剪候选足以修复全部偏差。

上述 TP2 默认算术的四 rank 最终报告现已全部产生且均失败。4866 项 compute/master 一致性检查全部精确相等，master 对原生参考的既定门槛通过；2496 项不通过的状态比较均为第二/第三更新的 `exp_avg` 或 `exp_avg_sq`，并非再次出现 compute 未同步到 master。汇总为 `preserved_layout_tp2_default_state_failure_summary_20261118.json`。当前继续隔离 Adam、裁剪及本地 Linear 算术的贡献，不把该分类结果当作已证明单一根因。

补充实际调用入口检查：`tests/megatron/tp_pp_clip_arithmetic_smoke.py` 现在验证 `clip_grads` 与 optimizer 模块内绑定的两个函数别名，并通过 optimizer 实际别名执行候选裁剪。使用同一实际梯度 fixture 的三组、811 参数样本均与原生裁剪逐位一致，旧函数负对照前两组各有 810 个参数样本不同；报告 `native_clip_arithmetic_mcore_bound_alias_actual_samples_20261118.json` 的 `mcore_optimizer_bound_alias_verified=true`。这是实际 MCore 调用入口的补充资格检查，未扩大为完整三更新已通过的声明。

在新保布局参考下，仅启用 native Adam 算术的 TP2 对照第二更新 observation 已产生：全局 raw-gradient 相对 L2 为 1.52309853e-4，`passed=false`；首更新与默认 TP2 的 raw gradient 相同。证据 `full_source_real_rf_preserved_layout_tp2_native_adam_fp32.update1.observation.rank0.json`。因此单独更换 Adam 算术仍未满足连续轨迹门槛，第三更新及完整状态报告继续由该次作业产生。

保布局数学序列已实际结束，三种配置最终均失败，`outputs/megatron_tp_pp_preserved_layout_math_sequence_20261118/summary.json` 为 `passed=false`。TP2 native Adam 第三更新全局梯度相对 L2 为 1.16590276e-3；没有通过只切换 Adam 修复连续轨迹。结束前独立复核 838 个受 guard 保护文件无变化。裁剪修正序列已自动启动 TP1 native Linear/native Adam 分支，实际训练子进程存在，后续仍需完成三更新与完整状态比较。

待运行的 decoder-only 显存回放工具已补充初始化显存事件和异常失败 summary：分别记录已开始 forward 与已完成 backward 数量、allocated/reserved/peak、异常类型及只读 spool stat 是否改变。异常继续以非零状态退出，不发布 checkpoint，不将日志供给数量解释为已完成反向。该诊断入口的 Ruff 检查及格式检查通过；尚未实际执行容量回放。

### 并行准备路径已有性能记录的阶段核算

重新核算已完成且输入逐槽匹配的 1205 无 profiler 单点：实际最慢 rank1 的 49.120714 秒由并行准备 19.460082 秒（39.62%）、forward/backward 主机区间 29.588202 秒（60.24%）、optimizer+sync 主机区间 0.072273 秒（0.15%）组成，剩余约 0.000158 秒。原生同更新端到端为 39.834613 秒。报告 `matched12_parallel_preparation_host_phase_accounting_20261118.json` 保存原始四 rank 日志及输入比较报告 hash；另列 1202/1203 预热观测，不把它们合并为稳态结果。

这些区间来自同一个实际最慢 rank，未把各 rank 的阶段最大值相加，也未把 owner 记录的并行 worker 耗时之和当作 wall time。forward/backward 包含通信、CPU 工作和等待，optimizer+sync 也不是独占 GPU kernel 计时；原生尚无相同阶段分解，不能将双方 9.286 秒总差直接归因于某个阶段。后续性能定位应优先核查准备及 forward/backward，仍需新的并行路径 Nsight 才能拆分 GPU 计算、通信与气泡。

性能比较工具现已自动输出实际最慢 rank 的上述 host phase accounting；缺少独立阶段记录时返回 null，不从跨 rank 最大值拼凑。新增回归专门覆盖准备最慢 rank 与整步最慢 rank 不同的情形，完整性能比较测试 6 项通过。用同一真实五步日志重新执行比较后，60 槽输入及 BF16 loss 门槛仍通过，49.120714/39.834613 秒端到端结果保持；新报告 `matched12_parallel_preparation_with_host_phases_20261118.json` 保留旧报告而不覆盖。该工具增强未改变训练或扩大单点性能结论。

裁剪修正序列的 TP1/PP2/DP2 native Linear/native Adam 分支已实际完成前两更新：两次全局 raw-gradient 相对 L2 均为 0，四 rank 的梯度/输出 observation 均通过。第二更新未修正裁剪时为 1.65322e-5，此次为 0；原始参考、输入、三更新计划及全部门槛不变，不采用逐步重置到原生状态。证据 `full_source_real_rf_preserved_layout_native_clip_tp1_native_linear_fp32.update{0,1}.observation.rank*.json`。第三更新与最终 master/moments/step 状态门槛仍待完成，TP2 组合也尚待执行。

真实 BF16 DP2 的失败仍为独立未完成项。原容量 TP2/PP1/DP2、global12 的 selective2 严格重启已排在 decoder-only 显存回放之后、原生 full576 长作业之前；使用同一 canonical TP2/PP1 bundle、原始 packing、compile/full AC、关闭 tokenizer compile、并行准备，A 三更新，B 一更新保存/全部退出后新进程恢复两更新。输出预定 `outputs/megatron_tp_pp_true_dp2_real12_selective2_strict_restart_20261118`。候选只卸载前两个 block 的保存激活并保留参数 CUDA storage/dense 布局，不能把小资格检查当作原容量已通过。

该队列通过 `/tmp/run_tp_pp_true_dp2_selective2_before_native_full576_20261118.py` 及独立 supervisor 等待显存回放协调器 PID1063649；当前仅暂时停止没有训练子进程的原生 full576 协调器 PID1071709，二者结束路径均核验 create_time 后恢复它。裁剪对照结束后仍会先恢复显存回放，之后才启动 DP2 实验，避免四卡作业重叠。当前 DP2 selective2 只是排队，生产默认路径及原始失败证据不变。

裁剪修正的 TP1/PP2/DP2 native Linear/native Adam 分支现已完整结束，returncode=0，四 rank 最终 `passed=true` 且 gate_failures 为空：三次全局 raw-gradient 相对 L2 均为 0；7317 项 DP-owned master/moments 比较的最大相对及绝对误差均为 0；2439 项 compute/master 检查全部精确一致，输出/norm/step/frozen-source 门槛全部通过。连续三更新，没有 teacher forcing，入口/helper hash 在运行前后保持相同。汇总 `preserved_layout_native_clip_tp1_three_update_summary_20261118.json`。这是实际 36 层源参数、非零 Adam 状态及完整文本的数值诊断，VIDEO/Action 仍裁剪；TP2、BF16 原容量严格重启及 full576 均未因此自动通过。下一个 TP2 native Adam/native clip 分支已实际启动。

TP2 保留 MCore local Linear、native Adam/native clip 分支的第二更新 observation 已产生：全局 raw-gradient 相对 L2=1.24241773e-4，`passed=false`，首更新为 1.11385649e-5 并通过。证据 `full_source_real_rf_preserved_layout_native_clip_tp2_native_adam_fp32.update1.observation.rank0.json`。因此裁剪修正虽然解决了 TP1 native Linear 的连续漂移，仍不足以使当前 TP2 local Linear 路径通过固定门槛；该次第三更新继续运行，之后已排队 TP2 native Linear/native Adam/native clip，隔离 local Linear 的贡献。

该分支第三更新 observation 也已产生，全局梯度相对 L2=9.12910222e-4，仍失败；完整状态汇总尚待该次进程完成。没有将首步通过或相对旧对照的局部变化当作 TP2 三更新已通过。

显存回放额外记录了 non-interleaved 1F1B 的在途槽位 FIFO，并将 before/after backward 与具体槽位绑定。安装的 MCore schedule 在稳态及冷却阶段均从 input/output 列表头部取反向，与该 FIFO 对应；失败 summary 保留 active forward/backward 及所有在途槽位，成功路径要求队列排空。这只增加 CPU 标识和日志，不改变张量/调度/优化器。实际完整容量回放仍未启动，不能提前保证能复现同一个 OOM。

TP2 MCore Linear/native Adam/native clip 分支最终四 rank 均失败，14598 项 owned-fragment master/moments 比较中有 2501 项不通过，4866 项 compute/master 检查仍全部精确一致。五组已完成对照汇总为 `preserved_layout_arithmetic_ablation_summary_20261118.json`，保存 20 份原始 rank 报告 hash；所有组均为连续三更新，禁止逐步回灌原生状态：

| 配置 | 第一次梯度相对 L2 | 第二次 | 第三次 | 完整门槛 |
| --- | ---: | ---: | ---: | --- |
| TP1 native Linear/native Adam/旧裁剪 | 0 | 1.65322e-5 | 4.22502e-4 | 失败 |
| TP1 native Linear/native Adam/native 裁剪 | 0 | 0 | 0 | 通过 |
| TP2 MCore Linear/默认 Adam/旧裁剪 | 1.11386e-5 | 1.37614e-4 | 7.82109e-4 | 失败 |
| TP2 MCore Linear/native Adam/旧裁剪 | 1.11386e-5 | 1.52310e-4 | 1.16590e-3 | 失败 |
| TP2 MCore Linear/native Adam/native 裁剪 | 1.11386e-5 | 1.24242e-4 | 9.12910e-4 | 失败 |

该表只覆盖完整参数、完整文本且 VIDEO/Action 裁剪的 FP32 诊断；TP2 native Linear 分支仍由当前序列执行，不将 TP1 通过替代 TP2 或生产容量验收。

优化器模式来源已直接核对：`cosmos_framework/utils/generator/optimizer.py:106` 的原生 FusedAdam 分支强制 `capturable=True`，而 MCore0.20 在 optimizer 构造时使用默认关闭的 `OptimizerConfig.optimizer_cuda_graph` 决定 capturable。因此 native Adam 算术候选对齐的是实际原生模式，不能将同一 Adam 超参数等同于完全相同的浮点更新。当前候选通过 step pre-hook 保持 LR/step 的 CUDA dtype，未在生产入口启用图捕获；正式默认路径的该项对齐仍未落地，需后续实施及验收，不能因 TP1 诊断通过就声称默认路径已修复。

TP2 native Linear/native Adam/native clip 的第二更新 observation 已产生，raw-gradient 全局相对 L2 同样为 1.24241773e-4，`passed=false`；前两次全局误差指标与 MCore Linear/native Adam/native clip 相同。因此目前没有证据支持仅替换 local Linear autograd 就能解决 TP2 漂移，不据此更换生产默认 Linear。该分支第三更新及完整状态比较继续运行。证据 `full_source_real_rf_preserved_layout_native_clip_tp2_native_linear_fp32.update1.observation.rank0.json`；这些相同的汇总指标不等于证明两分支的全部实际梯度逐位相等。

该数学序列现已全部结束，TP2 native Linear 的第三更新误差同为 9.12910e-4，四 rank 完整门槛均失败；其 14598 项 master/moments 比较有 2501 项失败，4866 项 compute/master 检查通过。六组最终汇总及 24 份原始报告 hash 保存为 `preserved_layout_arithmetic_ablation_complete_summary_20261118.json`。裁剪协调器和 supervisor 已正常退出并恢复显存回放；无 VAE 驻留的原始 full576 decoder 回放已实际启动，并已完成至少 134 次 backward，尚未完成全部 576 槽，也尚未训练至首个 long-UND 槽385。

### 补充未裁剪真实输入的 FP32/Flex 数学入口

完整模型 helper 新增默认关闭的 `--original-capacity --retain-full-text --attention-backend flex`，保留原 recipe 的容量设置，加载两个原始完整 prepared 槽，不删除样本、token、mask 或 target；浮点输入仅转为 FP32。既有 cropped 诊断与失败报告保留，门槛仍为原固定值。新参考显式记录容量模式与 attention backend，TP 对照要求二者及完整输入 hash 一致；`--activation-cpu-offload` 为 TP 诊断保留 dense saved-tensor 布局，未修改生产默认路径。

实际输入加载资格检查已通过：rank0/rank1 各 6 个样本、57600 VIDEO tokens、366 ACTION tokens、1064/1030 text tokens，每槽全部 120 个 tensor 字段均在 FP32 转换后与源值精确相同；源 `.pt`/`.loss.pt` SHA 保留，CUDA 未初始化。报告 `original_capacity_rf_inputs_actual_capture_qualification_20261118.json`。加载时清除已有 device-specific packing cache，避免 CPU 映射后的 tensor 与旧 CUDA device 描述不一致，后续在实际执行设备重建派生 metadata。

新增 `tp_pp_flex_fp32_smoke.py` 使用生产 `sample_block_mask/sparse_attend`，对照 FP32 math SDPA，覆盖 GQA32/8、非均衡样本和 padding。输出及 dq/dk/dv 均有限，相对 L2 为 3.29e-7 / 6.42e-7 / 6.89e-7 / 4.50e-7，全部满足原固定门槛；报告 `flex_fp32_math_sdpa_gqa_qualification_20261118.json`。Ruff 检查与格式检查通过，完整模型 helper 的 CLI 检查通过。这只是 kernel 和输入资格检查，不是完整模型或原容量训练已通过。

新的原生 GPU-leaf/Flex 参考与 TP2/native Adam/native clip 三更新对照已排在现有完整容量作业之后，协调器 `/tmp/run_tp_pp_original_capacity_flex_after_full_runs_20261118.py`，输出预定 `outputs/megatron_tp_pp_original_capacity_flex_fp32_sequence_20261118`。参考或 TP OOM 时保留失败，不缩小输入重跑冒充通过；该实验仅有两个完整 packed 槽，仍不等于一次 global576 更新或正式性能验收。

### 无 VAE 驻留仍 OOM，周期性循环引用收集对照启动

decoder-only full576 回放已实际失败：rank0 完成 204 次 backward、开始 206 次 forward，在 `(1200,1,12)`（全局零基 index204）的 backward 申请 700 MiB 时 OOM；当时 allocated 73.26 GiB、reserved 74.46 GiB、峰值 allocated 73.83 GiB、设备 free 569.94 MiB。该槽有 11 样本、1945 text / 59400 VIDEO / 671 ACTION tokens，尚不是首个 long-UND 槽385。原始 spool stat 不变，没有完成 optimizer update 或发布 checkpoint。证据 `full576_decoder_no_vae_oom_and_memory_growth_20261118.json` 及原始 `rank0/summary.json`。

同为一个在途微批时，rank0 after-backward allocated 从 B1 的 49.081 GiB 增至 B204 的 56.548 GiB；rank1 曾从 B160 的 55.052 GiB 回落到 B192 的 49.774 GiB。末 stage 零在途时也有增长。这提示未及时释放引用或其他运行时保留，需要实验区分，尚未证明 Python 循环引用为唯一根因；不能再将 VAE 驻留解释为唯一容量问题。

失败回放的证据已写出，但进程停在分布式退出阶段，随后只对已保存 fatal OOM 的 torchrun/PID identities 做 SIGTERM/SIGKILL 清理，全部原失败 worker 已退出。诊断工具异常路径现先关闭证据文件并退出失败 rank，让 torchrun 终止 peers，避免再次在 NCCL teardown 等待。成功路径仍正常销毁进程组。

新的同容量回放仅加 `--collect-cycles-every 16`，在每 16 次完成的 backward 后调用 `gc.collect()`，记录回收对象数、reclaimed allocated bytes 和 host time；不调用 empty_cache、不缩小输入、不加 activation offload。四 GPU 作业已经实际启动，输出 `outputs/megatron_tp_pp_full576_decoder_memory_gc16_replay_20261118`，836 个 core/config/probe 文件 hash 已保存，当前等待是否回收真实 GPU allocation 及能否完成全部 576 槽。

队列顺序现为 GC16 显存回放 → 原容量 DP2 selective2 严格重启 → 原生 full576 三更新 → TP2/PP2 selective2 full576 严格重启 → 原容量 FP32/Flex 数学对照。`/tmp/run_tp_pp_decoder_gc16_before_true_dp2_20261118.py` 与独立 supervisor 暂停没有训练子进程的 DP2 协调器 PID1137646，并在结束路径核验 create_time 后恢复它；原生 full576 协调器仍由 DP2 实验原有双重恢复逻辑管理。当前只有 GC16 GPU 回放运行，不将排队当作验收通过。

GC16 回放已实际完整结束并通过四 rank 检查：每 rank 恰好 576 forward/576 backward，原始 prepared 槽均保留，有限梯度 norm 为 0.10712552373896067，optimizer 推进至1201。最大 allocated 为 68.971 GiB；各 rank 的 36 次 GC 累计释放 26.54/27.37/33.42/33.42 GiB（累计值，不是峰值节省），GC host time 为 24.19/22.14/24.02/24.02 秒。836 个已记录源码/config/probe hash 均保持。汇总 `full576_decoder_gc16_complete_summary_20261118.json`。这是无驻留 VAE 的一次完整 decoder/optimizer 更新，未推进真实 loader/RNG、未保存 checkpoint，仍不等于正式全流程576及严格恢复验收。

新增训练 backend 独立模块 `memory.py`，将周期收集限定在一次 MCore schedule scope，异常退出也恢复原 backward 函数。新增真实 MCore backward、compiled/full AC、非零原生 Adam 三更新小资格检查：梯度/参数/moments/step 全部逐位一致，人工 GPU cycles 实际回收24MiB，异常恢复检查通过；报告 `microbatch_gc_mcore_compile_full_ac_native_adam_qualification_20261118.json`。模块尚未接入默认 driver；新增文件不在先前审计进程已经记录的文件集合中，原审计文件字节未改动。

新增 backend `optimizer_math.py` 保留原生 FP32 裁剪及 capturable FusedAdam 算术适配。实际811参数梯度样本的三更新适配均与原生逐位一致，候选汇总 `backend_native_adam_adapter_candidate_qualification_20261118.json`；包含 noncapturable 负对照的完整报告仍为 false 并保留，不用候选通过掩盖负对照差异。该模块也尚未接入默认 driver。

原生性能参考工具新增每16个局部 microbatch 回收（可用 `--microbatch-gc-interval 0` 禁用），每次更新报告实际收集记录，耗时计入 update wall time。严格重启工具新增显式同名参数和执行检查：正间隔要求每个 rank/更新的日志覆盖完整 backward 数及准确收集边界，不能仅凭环境变量认定启用；19项 runner 测试通过，相关 Ruff 检查/格式检查通过。先前12槽原生每 rank M3未达到16，不据此改写旧性能记录。

随后 DP2 selective2 原容量作业首更新仍 OOM：rank1/3 在申请700MiB `(59648,6144)` BF16 中间值时失败，allocated 75.21/74.78GiB；完整状态更新/重启仍未通过。原始失败目录 `outputs/megatron_tp_pp_true_dp2_real12_selective2_strict_restart_20261118` 保留。该协调器已结束并按预定恢复原生 full576 三更新，后者已实际启动；raw576生成也已完成，576槽6978样本，独立 unvirtualized native 数据比较正在运行。

为给已验证策略接入默认 backend 留出源码窗口，当前仅暂停尚无训练子进程的后续 full576 selective2 协调器 PID1102667，原生 full576 基线继续运行。默认 driver/train 文件须在当前原生 source guard 作业结束后修改；下一次正式 full576 应优先验证 GC16、不加 activation offload 的路径，并显式核验GC执行记录。原容量 FP32/Flex 协调器仍等待 PID1102667，替换/结束该协调器前必须防止原容量 FP32作业与新的正式训练重叠。目标仍未完成，不将新增独立模块视为默认入口已修复。

默认接入已在隔离副本完成：driver 每次 schedule 默认 GC16，可由 `COSMOS_TP_PP_MICROBATCH_GC_INTERVAL` 覆盖；恢复原生 optimizer/clip 算术，输出每 rank/更新的完整或未完成 GC记录；train 将 job root 传给 driver。两个 CPU 集成契约检查通过，覆盖梯度不变、完整日志、异常恢复及不完整标记。这里使用 CPU schedule test double，实际 MCore/compiled/full-AC/Adam 资格检查见前文，不将 CPU 集成测试扩大为完整模型验收。

待应用补丁 `/tmp/cosmos_tp_pp_default_runtime_policy_20261118.patch`、隔离文件 `/tmp/cosmos_tp_pp_default_runtime_policy_20261118/` 及 base/candidate SHA 已记录在 `pending_default_runtime_integration_20261118.json`；`git apply --check` 和 Ruff 检查通过。当前 workspace 的 driver/train 尚未修改，运行中的 native source guard 再次核对无变化。仅在 native 协调器1071709及训练子进程退出、重新核对 base SHA 后应用，再用实际完整训练验证。

另补充实际811参数样本的 CPU state_dict 序列化、新建优化器、恢复后两更新及 step 对照，重复安装适配器还检查 hook 不重复。该检查等待当前 native 基线退出后启动，协调器 `/tmp/run_tp_pp_optimizer_roundtrip_after_native_20261118.py`；这是优化器对象重建，不能替代新训练进程的 DDP 完整 checkpoint 恢复验收，当前尚未取得结果。

原生 full576 三更新作业现已完成首更新1201：四rank最慢耗时1813.734565秒，全局576槽/6978样本，loss=0.3195436308160425。其余两更新继续运行；首步不能当作稳态吞吐，也不能与12槽的39–41秒直接比较。观察时838个受保护源码文件SHA均未变化，media_failures记录文件数为0。完整四rank原始更新记录保存于 `artifacts/megatron_tp_pp/native_full576_first_update_observation_20261118.json`；同输入TP/PP性能比较仍待完成。

整步激活卸载诊断入口现在默认使用已独立验证的 `saved_tensor_cpu_offload` 保持dense布局；旧PyTorch实现通过 `COSMOS_TP_PP_ACTIVATION_OFFLOAD_POLICY=torch_default` 保留供复现旧失败。严格重启工具新增显式 `--activation-offload-policy preserve_layout|torch_default`，固定传递并在summary记录实际请求策略，避免继承环境造成口径不明。GPU资格检查入口也改为调用同一个策略工厂，尚待当前四卡作业退出后重跑。CPU入口接线检查通过（测试替身driver，精确输出/梯度、拒绝未知策略与重复安装），19项重启runner测试通过，三文件Ruff及格式检查通过；这些不构成真实DP2容量或完整重启通过。证据为 `activation_offload_entry_cpu_wiring_20261118.json`。当前native源码保护范围不包含这些诊断工具，受保护训练源码保持不变。

替代旧selective2队列的协调器1255305现已实际启动并等待native协调器1071709。它随后等待优化器恢复资格检查1219114，要求native三更新/最终保存成功、完整源码guard不变、恢复资格检查通过、补丁base/candidate SHA一致才应用默认runtime补丁；应用后再次核对candidate SHA、运行CPU集成/runner检查及GPU保布局卸载入口资格检查，最后启动无activation offload、显式GC16的原始global576严格重启（A连续三更新，B一更新保存退出后新进程再两更新）。当前只是排队，默认入口仍未修改、正式训练未启动。新协调器Ruff及预启动检查通过，路径和PID identity见 `artifacts/megatron_tp_pp/default_gc16_full576_replacement_coordinator_20261118.json`。旧协调器1102667保持STOP作为队列占位；只有新正式训练所有rank退出后才结束它，释放后续原容量FP32/Flex作业。前置检查失败会保留队列占位以避免GPU重叠。

性能比较CLI新增 `--logical-dp` / `--reference-microsteps`，完整576槽应显式使用192/3；未指定时仍只接受旧12槽，避免把部分输入当完整比较。报告新增匹配输入的样本/秒和text/vision/action各模态token/秒，输入不匹配时均为null。包含实际CLI的合成576槽完整覆盖/默认12槽拒绝回归，性能及重启工具共26项测试通过，Ruff检查通过。旧真实五步60槽日志重新核对仍全部输入/loss通过：无profiler单点1205的TP/PP为2.52439样本/秒，原生为3.11287样本/秒（吞吐比0.810953）；对应耗时仍49.120714/39.834613秒，未扩大为稳态结论。新报告 `matched12_parallel_preparation_throughput_rates_20261118.json` 保留旧文件。完整576槽比较预定首更新1201预热、1202/1203测量，命令见 `full576_matched_performance_measurement_plan_20261118.json`，当前未执行且仍不是长期稳态验收。

原生full576首更新的GC实际执行记录另已核算：各rank局部144微批、间隔16，共9次显式回收，host累计7.473–7.525秒；整步最慢rank0为7.525443秒，占1813.734565秒的0.415%。这些区间已计入update wall time，不能再相加，也不是独占GPU时间。九次显式回收前后allocated净减少均为0，不能据此推断原生没有其他自动回收或没有内存峰值。此前无VAE的TP/PP decoder-only GC16回放各rank36次累计22.14–24.18秒；两种范围不同，不将二者直接换算为完整生产性能差。报告 `native_full576_first_update_gc_host_accounting_20261118.json`；当前原生三更新尚未全部完成。

真实DP2保布局整步卸载重启已实际排队：协调器1272836等待原容量FP32/Flex协调器1162948结束，核验默认runtime应用SHA、卸载入口/策略SHA、小模型GPU资格结果及无训练rank残留后，启动TP2/PP1/DP2、原每槽容量global12、GC16、保布局CPU卸载的A3/B1+2严格重启。输出预定 `outputs/megatron_tp_pp_true_dp2_real12_preserved_offload_strict_restart_20261118`。这是为调查旧默认卸载699项状态差异的独立诊断，不将其混入无卸载full576性能，也不作为DP2完整576槽验收。当前只等待、未启动训练；协调器Ruff检查通过，PID identity/源码SHA见 `true_dp2_preserved_offload_replacement_coordinator_20261118.json`。

原生full576基线已连续完成1201/1202两个更新，最慢rank耗时分别1813.734565/1795.631634秒；实际样本分别6978/4469，说明固定576槽仍不能当固定样本batch。第二更新四rank峰值allocated最大70.992GiB，整步最慢rank2的显式GC合计7.301秒，均已包含在1795.632秒内。第三更新继续运行，最终数值状态保存和完整三更新结束尚未确认，仍不作为长期稳态或同输入TP/PP性能结论。完整原始四rank数据见 `native_full576_two_completed_updates_observation_20261118.json`。

随后原生full576第三更新1203完成，最慢rank耗时1795.299167秒、4807样本；三次更新为1813.734565/1795.631634/1795.299167秒。预定排除首步后的两步均值1795.465400秒，只是短窗口原生参考，不能与12槽耗时直接比较或宣称TP/PP收益。最终 `math_state_000001203/.math_complete` 已发布（数值对照快照，不是可恢复训练checkpoint）；协调器1071709及torchrun仍在退出阶段，源码guard最终结果和返回码尚未确认，后续仍等待它完全结束。原始四rank记录和观察状态见 `native_full576_three_completed_updates_observation_20261118.json`。

原生协调器随后已正常退出：返回码0，最终source_unchanged=true，所有训练子进程结束。实际811参数/189786元素样本的优化器CPU状态序列化、新建TE优化器、恢复后两更新检查已完成，三个 `native_streamed_vs_te_adapter_restart` 行全部bitexact，包含master/moments和step；源码guard不变。候选摘要 `backend_adapter_cpu_state_roundtrip_candidate_summary_20261118.json` 为passed=true。完整报告仍保留未适配TE的失败负对照，整体returncode=1，不改写为全报告通过；候选检查也不替代完整DDP/进程退出重启。

默认driver/train补丁现已实际应用：GC默认16、原生Adam/clip算术适配及每rank/更新GC执行记录已接入。应用后两文件SHA与隔离资格副本一致，workspace的CPU集成及重启runner共21项检查通过，Ruff通过；证据为 `outputs/megatron_tp_pp_default_gc16_full576_strict_restart_20261118_coordinator/{applied_runtime_hashes.json,cpu_integration.log,ruff.log}`。先前“待应用”是历史状态，`pending_default_runtime_integration_20261118.json` 已记录applied=true。协调器1255305当前运行保布局卸载入口GPU资格检查，成功后再启动无卸载global576严格重启；正式完整训练仍未通过，不将接入和CPU检查当作容量/连续性验收。

保布局卸载入口的实际GPU资格检查随后通过：逐block compile/full AC、2微批×3次非零Adam更新，输出/梯度/norm/参数/moments/step均精确相等，24次保存未改变布局；小模型峰值约65MiB。这不构成真实DP2完整状态验收，报告为 `activation_offload_production_entry_compile_full_ac_qualification_20261118.json`。无activation offload的global576/GC16严格重启现已实际启动runner（子进程1333101），目标A连续三更新、B一更新保存退出后新进程两更新。该runner的新源码guard已生效，运行结束前不再修改受保护训练/比较工具。

新增独立画像入口 `tools/run_tp_pp_full_update_profile.py`，补上旧短窗口从decoder才开始、漏掉并行准备的覆盖缺口。TP从并行准备入口捕获至optimizer结束，原生从首个prepare_slot捕获至144次局部累积后的optimizer结束；开始和正常结束处跨rank同步，避免首个rank的Stop截断其他rank的CUDA工作。异常路径保留部分画像并跳过结束barrier，继续抛出原异常。VAE初始化、末尾统计和checkpoint在区间外；加入的同步和profiling开销只用于诊断，不能作为未捕获的稳态吞吐。4项CPU边界检查通过（模拟CUDA profiler，包含TP准备、全部144微批、正常结束同步与异常退出），Ruff/格式检查通过；尚未取得实际Nsight画像。

完整global576的双方画像协调器1378370已启动并等待DP2协调器1272836，排在当前完整重启、原容量FP32/Flex数学检查及DP2保布局诊断之后。两边均从同一canonical1200、原始容量、GC16、AOT consume开始，各跑一完整更新，导出SQLite及分进程NVTX/CUDA归因，并核对完整576槽输入与BF16 loss。新画像不覆盖既有参考与结果；脚本在启动前核验画像入口SHA，运行期间保护训练源码、工具和recipe TOML，并检查所有4rank都触发开始/结束边界。队列和SHA见 `full576_matched_profile_coordinator_20261118.json`，当前未运行GPU画像。首更新阶段/首次kernel成本仍会包含在该诊断内，不将其声称为稳态性能证明。当前严格重启的844个受保护文件再次核对未变化。

### GC16生产完整反向通过，optimizer梯度norm collective失败

`outputs/megatron_tp_pp_default_gc16_full576_strict_restart_20261118` 的四rank实际GC记录均为schedule_completed=true、576次backward、36次精确的16步回收边界；累计回收allocated约26.68/26.99/33.42/33.42GiB，GC host累计25.50/22.47/25.15/25.08秒。随后rank0在 `ChainedOptimizer.step → get_grad_norm_fp32 → all_reduce(total_norm, SUM)` 报NCCL ncclUnhandledCudaError，底层CUDA错误2 out of memory；Adam未执行，没有成功update、输入清单或checkpoint。其他rank由torchrun终止，runner/协调器已非零退出，844文件guard未变化。失败摘要 `default_gc16_full576_grad_norm_nccl_oom_20261118.json` 保留原始GC记录hash；不将供给576槽或完整反向当作更新验收通过。

下一项原容量FP32/Flex原生参考已按既有队列实际启动，协调器1162948仍保护全部Cosmos Python源码。为给新失败定位留出后续窗口，暂时STOP没有训练子进程的DP2协调器1272836（create_time=1795034380.53）；画像协调器1378370继续等待它，当前没有GPU作业重叠。不能直接恢复旧DP2队列而忽略当前失败及新runtime资格检查。

隔离候选 `/tmp/cosmos_tp_pp_collectives_warmup_20261118/{collectives.py,driver.py,runtime.patch}` 使用实际optimizer梯度统计组，在activation峰值前对FP32 `[1]` 零标量执行SUM/MAX并同步，记录raw CUDA free、allocated/reserved及组大小。只操作独立零标量，不改变参数/梯度/状态/RNG，也不清空CUDA缓存。它用于验证首次collective初始化/缓存压力这一候选原因，当前未确认根因、未取得CUDA资格结果、未应用到默认driver。`git apply --check` 和Ruff通过；SHA/base记录于 `optimizer_collective_warmup_candidate_20261118.json`。独立入口 `tools/run_tp_pp_with_optimizer_collective_warmup.py` 可通过 `COSMOS_TP_PP_COLLECTIVES_CANDIDATE` 选择该候选，并用 `--run-path tests/megatron/tp_pp_network_smoke.py` 在完整GPU窗口做真实MCore/非零Adam/compile资格检查；当前尚未执行。默认源码须等1162948及所有子进程结束、重新核对guard后再修改，之后仍须重跑原始global576完整重启。

预热候选的四卡资格协调器1404824已实际启动并等待1162948。诊断 `tests/megatron/tp_pp_optimizer_collective_pressure_smoke.py` 在实际MCore optimizer首次get_grad_norm前制造PyTorch空闲缓存压力，cold/warm均要求原始CUDA余量约256MiB、缓存至少1GiB；cold必须在norm collective阶段出现NCCL CUDA OOM，填缓存时OOM不算复现。warm要求四rank norm成功，并在释放人工压力后继续通过原有compiled/full-AC网络的四次非零Adam/梯度/master/moments比较；再运行BF16 TP2/PP2/DP1与TP2/PP1/DP2的master/moments/step保存恢复。人工压力测试中的 `empty_cache` 只在norm已成功后用于恢复数值检查的正常环境，不在候选/生产训练中调用。Ruff及CLI检查通过，当前没有CUDA结果，也未应用默认补丁；若cold不能复现，资格摘要不会判为通过。队列见 `optimizer_collective_warmup_qualification_coordinator_20261118.json`，DP2协调器1272836继续STOP作为后续窗口占位，画像继续等待它。

执行节点于2026-11-19变更，用户确认硬件规格相同；新节点四卡之间为NV18互联，Torch/MCore/TE已按原版本恢复。旧节点作业状态仍未确认，继续保持共享训练源码不变，旧性能记录保留但新测试单独标记节点及环境。新节点人工缓存压力对照已完成：约240–246MiB raw CUDA余量下，cold四rank均在真实MCore norm collective发生CUDA OOM；提前预热实际梯度统计组后，warm四rank norm均通过，随后compiled/full-AC小模型四次非零Adam更新的梯度/master/moments检查均通过。证据为 `artifacts/megatron_tp_pp/optimizer_collective_pressure_qualification_20261119.json`。这支持首次collective初始化受缓存压力影响的机制，不替代完整global576更新、重启或性能验收；BF16保存恢复资格检查继续进行。

### 2026-11-19：FP32问题定位与TP4/FSDP4逐微批性能口径

用户要求继续解决数学误差，并将性能目标改为TP4/PP1/DP1对比原生FSDP4：对齐每个micro_step/rank的batch，不再对齐global batch或每更新local累计样本。先前超门槛的完整模型对照已经使用FP32且禁用TF32，不能归因于BF16。新节点使用隔离源码副本 `/opt/cosmos-tp-pp-work-20261119`，共享训练源码保持不变。

新计算性能入口 `tools/run_tp4_fsdp4_microbatch_benchmark.py` 在原生产训练循环中重放同一组真实prepared batch，双方均BF16、compile/full AC、3个micro_step，前2更新预热、后5更新测量。三个微批实际9/11/13样本，真实text/vision/action长度及payload SHA在四rank资格检查中完全一致；TP全局33样本/更新，FSDP全局132样本/更新。媒体读取、VAE/noising及其初始化不在本轮计算性能范围；不将此结果称为端到端数据管线性能。汇总取最慢rank、同时报告更新耗时、摊销micro_step耗时及实际吞吐。3项汇总回归检查通过，尚未取得真实性能结果。

为恢复完整FP32参考，新增原生逐block full AC入口，先安装冻结权重流式搬运hook，再包checkpoint以支持反向重计算；保布局卸载下，小模型三次非零Adam更新的输出/梯度/norm/master/moments全部精确一致。旧save_on_cpu布局负对照仍失败，未删改。逐层输出/反向梯度采样入口的CPU检查确认不改变梯度/RNG，且重计算不会重复计数。另有四GPU汇集权重诊断算子的三次非零状态更新资格检查通过；该慢速原生block对照用于隔离分片GEMM/归约，明确不能作为生产TP验收。

新实际协调器、输入合同、小资格结果与命令保存在 `outputs/megatron_tp4_fsdp4_microbatch_20261119`；协调器已启动，等待节点丢失后重新转换的完整checkpoint。随后运行原生FP32参考、新口径双方性能及连续三更新数学诊断，所有GPU作业串行。FP32门槛、参考状态和连续轨迹要求不放宽；TP2及新TP4/PP1完整数学结果仍待取得。

同micro_step的真实prepared输入计算对比已完成：两边均7次更新，排除前2次、测量后5次，TP4/PP1/DP1为33样本/更新、平均8.425880秒、摊销2.808627秒/micro_step、3.916505样本/秒；原生FSDP4为132样本/更新、平均22.824324秒、摊销7.608108秒/micro_step、5.783304样本/秒。TP整机吞吐是FSDP的0.677209，不能用较低更新延迟宣称吞吐更高。TP峰值allocated约64.678GiB。`outputs/megatron_tp4_fsdp4_microbatch_20261119/performance_comparison.json` 保留全部最慢rank时间和输入合同；本组不含VAE/noising或媒体读取，包含它们的后续原始fixture已按原生SHA恢复。两边保留各自backend梯度缓冲策略，未据此宣称BF16数学轨迹一致。

新的36层FP32原生参考已完成三次连续更新，逐层实际输出dtype均为float32，进程RSS约110GiB，full AC避免旧参考的巨大激活驻留。旧失败输入的两批raw数据也已恢复，SHA与1203原生记录一致，尚待恢复VAE/noising。新输入的TP2首步CPU校验已复现超门槛：全局梯度相对L2=2.348464350689067e-4；不能将这项工作称为正确性通过。

为缩短全参数定位周期，隔离CUDA比较副本只改变校验actual/target的设备；FP64误差运算、既定阈值、训练/优化器计算均保持不变。CPU/CUDA校验器在exact、相对误差通过/失败、近零绝对误差通过/失败五组检查中判定一致，误差指标差异不超过1e-12。缓慢CPU用例的四rank完整首步pre-clip observation保留，其未完成轨迹已明确终止；后续将用CUDA比较副本重新跑完整三步，并逐rank核对首步与CPU observation一致。代码差分为 `artifacts/megatron_tp_pp/fp32_gpu_comparison_only_20261119.patch`，当前运行的协调器为 `continue_gpu_fp32_validation_20261119.py`；不把中止的CPU轨迹算作三步已完成。


新输入的正式FP32连续三更新已完整结束，固定门槛未放宽，结果仍失败。TP2/PP2的pre-clip梯度相对L2依次为0.000234846435、0.002947948754、0.577237553349；TP4/PP1为0.000294172922、0.001437761055、0.109165457886。第三步已经是显著差异，不能解释为“FP32正确性已完成”。四rank CPU/CUDA首步校验指标在1e-12内一致、失败参数名称相同，排除了仅由校验器迁移造成的差异。详见同输出目录下 `fp32_tp{2,4}_standard_gpu.rank*.json` 和 `full_source_cpu_cuda_first_gradient_comparison.json`。

汇集全权重并执行原生block的慢速诊断有一版实现错误：参数在meta物化后丢失标记，未清除Q/K norm的partial-gradient SUM，导致这两类梯度多乘TP倍数。这版结果已在 `native_block_diagnostic_invalid_v1.json` 明确作废；它不是正式TP路径的缺陷证据。修正诊断后，前两步原始梯度与原生参考精确相等，第三步相对L2为8.03880421e-5，仍有17个rank0状态门槛失败。第二步native norm=7.792957782745361，MCore norm的舍入不同，使FP32裁剪系数差一个ULP，随后轨迹发生漂移。当前正在资格检查逐参数FP32 norm及原生登记顺序的候选，不能将其称为已修复。该候选只使用参数名顺序和实际本步梯度，不读取参考norm、参考梯度或参考状态。

原1203失败输入的两个逻辑rank已恢复raw数据、VAE/noising和packed payload，三类SHA都与原记录一致，见 `legacy_prepared_inputs/manifest.json`。另一个待测候选显式使用FP64 Linear GEMM/TP归约中间累积，参数、激活及优化器状态仍FP32；小网络四GPU四步资格通过，但尚未完成完整模型三步验证。即使后者通过，也必须标明高精度累积，不能冒称标准FP32路径已通过。

包含原生VAE/noising的第二轮TP4/FSDP4性能测试正在运行，仍对齐每micro_step/rank的9/11/13样本、双方7更新排除前2。媒体解码在本轮外；VAE启用相同的全部空间bucket AOT编译，冷启动编译仍进行中，暂时没有可报告的稳态结果。已完成的prepared计算性能结果不受影响。


2026-11-19即时集成补充：独立worktree修正CPU planning测试漏mock的 `torch.tensor(device="cuda")`，GPU完全隐藏后101项通过、1项跳过，Ruff通过。原生完整block对照使用逐参数原生FP32 norm和独立meta模型参数顺序后，四rank连续三更新的固定门槛全部通过；这仍是汇集权重的诊断，不代表正式分片TP已通过。正式FP32数值和完整576槽容量限制继续有效。

PR的GitHub基础大文件/secret检查通过；完整pre-commit在安装基线 `.pre-commit-config.yaml` 原有锁定的 `rumdl==0.1.62` 时失败（没有可安装发行包），尚未执行后续完整检查。本次按用户明确要求立即集成当前进度，不扩展修改无关Markdown、脚本权限或CI规则，也不将该CI状态记录为通过。


### 2026-11-19：追加的正确性证据和性能候选

用户进一步明确，性能完成门槛为同每rank/microstep batch下四卡整机samples/s打平FSDP4，而不是仅比较每更新秒数。当前prepared计算TP4/FSDP4吞吐比0.677209，仍未达标。含VAE/noising的首轮完整7更新已结束，TP约18.7710秒/更新、FSDP约34.2760秒/更新，吞吐比0.45650；其中FSDP窗口遭遇并发CPU测试误触CUDA初始化，已在原始JSON中明确provisional及clean_repeat_required，不作为性能验收证据。

原生完整block＋原生FP32 norm对照四rank三步全部通过，原始梯度相对L2和master/moments最大绝对误差均为0，compute/master一致，冻结参数精确；报告 `native_block_native_norm_control_summary.json`。这是复制完整block计算的诊断，不满足真正分片TP计算要求。仅加宽Linear累积的TP2新输入三步仍失败，梯度相对L2=1.16240489e-5/5.17069218e-4/0.0263086530；第一步采样的UND逐层输出已精确，但VIDEO/Action输出存在微小误差。继续保留固定门槛，不声称该候选已修复标准FP32。

性能候选 `tools/tp_pp_shared_input_reduction.py` 保留独立MCore GEMM与canonical参数，将共享输入的Q/K/V或gate/up本地dgrad先累积再做一次TP SUM。CPU覆盖全使用/部分断开/全部断开三种情况，未使用参数保持grad=None，输入梯度归约次数从活跃分支数降为1或0。真实TP2/PP1两层生产I/O、compile/full AC、12微批×四次非零Adam资格检查通过，最大逐参数梯度相对L2=7.4872e-7；不是完整模型验收。最初一次GPU资格失败于Dynamo不支持bool(tuple)，已将eligibility移至构建阶段，保留失败日志并用v2重新通过。真实TP4七更新和预热后完整更新Nsight对照等待当前数学窗口结束后串行运行。画像入口的三项边界测试通过，画像不用于稳态吞吐结论。

原始未裁剪FP32/Flex/full-AC参考也已排队，目标保留原始每个sample/token/target，再做TP4和TP2三步对照。旧原容量参考因内存压力未完成，本次利用已资格通过的full AC重试；仍未取得完整输入参考或验收结果。全部新命令和进程身份记录于 `outputs/megatron_tp4_fsdp4_microbatch_20261119`。上述原始JSON报告也位于该目录。


同输入的真实TP4 prepared计算性能已取得：共享输入归约候选前2次预热、后5次测量，平均7.913418秒/更新、4.170132样本/秒，相比原TP4的3.916505样本/秒提升约6.48%；原生FSDP4为5.783304样本/秒，候选仅达到72.11%，尚未达标。双方每microstep按rank分别处理相同9/11/13样本，global batch不同；本轮不含VAE/noising和媒体解码。原始合同及逐rank计时见 `shared_input_tp4_comparison.json`。两卡四更新和未使用梯度资格已通过，但该候选尚未通过完整模型正确性验收，默认后端没有启用它。

标准FP32三步仍失败。线性FP64中间累积诊断的TP4三步梯度相对L2为2.940852e-4/1.966250e-3/0.1288971，仍失败；逐层采样显示第二输入的VIDEO反向误差在30–32层明显放大。新增独立attention/QK norm录制入口，仅用于固定输入、定位算子及梯度，不作为正确性验收。新增Inductor通信重排及选择性AC两个隔离候选；CPU配置与录制检查共17项通过，但还没有真实GPU资格和性能结论。全原始packed输入的FP32/Flex/full AC参考正在排队，不得以裁剪样例代替。


选择性AC和通信调度重排候选现已在实际TP2/PP1生产I/O小网络通过compile/full-AC四次非零Adam更新；调度候选使用各rank独立全新缓存，两个rank均观察到含collective的scheduler调用，逐参数梯度最大相对L2约8.05e-7。它们仍是opt-in，尚未通过完整模型数学和性能验收。首次调度资格测试因向 `torch.compile(options=...)` 传入不受支持的缓存选项失败；改为Torch支持的环境变量后用全新目录重试，失败记录保留。组合真实TP4七更新将共享输入归约与通信重排叠加，使用每rank新缓存但保持正常稳态缓存策略，等待原生参考释放全部四卡。

原始容量FP32验证入口的初始PP fallback布局此前沿用裁剪用的12/6 token占位。现有训练实际通过metadata传输动态layout；隔离数学验证副本的fallback已修成原始配置11776/59648/2944，实际短UND仍以metadata为准。可用 `tools/build_tp_pp_original_capacity_helper.py` 基于原始验证脚本生成隔离副本；本机变更补丁与SHA保存在 `artifacts/megatron_tp_pp/`。这不表示已在共享训练核心部署或完成原始容量验收。


### 2026-11-19：通信重排实测和原始容量参考

在共享输入梯度归约候选上进一步启用 Inductor 计算/通信重排，同一 prepared 输入的真实 TP4/PP1 四卡测试完成 7 次更新，前 2 次预热、后 5 次计时；每 rank 每 microstep 分别为 9、11、13 个样本。全部 4 个 rank 均实际观察到带 collective 的调度器调用。TP4 后 5 次慢 rank 耗时为 7.868624、7.855427、7.866675、7.886395、7.918960 秒，平均 **7.879216 秒/更新、4.188234 样本/秒**。同合同 FSDP4 为 **22.824324 秒/更新、5.783304 样本/秒**，四卡整机吞吐比 **0.724194**；由于每次 TP 更新处理 33 个全局样本、FSDP 更新处理 132 个，不能只比较更新耗时。相较仅共享输入归约的 4.170132 样本/秒，此组合只增加约 0.43%。本实验使用 BF16，排除媒体加载、VAE/noising 与 checkpoint 写入；不是完整模型数值验收。逐 rank 调度观察和原始比较分别见运行目录的 `overlap_shared_tp4_reports/` 和 `overlap_shared_tp4_comparison.json`。

原始容量 FP32/Flex/full-AC 的原生参考在首次前向 VIDEO MLP、完成第 29 层附近遇到 CUDA OOM（需额外 2.73 GiB，当时设备约剩 2.29 GiB），未完成任何更新。按 token 分块 MLP 的实验入口仅用于降低参考峰值显存，已通过 CPU 输出和梯度检查，但小模型首次 GPU 配置未实际触发分块，不能当作分块路径的资格证据；完整容量、标准 FP32 数值门槛与性能打平仍未通过。


### 2026-11-19：选择性重算和小模型 FP32 候选

在共享输入梯度归约基础上，最后 4 层使用选择性 activation checkpointing、保存规则为 `^mm\.default$`，其余层维持 full AC。真实 TP4 prepared 数据完成 7 次更新、前 2 次预热，后 5 次慢 rank 平均 **7.823829 秒/更新、4.217883 样本/秒**，为同输入 FSDP4 的 **72.93%**；相对仅共享输入归约的 4.170132 样本/秒提高约 1.15%。rank 0 峰值 PyTorch 分配 76,249,869,824 bytes，后者为 68,435,028,992 bytes。该 TP 候选对最后 4 层使用选择性重算，而 FSDP 对照仍使用 full AC，因此 72.93% 是单侧优化探索，不是重算策略一致的最终公平比较。两者均不含媒体加载、VAE/noising 或 checkpoint 写入；此轮尚未通过 GPU 画像验证实际保存的算子及重算次数，不是完整模型数值验收。完整合同比较和各 rank 日志在运行目录的 `selective4_shared_tp4_comparison.json` 与 `selective4_shared_tp4/`。

修复资格入口的 CUDA 初始化顺序后，隔离的原生 FP32 范数候选在 TP2/PP1 小模型完成四次非零更新：两个 rank 的参数和预测均为 FP32，四次均调用候选范数，顺序通过 meta 模型独立派生。首次资格尝试在模型计算前因导入时 CUDA 已初始化而退出，不能作为算法数值失败；重跑报告为 `native_norm_driver_tiny_v2*.json`。这不是 36 层或 TP2/PP2 的验证，原生范数只是一项候选，标准 FP32 连续轨迹仍未通过。真正触发 token 分块的 TP2/PP1 小模型四次更新也通过，最大逐参数梯度相对 L2 约 **6.80e-7**，完整容量仍待验证。


### 2026-11-19：本地投影融合试验

独立的 QKV、gate/up 及两者组合在 TP2/PP1 小模型上分别完成 compile/full AC 的四次非零 Adam 更新，各 rank 对应启用报告均有效，最大逐参数梯度相对 L2 约 **7.49e-7**。组合候选的真实 TP4/PP1 prepared 输入七更新测试，前二预热后五次平均 **7.872503 秒/更新、4.191805 样本/秒**，为同输入 FSDP4 的 **72.48%**；仅共享输入归约为 4.170132 样本/秒。rank 0 峰值 PyTorch 分配为 69,447,804,416 bytes；每个 rank 报告 108 组 QKV 和 108 组 gate/up 已安装。安装报告并不证明 GPU kernel 实际融合，需 profiler 验证；小模型结果不构成完整 36 层数值验收。完整合同、更新记录和报告在运行目录的 `fused_input_both_tp4_comparison.json` 与 `fused_input_both_tp4_reports/`。


### 2026-11-19：实验性原生 FP32 梯度范数

`COSMOS_TP_PP_NATIVE_FP32_NORM=1` 显式启用逐参数原生顺序 FP32 norm 候选，默认关闭；开启只支持已验证的 FP32、物理 DP1、正且有限的梯度裁剪和分布式 Adam 配置，不支持独立 norm group 或其他梯度变换。所有 rank 在启动时执行一次 world 协商，即使默认关闭也有一次启动 collective；之后默认训练仍使用原 norm 路径。顺序从新建的 meta 模型及 canonical 参数身份派生，不读取参考梯度或范数；各 TP 组及 PP owner 在重建梯度前做布局检查。

隔离环境中，修订后候选通过 TP2/PP1 两卡和 TP2/PP2 四卡小模型各四次更新，默认关闭 TP2/PP1 冒烟也通过；另有 37 项 CPU 测试和两进程标志协商检查。候选按参数逐项通信，可能增加显存及耗时；单 rank OOM 等错误仍可能使同伴等待。它不解决已观测的全部分片计算误差，尚未通过真实 36 层、原始容量或完整轨迹验收，也未用于 BF16 性能对比。

### 2026-11-19：包含 VAE 的吞吐和 FP32 投影误差

原始输入回放的 BF16 TP4/PP1 与 FSDP4 各完成 7 次更新，前 2 次预热、后 5 次计时；双方每 rank 的三个 microstep 分别处理 9、11、13 个样本，分别为每更新 33 和 132 个全局样本。计时包含输入副本、H2D、VAE/noising、训练和 optimizer，不包含媒体解码与 checkpoint 写入。TP4 平均 **18.794247 秒/更新、1.755856 样本/秒**，FSDP4 平均 **34.183384 秒/更新、3.861525 样本/秒**，整机吞吐比 **45.47%**，没有达到打平目标。该结果暂定：GPU PID 监控处于 restricted 模式，无法排除短暂外部干扰；此外一个仅 CPU 的 pytest/Gloo 测试与 TP 最后一次计时更新重叠，其影响未知。原始比较、逐步耗时和运行合同见 `outputs/megatron_tp4_fsdp4_microbatch_20261119/raw_clean_after_attention_20261119/comparison.json`，重叠证据见 `full_vjp_cpu_test_overlap_20261119.json`。这不是数值验收或严格独占的性能验收。

标准 FP32、关闭 TF32 的真实 36 层裁剪输入三次连续更新仍未通过固定的 1e-4 梯度相对 L2 门槛：TP4 分别为 **2.9417e-4、1.43776e-3、1.09165e-1**，TP2 分别为 **2.3485e-4、2.94795e-3、5.7724e-1**。第 0 层 VIDEO K 投影的相同捕获输入及 canonical 主权重，在 H100 上分别按原生完整宽度和 TP 分片宽度重放，均逐位复现各自的捕获输出；两种输出相对 L2 为 **2.6011e-7**、最大绝对差 **1.4901e-7**。运行时参数字节未独立捕获，因此该对照只说明矩阵乘宽度及布局产生的舍入可以复现最早捕获差异，并不能证明它是后续漂移的唯一原因。证据见 `outputs/megatron_tp4_fsdp4_microbatch_20261119/gemm_after_raw_20261119T094130_536321/k_projection_replay.json`。K/V 完整宽度前向候选在真实 TP2 小模型四次更新通过，但其反向仍采用分片矩阵乘和归约；完整宽度反向及真正 36 层轨迹仍待验证。

原始容量原生 FP32 参考在平坦 checkpoint 输入和 4096 token MLP 分块后完成 36 层前向，但第一次反向 VIDEO MLP 仍因显存不足失败，没有完成更新。流式 CPU 梯度候选已通过小模型 CUDA 原生 norm/clip/FusedAdam 对照，以及 811 个真实参数形状的合成 norm/clip 对照；真实 36 层连续更新参考仍在验证。这些诊断不能替代原始容量的完整数值验收。

### 2026-11-19：流式参考与全宽 K/V 实测

36 层原生 FP32 的裁剪输入、每次更新两微批、连续三步对照中，流式梯度参考与未流式参考的输出、梯度、master、Adam 一阶矩、二阶矩及 step 均逐位一致：共检查 **2433 个参数文件、12165 个张量字段**，三个更新的输出均通过，无不一致项。报告为 `outputs/megatron_tp4_fsdp4_microbatch_20261119/gemm_after_raw_20261119T094130_536321/streamed_vs_native.json`。这验证了流式原生参考在该输入下的行为，不是分片 TP 或原始容量验收。

四卡固定捕获输入及共同原生上游梯度的 K/V 投影受控回放中，全宽前向、输入梯度和权重梯度在四个 rank 均逐位匹配原生；实际 TP 捕获的上游梯度与原生并不相同，因此不能将共同梯度的受控结果当作实际轨迹相等。TP2 小模型 full-AC 四次更新也通过，最大逐参数梯度相对 L2 为 **5.934e-7**。报告为 `full_vjp_after_streamed_20261119T100421_568913/summary.json`。

真正的 TP4 36 层裁剪输入连续三步应用全宽 K/V 前向及反向后，梯度相对 L2 为 **4.31973e-4、2.65568e-3、1.69782e-1**，三步均超过原有 1e-4 门槛，且大于标准 TP4 的对应数值；该候选没有修复模型轨迹。四 rank 的完整输出、状态比较和源码记录在 `full_vjp_real36_20261119T100656_574178/`；接下来应按共同输入及上游梯度继续定位其他算子，不降低门槛。目录均位于 `outputs/megatron_tp4_fsdp4_microbatch_20261119/`。

### 2026-11-19：原始容量参考完成及性能合同补充

消除逐层 FP64 trace、使用平坦 checkpoint、逐块嵌套重算和流式梯度后，原生 FP32/Flex 原始容量参考已完成两份真实微批的连续 **3 次更新**，梯度范数分别为 **0.29377660、1.10465622、0.42172375**；参考 manifest 标记完成，包含 811 个参数及全部三步状态。同输入的 TP4 对照已启动，其数值门槛与 TP2 对照尚无结果；这不是 TP 通过，也不是完整 576 槽验收。过程和输出位于 `outputs/megatron_tp4_fsdp4_microbatch_20261119/original_capacity_nested_20261119T1019/`。

上文包含 VAE 的 45.47% 结果还存在配置不一致：TP 实际记录 `cudnn.deterministic=false`、`cudnn.benchmark=true`，FSDP 参考强制为 `true`、`false`；VAE 属于计时范围，二者可能选择不同卷积算法。两侧 TF32 关闭，均使用 `CUDA_DEVICE_MAX_CONNECTIONS=1`，但运行时 `torch.are_deterministic_algorithms_enabled()` 未记录。FSDP 各 rank 重复同一三份 fixture 和初始 RNG，因此每更新 132 是样本处理次数，不是 132 个不同样本；该基准不证明真实数据连续性，旧结果也没有逐次 prepared payload/RNG 哈希。旧结果除 GPU PID 监控与 CPU 重叠限制外，还应因这些合同差异标记为暂定，不用于最终性能验收。同设置、实际运行时取值及匹配输入/状态的重测仍待执行。

### 2026-11-19：TP 参考中止与报告核对

上述原始容量 TP4 对照在首次更新完成前由主机内存余量保护中止；协调器记录的原因是 `conservative memory headroom guard`，不能将其描述为 CUDA OOM。原生参考成功不代表 TP 对照已通过，TP2 对照也尚无结果。协调器记录在 `outputs/megatron_tp4_fsdp4_microbatch_20261119/original_capacity_nested_20261119T1019/coordinator_result.json`。

裁剪输入的 TP4 全宽 K/V 诊断中，将每次更新开始的 master、Adam 状态和 step 重置为对应原生状态后，三次整体梯度相对 L2 为 **4.31973e-4、2.55243e-5、8.99880e-5**；逐项比较仍有大量失败，不能视为连续训练通过。与不重置状态的结果对照表明轨迹会放大误差，但不证明具体根因。运行记录在 `outputs/megatron_tp4_fsdp4_microbatch_20261119/replay_native_state_after_vae_20261119T1038/queue_result.json`。

早期裁剪 FP32 报告 `fp32_tp4_standard_gpu.rank*.json` 中 `pipeline_parallel_size=2` 是报告字段硬编码错误：执行记录 `sequence_events.jsonl` 显示四卡 `--tp 4`，该次执行所用隔离测试脚本在 TP4 时实际构造 PP1。原始报告保留，不能凭错误字段推断运行拓扑，也不能将数值失败改写为通过。后续验证应从实际拓扑生成并核验报告字段。

此外，原容量 TP 验证协调器的报告门槛此前仅信任顶层 `passed`：独立 CPU 反例表明，包含逐项失败和不完整覆盖的矛盾报告可能被错误接受。实际报告生成器会汇总失败，反例并不证明已有真实运行误报。诊断 helper 现从实际拓扑记录 PP 大小。

原始容量三更新诊断现要求对四个 rank、三次更新逐项核对 trainable 参数、梯度、master、Adam 矩和 step、compute/master、梯度 norm 及输出样本，并按参考平方和重算相对或近零绝对误差门槛。新增诊断字段仅用于数学对照，未纳入性能测量；缺少字段的旧报告不能通过新的严格 gate，也不回填历史报告。新增 gate 本身不代表 TP 正确性已通过，完整 576 槽验收仍需单独完成。

新的 TP 数学诊断会逐更新读取实际 trainable 参数、梯度、master、Adam 浮点状态的 dtype，并在每个 decoder 层边界采样三个输出流的 dtype、autocast 和 TF32 开关；该采集不进入性能训练路径。层边界检查不证明融合 kernel 的内部累加精度。此前已经生成的原生参考没有完整的运行时 dtype 与 TF32 实测记录，不能将 TP 侧记录推断为原生侧也通过了 FP32 检查。

### 2026-11-19：两侧重算策略对照及测试重叠

四卡 BF16、prepared 输入、每 rank 每微步 9/11/13 样本、每次更新三个微步，四个配置各完成七次更新，前两次预热、后五次计时。同为 full AC 时 TP4 为 **7.902223 秒/更新、4.176040 样本/秒**，FSDP4 为 **22.866170 秒/更新、5.772720 样本/秒**，整机吞吐比 **72.34%**。两侧均对 36 层中的 8 层采用 FA3 选择性重算时，TP4 为 **7.847071 秒/更新、4.205391 样本/秒**，FSDP4 为 **22.635285 秒/更新、5.831603 样本/秒**，吞吐比 **72.11%**。计数是样本处理次数，FSDP 四个 rank 重复使用 fixture；比较不包含 VAE/noising、媒体解码或 checkpoint 写入，也不代表完整模型数值通过。实验记录在 `outputs/megatron_tp4_fsdp4_microbatch_20261119/flash3_ac_queue_20261119T1040/sequence/`。

这两组结果均为探索性测量：两侧 cuDNN 运行设置并不相同，prepared 输入范围内的影响未知；GPU PID 监控受命名空间限制，不能保证严格独占。FSDP selective 计时阶段附近于 13:52:02 UTC 发生了 CPU 测试，测试未使用 GPU，但其对耗时的影响未知，另有更早 CPU 工作的精确重叠待核查。因此 selective 结果不能用于无干扰的最终性能验收。测试时间及限制见 `outputs/megatron_tp4_fsdp4_microbatch_20261119/flash3_ac_queue_20261119T1040/cpu_test_overlap_20261119.json`，新的对齐设置和完整时窗重测仍待完成。

### 2026-11-19：FA3 profiler 与固定输入 SDPA 回放

上述 prepared 输入四组 profiler 已完成：TP4 与 FSDP4 的 full AC 均为每卡 **216 次** FA3 前向 kernel，两侧选择性重算均为每卡 **192 次**。TP4 的配对计时从 4.176040 增至 4.205391 样本/秒，约 **0.7%**；单轮测量不能排除波动，也不代表完整模型数值已通过。完整结果见 `outputs/megatron_tp4_fsdp4_microbatch_20261119/flash3_ac_queue_20261119T1040/sequence/summary.json`；其中资格结果 `passed` 不表示性能或数值验收通过。

H100 固定捕获输入的 math SDPA 回放覆盖两个微批、两个 attention 调用及四个 rank：Q/K/V 与 mask 相同，完整 heads 和 TP 局部 heads 各自的前向都逐位复现对应原捕获，但两种布局间前向最大相对 L2 为 **3.5758e-7**。对共同原生上游梯度的反向，dq 最大相对 L2 为 **2.5358e-6**，dk/dv 均约 **1.60e-6**。这证实了分片计算中一个微小差异来源，并不证明它单独造成 36 层训练漂移；原 storage offset 没有独立捕获。报告为 `outputs/megatron_tp4_fsdp4_microbatch_20261119/sdpa_captured_replay_20261119T1045/sdpa_replay.json`。
