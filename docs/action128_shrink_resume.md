# Action128 缩卡后继续数据进度

`shared_prefetch` 的 EpisodeBlockDistributor 可以通过离线迁移 checkpoint
改变 DP 卡数，同时保留原逻辑分片、各分片已提交的 `(epoch, index)` 及其
CPU RNG。它不按新 DP 大小重新计算原始样本分区。

例如 192 → 160：每个数据源仍保留原来的 192 份逻辑分片，32 个新 rank
各接管两份，128 个各接管一份。新 rank 在接管的分片之间轮转。
一份原逻辑分片只分配给一个新 rank；撤下 rank 的未消费数据也会被保留。
原分片的游标和 shuffle seed 不变，所以同一原 epoch 中已消费的位置不会重放。
后续正常进入新 epoch 时，仍按原有规则重新遍历数据集。

## 准备 checkpoint

仅处理可信的本框架 checkpoint（会反序列化 pickle）。迁移要求 CP/TP/PP=1、
每个原 global rank 对应唯一 DP rank，所有 rank 文件齐全，且保持数据源、
处理配置、shuffle seed 和指纹不变。运行时只允许 EpisodeBlockDistributor；
其他 distributor 会拒绝这种恢复。

先确保停机前最后一个已完成 step 的数据状态已经保存到 `.save_complete`
checkpoint，再退出原训练。不要直接 kill 掉仍未保存的数据进度后宣称无重复：
旧 checkpoint 不包含其后的已消费样本，也不包含其后的模型更新。

以下命令应在项目根目录、训练虚拟环境中运行。未指定 `--output` 时只核对状态：

```bash
python -m tools.reshard_action128_checkpoint \
    --checkpoint /absolute/path/to/iter_000003000 \
    --target-dp-size 160
```

生成独立 checkpoint，输出目录必须不存在：

```bash
python -m tools.reshard_action128_checkpoint \
    --checkpoint /absolute/path/to/final_saved_checkpoint \
    --target-dp-size 160 \
    --output "$PWD/outputs/shrink160/checkpoints/iter_000003000"
```

输出名中的 iteration 应与实际源 checkpoint 对齐。工具读取原 dataloader
pickle，将原分片重新分配，写入新的 rank pickle 和 `reshard_manifest.json`，
最后写保存完成标记。模型、优化器、scheduler、trainer 文件通过同一文件系统
的硬链接保留，不反序列化或复制大型参数；如果硬链接不可用则失败并清理临时目录。
原 checkpoint 不变，原目录被正常 retention 删除后硬链接仍然有效。
不要原地修改任一 checkpoint 的参数文件。

## 160 卡启动

在重新编号的 20 个节点上设置一致的 `NNODES=20 NPROC_PER_NODE=8 DP_SHARD=16`，
`NODE_RANK` 为连续的 0–19。使用包含虚拟分片恢复代码的新启动快照，
`BASE_CHECKPOINT_PATH` 指向迁移目录，`OUTPUT_ROOT` 使用新目录，避免原输出目录
的自动恢复指针抢先选择未经迁移的 192-rank checkpoint。

保持原来 58 个源及处理配置，用生产入口显式设置完整状态恢复：

```bash
export NNODES=20 NPROC_PER_NODE=8 DP_SHARD=16
export BASE_CHECKPOINT_PATH="$PWD/outputs/shrink160/checkpoints/iter_000003000"
export OUTPUT_ROOT="$PWD/outputs/pretrain58src_state_h1_nano_160gpu"
bash examples/launch_pretrain_state_h1_58src_192gpu.sh \
    checkpoint.action128_load_mode=auto
```

其他环境、资产、凭据和 rendezvous 按原生产命令设置。
这是数据恢复的启动配置，不代表完整 160-GPU 模型/优化器分布式恢复已经验证。
卡数变化会改变全局 batch、分片交织次序和训练轨迹；保持 GA=3 时每步总 pack 数
从 576 变为 480。它不保证与 192 卡逐步数值一致。

迁移后的 checkpoint 会继续保存虚拟分片及原分片 RNG seed；后续同规模恢复
和再次迁移都保留这些地址，不能删除 execution state 中的虚拟分片字段。
物理 CPU worker 数仍由配置控制，不会为每个逻辑分片创建独立进程。

## 144 卡生产恢复（2026-11-02）

18 个节点使用 `NNODES=18 NPROC_PER_NODE=8 DP_SHARD=16`、GA=3，
DP replicate=9；每步合计 432 个 pack。生产入口仍为
`examples/launch_pretrain_state_h1_58src_192gpu.sh`，节点数由环境变量覆盖。

本次从 160 卡的 step3800 checkpoint 迁移到
`outputs/prepared_checkpoints/pretrain58src_step3800_dp144/iter_000003800`，
输出为 `outputs/pretrain58src_state_h1_nano_144gpu_ga3_from3800_run01`。
旧实验 step3801–3974 未写入恢复点，这部分样本可能再次消费；缩卡改变
全局 batch 和分片交织，不能用新实验走到同一步号判断旧尾段数据已全部追平。
已保存的 step3800 及之前的原逻辑分片游标继续保留。
缩卡不保留旧实验的全局 batch 大小或数值轨迹。

迁移核对了 11,136 份原 source/shard 状态，644 个模型、优化器、scheduler、
trainer 文件通过硬链接保留。144 rank 已成功恢复全部五个 checkpoint 组件，
outer data global_id=7800；初始 10 个 microbatch ×144 rank 共 1,440 条
样本序列均与原逻辑分片的预期续接一致。

审计和启动记录位于输出目录的 `logs/launch_20261101T161145Z/`，
其中 `monitor/initial_data_resume_verification.json` 保存运行时续接检查。
训练使用节点内固定代码快照，修改共享仓库不会热更新现有训练进程。

本轮另设只读 step4000 checkpoint 检查，状态位于上述启动目录的
`monitor/checkpoint4000_audit_status.json`。它先等待 `.save_complete`，再检查
四个 DCP 组件的文件引用与字节范围、144 份 dataloader 状态、global_id=8400、
全部 11,136 份原逻辑分片的归属/种子，以及相对 step3800 单调前进的提交游标
和已有 RNG 状态的保留。完成结果写入 `monitor/checkpoint4000_audit.json`。
`waiting_for_checkpoint` 仅表示尚未看到保存完成标记，不是验证通过；该检查
也不读取完整模型/优化器 tensor 内容或执行真实分布式恢复。

2026-11-02 03:10:06（北京时间），`checkpoints/iter_000004000` 已完成保存，
03:11 的只读检查通过。data global_id=8400，144 rank 文件齐全；
11,136 份原逻辑分片的地址/种子保持一致，8,256 份提交游标前进，其余未回退，
10,752 份此前已有的 RNG 状态仍存在。四个 DCP 组件的全部存储引用均在
对应文件范围内。同目录 `latest_checkpoint.txt` 已指向 `iter_000004000`。

2026-11-02 04:55（北京时间）另启动 step4200 的只读等待/检查进程，脚本为
上述启动目录的 `monitor/audit_checkpoint4200.py`。保存完成后，它相对 step4000
检查全部144 rank 的 global_id=9000、fingerprint/schedule、原逻辑分片归属、
种子、游标单调性和 RNG 保留，并检查 DCP 存储引用。状态写入
`monitor/checkpoint4200_audit_status.json`；只有通过后才更新
`monitor/latest_recovery_point.json`。等待状态不代表 checkpoint 已保存或验证通过。

2026-11-02 05:35:19（北京时间），step4200 保存完成；05:35:37 只读检查通过。
全部144 rank 的 global_id=9000，58 源的 fingerprint/schedule 与 step4000
一致，11,136 份原逻辑分片的归属/种子保留，9,312 份提交游标前进、其余未回退，
此前10,752 份非空 RNG 状态全部保留。四个 DCP 组件的文件引用/字节范围通过，
最新指针及 `monitor/latest_recovery_point.json` 均已更新为 `iter_000004200`。
此次同样未读取完整模型/优化器 tensor 内容，也未实际执行分布式重载。

2026-11-02 08:16:29（北京时间），step4400 保存完成；08:17:12 只读检查通过。
全部144 rank 的 global_id=9600，58 源 fingerprint/schedule 与 step4200 一致，
11,136 份原逻辑分片的归属/种子保留，9,120 份提交游标前进，其余未回退；
已有10,752 份非空 RNG 状态保留。四个 DCP 组件的存储引用/文件范围通过，
最新指针和 `monitor/latest_recovery_point.json` 已更新为 `iter_000004400`。
同样未读取完整模型/优化器 tensor 内容或执行分布式重载。

随后启动 step4600 的只读等待检查（`monitor/audit_checkpoint4600.py`），
基准为 step4400/global9600，目标为 step4600/global10200；只有检查通过才
更新恢复记录。六小时观察预算到期仅表示观察结束，不代表训练失败。

同一输出目录重启时，这个最新指针优先于初始 `checkpoint.load_path`；
见 `cosmos_framework/checkpoint/dcp.py:784`。实际分布式重新加载尚未执行。

## 验证范围

`cosmos_framework/data/test/reshard_test.py` 的 7 个用例覆盖缩卡后的逐分片
样本序列、跨 epoch 游标、不重放已消费位置、同规模再次恢复、再次缩卡、
真实 CPU worker/prefetch、sequential packing 和随机增强 RNG，以及原 checkpoint
被 retention 删除后的硬链接有效性。已有 shared-prefetch 恢复与拓扑检查另有
7 个相关用例通过。

step3000 的真实 192-rank checkpoint 已完成 160-rank 预备转换：58 个源共
11,136 份原逻辑分片的地址、游标、RNG 字节摘要及原 worker seed 均逐项相同。
这是元数据迁移与数据层行为验证，不包含真实 160-GPU 分布式模型/优化器恢复。

## 顺序不变的媒体版本迁移

EgoDex 与 Lightwheel 的生产 catalog 已切换到 `*_media1024_g16.parquet`。
更换 LeRobot 根目录会改变数据指纹，因此不能直接用旧 checkpoint 的指纹恢复。
使用 `tools.migrate_action128_media_checkpoint` 显式准备独立的媒体迁移 checkpoint：

```bash
python -m tools.migrate_action128_media_checkpoint \
    --checkpoint /path/to/resharded/iter_000003000 \
    --source-recipe /path/to/recipe_snapshot_before_catalog_change.toml \
    --catalog-pair /path/to/old_egodex.parquet /path/to/egodex_media1024_g16.parquet \
    --catalog-pair /path/to/old_lightwheel.parquet /path/to/lightwheel_media1024_g16.parquet \
    --output /path/to/media_migrated/iter_000003000
```

工具核对完整训练 episode 顺序、窗口 counts/ends、动作时钟与行范围、
domain、相机身份/顺序、caption 引用、State 与模型归一化/投影契约。
旧数据实际指纹必须与 checkpoint 一致。通过后只更新明确列出的源指纹，
保存 old/new 路径及审计 manifest；所有数据游标、虚拟分片和 RNG 保持不变。
运行时仍执行严格指纹检查，不提供全局忽略指纹的开关。

视频编码、分辨率和媒体时间映射允许变化；全部动作 Parquet 数值等价由发布者声明，
本工具不扫描全量动作值或视频像素。packing 的批次边界可能随视频 token 数变化，
所以保证的是保留消费位置，不是逐 batch 或逐步数值相同。
新版本恢复后产生的 checkpoint 会自然保存新指纹，后续无需再次执行同一迁移。
