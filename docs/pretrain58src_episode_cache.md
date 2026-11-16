# 58 源 episode 初始化共享缓存

## 缓存内容

每个数据源、源文件版本和窗口 stride 对应一个 Arrow IPC 文件。只保留
`data_role == training && training_eligible` 的 episode，保持原始文件排序和行顺序。

保存三类确定性结果：

1. episode 元数据：episode_index、action_domain、action_space、embodiment_name、
   data_role、training_eligible、split、duration_seconds、fps、length、data_path、
   dataset_from_index/to_index、caption_path/index、video_views、view_mapping、source_times_path。
   其中视频、字幕和时间轴字段都是路径或定位信息，不是媒体正文。
2. 窗口索引：每个 episode 的窗口数 `ceil(duration / stride - 1e-9)` 和累计结束位置。
   全局 sample index 到 episode/window 的映射因此保持不变。
3. 校验摘要：action_space/domain 集合、domain 与 embodiment 的配对、是否有原生时间轴、
   视角的 storage 类型，避免每次扫描列并转换成 Python 对象。

不保存视频帧、压缩视频、action/state 行数据、字幕正文、未使用的 episode stats 数组，
也不保存模型、训练进度、随机数状态或 dataloader 游标。全局 registry、normalization、
State128 和媒体发布契约仍从源数据读取并校验，缓存不能绕过这些检查。

## Resume 语义

缓存独立于 job name、output root、节点、DP rank、checkpoint 和训练步数。
checkpoint 仍负责恢复数据游标、混合源调度和随机数状态；缓存负责恢复相同的 episode 表与窗口索引。
缓存路径、是否命中缓存都不进入 dataset fingerprint，原有 fingerprint 的计算语义不变。
因此启用缓存前保存的 checkpoint 可以在启用缓存后恢复，之后的重启也复用同一份缓存。

key 包含缓存格式版本、原始 root 路径、排序后的元数据文件路径/大小/mtime_ns、stride。
命中时仍枚举并 stat 原始元数据文件，以检测新增、删除、改名或内容版本变化；不再逐个打开和解码 Parquet。
校验沿用原有 fingerprint 的 size/mtime 策略，不额外做全文件内容哈希。
数据路径的别名改变也会改变 key，与既有 fingerprint 行为一致。

共享目录通过 flock 串行化同一 key 的构建，临时文件写完后原子替换为正式文件。
写入按最多 65,536 行分批合并，避免一次 combine 全源触及 Arrow 的 32-bit 嵌套偏移上限。
构建中源文件变化会拒绝发布；不可解析的缓存会重建；缓存目录不可写等 I/O 错误会退回原始读取。
已发布文件保持不可变，不能原地 truncate/覆盖正在映射的缓存文件。
旧版本缓存不会自动清理，清理应在没有读取进程时进行。

## 使用

`examples/launch_pretrain_state_h1_58src_192gpu.sh` 默认导出：

```bash
export ACTION128_EPISODE_CACHE_DIR=/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1
```

这是共享混闪目录，不依赖本机 root。环境变量也会传给节点本机暂存的训练代码。
可显式设置其他共享目录；设为空字符串禁用缓存。
首次缺失时自动构建，后续启动/resume 命中；日志显示 `Action128 episode cache hit/built`。
无需为了生成缓存启动训练，可在一台机器上预热：

```bash
export LD_LIBRARY_PATH=''
python -m cosmos_framework.tools.build_episode_cache \
  --recipe examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml \
  --cache-dir /share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1 \
  --workers 4
```

## 本地验证

所有验证均在本机进行，没有连接线上节点、启动或重启训练。
记录在 `outputs/episode_cache_validation_20261031/`。

`legacy_parity.json` 对照修改前的完整 reader，而非只对照新实现的两个分支。
EgoPro：5,338 个原始元数据文件、170,476 个训练 episode、3,680,827 个窗口。
旧初始化 33.743 秒；缓存命中初始化 0.097 / 0.117 秒，约 289–348 倍。
两者完整 episode 表、窗口 counts/ends 和 fingerprint 相同。
缓存约 987.2 MiB。

同一源开启 `use_state=True` 也与旧实现完整一致（`legacy_state_parity.json`）：
旧初始化 33.004 秒；命中 0.094 / 0.121 秒，约 273–351 倍。

本地同时启动 8 个读取进程（`local8_hits.json`），均命中缓存、fingerprint 及首/中/末窗口标识一致；
每进程初始化并读取这三个窗口标识耗时 2.58–2.72 秒，整个进程池 wall time 2.80 秒。
这仍只是单机并发测试，不代表 24 节点的启动耗时。

这里测量的是本地 metadata/dataset 初始化，包含源文件列表和 stat 检查。
缓存采用内存映射，后续访问可能触发页面读取；完整表的一致性比较另行执行，不计入初始化耗时。
源文件及系统缓存已被预热，不能解读为冷存储速度、192 ranks 并发性能或线上总启动加速倍数。
缓存也不会消除后续视频解码、action/state 正文读取、模型/checkpoint 加载等耗时。

测试覆盖：无缓存保存游标后启用缓存恢复、缓存到缓存恢复、真实 Joint dataloader checkpoint
校验与后续样本/RNG 一致性、2 ranks × 2 workers 的生产 distributor 游标协议、命中不读原始
Parquet、文件变更/增删和 stride 失效、缓存损坏重建、多构建者竞争、写入失败回退、
构建中源文件变更拒绝发布，以及命中后仍执行 registry 校验。

本地相关测试共 9 项通过（6 项 episode_cache + 3 项 episode_metadata_cache），
最终统一重跑结果见 `tests_all.txt`；代码 lint/format 和启动脚本语法检查通过。

全量本地预热实际触发了旧整表 combine 的 `offset overflow while concatenating arrays`
（`build.stderr`）。改为分批写入后 EgoLive 的 915,403 行已成功发布为 4.74 GB 的缓存
（`egolive_bounded.jsonl`）。这是首次构建实现的问题，与 checkpoint 游标无关。

## 全量本地预热结果

最终 58 个源全部发布，再次读取 **58/58 命中**，见 `summary.json`、`build_final.jsonl`
和 `warm_hits.jsonl`。共覆盖 76,851 个原始 metadata 文件、7,539,751 个训练 episode；
缓存合计 16,920,448,620 bytes，约 **15.76 GiB**。

本地 4 源并行复读时，最大源 EgoStandard、EgoLive 分别耗时 **11.473 / 12.365 秒**；
EgoPro 此轮为 3.495 秒。这个结果与单源单进程的 0.1 秒不同，受并发、源规模和文件/页面缓存状态影响。
所有数字都包含严格源文件检查，不能用单源最快命中结果推断全部源耗时。
预热、复读和测试进程均已结束，没有修改或测试线上训练进程。

## 同口径 58 源串行初始化对照

为回答“全部数据初始化从多久降到多久”，使用 `benchmark_total58.py` 分别在独立本地进程中运行
修改前的 reader 快照和缓存 reader。两者使用线上 TOML 的 58 个 catalog、相同的全局
Action128 registry、State128、30fps cap 和窗口参数，并保留全部 dataset 对象。

计时包含读取 catalog、构建全局 spec 和串行构造全部 dataset；不包含 Python/Torch 导入、
模型/optimizer/checkpoint 加载、worker 启动、首批媒体解码、分布式 barrier 或 compile。
额外的一致性验证单独记录，不混进初始化计时。
缓存轮逐源校验 fingerprint、episode/window 数、完整 counts/ends 哈希，以及首/中/末窗口标识。
这仍是单机单进程的数据初始化对照，不能直接替代线上 24 节点端到端启动时间。
结果记录在 `total58_old.json` 和 `total58_cache.json`。


实测结果（2026-10-31，本地单进程串行）：

| 范围 | 修改前 | 缓存命中 |
| --- | ---: | ---: |
| catalog 与全局 spec 准备 | 6.943 秒 | 2.058 秒 |
| 全部 58 源 dataset 构造 | 636.448 秒 | 12.529 秒 |
| **数据初始化合计** | **643.391 秒（10 分 43 秒）** | **14.587 秒** |

合计提速 **44.1 倍**，耗时减少 **97.73%**，节省约 **628.8 秒**。
58 源的 fingerprint、episode/window 数、完整 counts/ends 哈希和首/中/末窗口标识全部一致。
包含额外验证的实际 wall time 分别为 643.684 / 16.890 秒。

这次未清空系统缓存；结果反映本地当时的文件/页面缓存状态。两个独立进程先运行旧实现，再运行缓存实现。
这是全部 **数据初始化** 的总量实测；线上历史约 30 分钟包含其他启动阶段和多节点等待，尚未测量其优化后总耗时。
机器可读汇总见 `total58_comparison.json`。
