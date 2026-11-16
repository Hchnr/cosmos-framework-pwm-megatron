# Action 视频临时容错

**临时方案，不是正确的数据修复。跳过窗口会改变实际训练样本分布；最终必须清洗或重新导出上游坏视频。**

`action128_wam_nano_160gpu.toml` 已显式开启该方案，其他配置默认保持解码失败时报错。
不修改源视频、源 Parquet 或 catalog。

```toml
[dataloader_train.media]
action_skip_media_errors = true
action_max_consecutive_skips = 100
# 可选：不设置时使用 IMAGINAIRE_OUTPUT_ROOT（其次 OUTPUT_ROOT，最后 outputs）下的 media_failures。
# action_failure_log_dir = "/absolute/path/media_failures"
# 可选：启动前准备好并冻结内容，包含下述窗口标识。
# action_exclusion_jsonl = "/absolute/path/action_bad_windows.jsonl"
```

Action 的窗口读取在 `EpisodeBlockDistributor` 内完成，因此在这里捕获 `MediaDecodeError`。
任一相机失败时丢弃整个窗口，重置 Action128 的解码进程，并从同一数据源继续取样补足 batch；不跳过训练 step、不补假帧。
配置、schema、shape 等非媒体异常继续抛出。每个 worker 对每个数据源独立计数，连续跳过达到上限就报错，计数跨数据 epoch、不跨进程重启；成功读取窗口后清零。
预先排除的窗口也计入该上限，防止整条流被排除后无限循环。

失败日志按数据源、全局 rank、worker、PID 分文件写 JSONL，每次失败立即关闭文件。
记录 root、episode/window、原始流游标、时间、异常、相机和请求的视频时间范围（解码失败时），以及累计 accepted/skipped 和 skip_fraction。
比例为该 worker 自启动以来在该数据源的候选窗口统计，不能直接当作全局训练 batch 丢弃率；预取日志也可能领先于模型训练步数。
日志同时输出 warning，供后续清洗定位使用。日志目录写入失败也会报错，避免无记录地丢弃数据。

排除清单每行一个 JSON 对象，例如：

```json
{"root":"/share/mixed/projset/eai_pwm/datasets/action_processed_r2/libero","episode_index":924,"window_index":2}
```

这里只是格式示例，不会自动启用。`window_index` 从 0 开始；相同 root 下的多个数据源不应使用不同的窗口划分却共用该清单。
可以从解码失败日志人工审核后提取这些字段；不要把所有超时直接认定为永久坏数据。
所有节点应读取同一个冻结清单。训练及断点恢复期间保持清单、窗口长度、stride、数据集和分片设置一致。
候选游标在跳过后照常前进，成功 batch 记录的游标涵盖此前跳过的窗口；最后一个已保存 checkpoint 之后的预取失败仍可能在恢复时再次遇到。
此机制不保证清单变更或非确定性 I/O 故障下的逐位一致恢复。
