# Action128 视频采样频率

训练使用 `meta/episodes/**/*.parquet` 的 `video_views[].fps` 作为视频采样频率，
`episode.fps` 独立表示 action 频率；有效 action 时间戳继续来自数值表 `timestamp`。

`dataloader_train.action_video_fps_default` 保留旧配置名称，但现在表示**上限**，不是缺省填充值。
设置为 30 时，只将高于 30 的视频采样频率降到 30；15、20 等低频保持原值。
缺失或非法 view FPS 会报 schema 错误，不再使用 action 频率补齐。
显式 catalog `data.fps` 也作为采样上限，不能提高低频视角的采样频率。
多相机拼图必须共享时间网格，因此各被选中的相机频率不一致时，取最低值，再应用配置上限。

2026-10-09 已按该规则更新训练配置指向的发布数据集。
检查了全部七个数据集、全部 split 的 episode 元数据；只有 RoboMIND2.0 需要改动：
7 个 Parquet 文件，206,517 条视角引用，修改字段仅为 `video_views[].fps`。
其他字段（包括 episode/action FPS、视角时间边界、路径）经备份逐字段比对保持一致。
数值数据、catalog、MP4 文件均未重写。

这定义的是训练采样策略，不是 MP4 编码 FPS，也不是源时间轴修复。
视频仍按照原有时间戳 seek/解码；其他读取器应按时间采样，不应把这个字段当作编码帧率去重新解释 MP4 时长。
发布数据内的说明：`RoboMIND2.0/meta/video_fps_sampling_policy.json`。

原文件备份保存在 `RoboMIND2.0/meta/video_fps_cap_backup_20261009/episodes/`。
文件通过临时文件校验、原文件硬链接备份、原子替换更新；不会改到同 inode 的其他发布快照。
更新清单和校验结果位于本仓库 `outputs/action128_view_fps_cap_20261009/`。
已有进程可能缓存旧 metadata；要使用新采样规则需重启并加载更新后的代码和元数据。
