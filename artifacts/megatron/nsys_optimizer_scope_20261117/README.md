# 扩大 post-autograd 更新范围的 Nsight 补测

完整结论和局部图见 [中文报告](../../../docs/megatron_fsdp_nsys_attribution_zh.md)。

- `raw/`：两端各 4 个 rank 的原始 `.nsys-rep`，通过 Git LFS 保留，未编辑。
- `raw_reports_manifest.json`：原始来源、字节数和 SHA256。
- `optimizer_analysis.json`：完整 CPU / GPU 完成边界与逐 callback 子阶段。
- `profile_analysis.json`：完整更新的 GPU 时间并集统计。
- `*_inputs_rank_*.json`、`input_comparison.json`：60 个微批逐项一致。
- `*_measurement.json`：整步计时、loss 与配置。
- `manifest.json`：采集代码哈希与运行范围。

下载后用 Nsight Systems 2026.2.1 打开。若本地文件只有 Git LFS 指针：

```bash
git lfs pull --include='artifacts/megatron/nsys_optimizer_scope_20261117/raw/*.nsys-rep'
```

节点本地完整采集目录：`/opt/cosmos-nsys-optimizer-scope/final/`，包括 SQLite 和启动日志。
共享盘新文件 ENOSPC 后改用节点本地目录，Git LFS 原始报告为额外持久副本。
