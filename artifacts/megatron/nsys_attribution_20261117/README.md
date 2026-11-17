# 四卡 Megatron-FSDP nsys 归因

中文报告与原生 Nsight App 截图：[报告](../../../docs/megatron_fsdp_nsys_attribution_zh.md)。

- `profile_analysis.json` / `summary.json`：原生 FSDP2 与默认 Megatron ZeRO-3，4 ranks × 2 更新。
- `ablation_no_prefetch_*` / `ablation_comparison.json`：实际禁止 lookahead prefetch 的有效消融；rank 0 记录 1106 次拦截。
- `input_comparison.json`、逐 rank 输入记录及 `validation.json`：输入、完整采集窗口与状态检查。
- `raw_reports_manifest.json` / `archive.json`：原始 `.nsys-rep` 路径、字节数和 SHA256。
- `rejected_ablation.json` / `ablation_no_param_overlap_analysis.json`：无效配置尝试。MCore 覆盖了传入的 false，不能用于归因。

原始报告与约 218 MB 的 `nsys_raw_reports.tar` 保留在共享目录，未放入 Git：

```text
/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron/outputs/megatron_validation/nsys_attribution_20261117/
```

数据包保留 8 份主对比、4 份有效消融、4 份明确标注的无效配置尝试。统计中的 kernel 服务时间和 CPU API 时长不能直接相加当作墙钟收益。
