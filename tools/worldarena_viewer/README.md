# WorldArena 离线对比页

在仓库根目录执行（仅使用 Python 标准库，不运行推理或重新打包视频）：

```bash
LD_LIBRARY_PATH='' python tools/build_worldarena_viewer.py
```

默认读取 `outputs/worldarena_eval25_480p_steps50/viewer_data.json`，同时更新结果目录和同级 `_transfer` 目录的 `index.html`。HTML 内嵌全部样本数据、CSS 和 JavaScript，可直接用 `file://` 打开，无需网络或服务。

- `index.html`：页面结构。
- `viewer.css`：响应式布局和统一标签颜色。
- `viewer.js`：筛选、选择记忆、按需加载、样本切换和同步播放。
- `../build_worldarena_viewer.py`：数据校验、排序、旧页面迁移和生成。

## 维护实验列表

在 `viewer_data.json` 的 `runs` 中维护实验，保留稳定的 `id` 以兼容已有选择记忆。`directory` 是压缩包内结果目录，`title` 保存完整原名，`step` 是实际训练步数（baseline 为 `null`）。`family` 为 `baseline` / `native` / `v1` / `v2`，`prompt` 为 `original` / `reasoner`，`tags` 是 `[短语, 颜色类别]` 列表。

公共的 Cosmos3-Nano、T2V SFT 和 online 权重在页头、分组标题或视频底部显示；实验差异以标签显示。完整名称保留在选择项悬浮提示及视频下方的展开区。标签只用于描述，筛选由搜索框和系列下拉框控制。

生成时 baseline 在前，其余实验按数字 step 全局递增；同一步数按模型系列排序，同系列的常规版在 FP8 版之前。勾选列表、视频卡片、运行参数链接及打包列表共享此顺序。筛选不会取消已选实验；“全选”和“清空”始终作用于全部实验。

旧页面首次迁移，或从已生成页面恢复数据：

```bash
LD_LIBRARY_PATH='' python tools/build_worldarena_viewer.py --import-html PATH/TO/index.html
```

自定义数据和输出：

```bash
LD_LIBRARY_PATH='' python tools/build_worldarena_viewer.py --data PATH/TO/viewer_data.json --output PATH/TO/index.html
```

无需继续使用历史审计脚本中的 HTML 字符串替换片段。新增正式实验应更新数据后重新生成；磁盘中保留的非正式实验不会自动进入页面。

## 金山云发布（固定目标）

WorldArena 页面或 transfer 文件更新后，交付流程包含同步金山云并校验远端；本地生成完成不等于已发布。

| 项目 | 固定位置 |
| --- | --- |
| KS3 目标 | `ks3://baai-eai/eval/worldarena_eval25_480p_steps50_transfer/` |
| 本地上传目录 | `/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm/outputs/worldarena_eval25_480p_steps50_transfer/` |
| 同步脚本 | `/share/project/eai_pwm/home/hcr/app/sync_worldarena_eval25_480p_steps50_transfer.sh` |
| KS3 工具 | `/share/project/eai_pwm/home/hcr/app/ks3util` |
| 凭据配置文件 | `/share/project/eai_pwm/home/hcr/app/.ks3util.conf`（仅引用路径，不记录凭据内容） |
| 机房内网 endpoint | `ks3-cn-beijing-internal.ksyuncs.com` |

从仓库根目录生成并发布页面：

```bash
LD_LIBRARY_PATH='' python tools/build_worldarena_viewer.py
bash /share/project/eai_pwm/home/hcr/app/sync_worldarena_eval25_480p_steps50_transfer.sh
```

如果新增实验，先使用 `tools/package_worldarena_checkpoints.py --runs <结果目录名>` 更新对应压缩包和清单，再执行同步脚本。仅修改页面或文档时，无需重新推理或打包。公共文档在结果目录和 transfer 目录均需更新，以免后续打包复制旧文档。

脚本将 transfer 目录的内容直接同步到上述 KS3 前缀，使用增量模式 `-u`、`--jobs=32`、`--parallel=2`，不会额外嵌套一层目录。同步日志与任务清单位于 `/share/project/eai_pwm/home/hcr/app/logs/` 和 `/share/project/eai_pwm/home/hcr/app/ks3util_output/`；检查失败数为 0。不要给日常增量发布添加 `--delete`。

上传后查看对象时间与大小，并下载新版页面进行逐字节核对：

```bash
cd /share/project/eai_pwm/home/hcr/app
./ks3util ls ks3://baai-eai/eval/worldarena_eval25_480p_steps50_transfer/index.html \
  -c .ks3util.conf -e ks3-cn-beijing-internal.ksyuncs.com
worldarena_verify_dir=$(mktemp -d /tmp/worldarena-ks3-verify.XXXXXX)
./ks3util cp ks3://baai-eai/eval/worldarena_eval25_480p_steps50_transfer/index.html \
  "$worldarena_verify_dir/index.html" \
  -c .ks3util.conf -e ks3-cn-beijing-internal.ksyuncs.com -f
cmp /share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm/outputs/worldarena_eval25_480p_steps50_transfer/index.html \
  "$worldarena_verify_dir/index.html"
```

`cmp` 返回 0 表示云端页面与本地一致。本次同时修改的公共文档也应下载核对。页面维护说明在仓库的 `tools/worldarena_viewer/README.md`；下载与解压说明随 transfer 一起保存在 `DOWNLOAD.md`。
