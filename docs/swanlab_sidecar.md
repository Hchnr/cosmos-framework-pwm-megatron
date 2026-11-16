# W&B 日志旁路同步到 SwanLab

训练保持 `wandb.init` / `wandb.log` 和原始 `.wandb` 格式，使用 `WANDB_MODE=offline`。
独立工具节点上的旁路读取文件的固定长度快照，忽略尚未完整写入的尾部，
将原 step、标量名称和数值上传到 SwanLab。旁路每15秒扫描，实际延迟还包含训练日志刷盘和云端入库时间。
不修改正在运行的训练进程，不要求在训练环境安装 SwanLab。

系统指标来自原训练进程的 W&B 系统采样，每个训练 step 取最近一次采样；
旁路自身硬件采样关闭。DeviceMonitor 表格显示为 HTML 表格；原始数据和文件仍保留。

## 训练命令

现有58源启动链已经把 `WANDB_MODE` 作为最终 `job.wandb_mode` 覆盖项传入，
因此只需将原来的 `export WANDB_MODE=online` 换为：

```bash
export WANDB_MODE=offline
unset COSMOS_REQUIRE_WANDB_ONLINE
```

保持现有24节点启动流程、OUTPUT_ROOT 和配方。不要使用 `WANDB_MODE=disabled`，否则旁路没有 W&B 数据可读。
SwanLab Key 和代理不需要导入192个训练进程；旁路自己从私有文件读取。

## 启动旁路

58源线上入口 `examples/launch_pretrain_state_h1_58src_192gpu.sh` 默认在进入
torchrun之前自动调用管理器的 `--background` 模式。所有节点共用输出目录中的互斥锁，
仅启动一个后台管理器；若工具节点已有管理器，则直接复用。
无需在启动命令中再添加旁路命令。设置 `SWANLAB_SIDECAR_ENABLE=0` 可关闭自动启动，
`SWANLAB_PYTHON` 可指定独立旁路环境的Python。启动失败仅告警，不阻断训练。
这个入口不覆盖显式设置的 `WANDB_MODE`，原命令仍需按上文改为 `offline`。

也可以在能访问共享输出目录的一个工具节点上单独启动，不要放到每个训练rank内部：

```bash
cd /share-mixed/project/eai_pwm/repos/cosmos-framework-pwm
/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-swanlab-bridge/bin/python \
  tools/swanlab_sidecar.py \
  --output-root "$PWD/outputs/pretrain58src_state_h1_nano_192gpu_ga3_run01" --background
```

当前该管理器已经运行，继续使用同一OUTPUT_ROOT时不需要重复启动。
管理器在共享目录使用互斥锁，重复启动不会生成第二个管理器。
若工具节点/容器整体重启，可重新运行上述命令，或在下次通过58源入口启动训练时自动启动；
尚未配置平台开机自启。改用新OUTPUT_ROOT时，58源入口也会为新目录启动管理器。

需要单独对比重启后的性能时，使用新的OUTPUT_ROOT，并为管理器传入
`--name <新实验名称>`，让新曲线显示有意义的名称。已有会话仍按持久化run ID续传。
传入 `--step-log /path/to/node0.log` 可同时同步 `performance/step_seconds`，用于比较每步实际耗时。

旁路环境：`/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-swanlab-bridge`。
该独立环境上传图片预览时需要安装 `pillow`；仅标量同步不依赖图片处理包。
单个媒体转换失败会记录到 `online_state.json` 的 `media_conversion_errors`，
标量继续上传，失败媒体保持原来的逐key游标，在后续周期重试。
私有凭据文件：`/share-mixed/project/eai_pwm/.secrets/swanlab_api_key`。
私有代理配置：`/share-mixed/project/eai_pwm/.secrets/swanlab_proxy`。
不要在聊天、命令行参数或日志中打印凭据内容。

## 手动重启与断点接续

管理器直接扫描 `OUTPUT_ROOT/cosmos3_action/action128/*/wandb/*/run-*.wandb`，
选取最新启动的日志目录，不再依赖 `active_launch.txt` 或额外manifest。
因此用户手动执行原有启动命令、生成新的 W&B 目录后也会被发现。
即使 W&B 复用同一个run ID，新目录仍作为新的同步会话，创建独立SwanLab run，保留旧曲线，
避免从较早checkpoint恢复后覆盖旧训练step。

同一日志目录的旁路进程退出时，管理器会重新拉起它，接续持久化的SwanLab run ID，
根据服务器确认的各标量step补传。云端run不随旁路进程重启而改变。
旧W&B目录、训练checkpoint和所有跳过样本原始记录均由原训练流程保留。

## 状态与验证

`OUTPUT_ROOT/logs/swanlab_bridge/manager_status.json` 提供当前云端链接、上传进程PID、
源文件最新step及 `server_confirmed_steps`。服务器确认与本地入队分开记录。
每会话的 `state_dir` 中保存 `online_state.json`、提取审计和原始SDK私有日志。
当前旧旁路可被管理器接管，迁移管理器不会重启正在运行的上传或训练进程。

代码：`tools/swanlab_sidecar.py`（发现和监督）、`tools/swanlab_wandb_bridge.py`（读取和上传）。
已验证：真实日志提取、标量与DeviceMonitor表格离线导入、云端指标回读、
复用W&B ID的新目录识别、排除验证日志目录、现有上传器接管。
没有为了验证而重启当前训练。
