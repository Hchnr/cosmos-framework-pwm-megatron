# 58 源 State-H1 线上续训入口

正式入口为 [`launch_pretrain_state_h1_58src_online.sh`](launch_pretrain_state_h1_58src_online.sh)。
它包含容器环境配置、节点本地代码部署、完整 checkpoint 检查及 SwanLab 旁路启动。

在平台的全部 20 个容器中执行相同命令：

```bash
bash /share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/examples/launch_pretrain_state_h1_58src_online.sh
```

平台必须提供各节点的 `NODE_RANK` 或 `RANK`（0–19），以及相同的 `MASTER_ADDR`。
当前配置为 20×8 GPU、DP shard 16、replicate 10、GA 3。

## 实验名称与恢复状态

- 默认固定代码提交 `29ba8fc703ef5e7f65a345c4c27769a7bdea0fac`，不包含后续工作区修改。
- 从脚本中明确指定的物理归一化版本 `iter_000005800` 恢复模型、优化器、调度器、trainer 和全部 160 rank 的数据状态；没有自动选择最新 checkpoint。
- 包含 Resize1F、MP4 四线程和默认启用的 CPU 后台结果接收；CPU 后台接收的生产性能仍待验证。
- `RUN_TAG` 决定输出目录、rendezvous ID 和 SwanLab 实验名称。新实验需在所有节点设置相同的新值，例如：

```bash
RUN_TAG=20261103_r29ba8fc7_run02 \
  bash /share-mixed/project/eai_pwm/repos/cosmos-framework-pwm/examples/launch_pretrain_state_h1_58src_online.sh
```

更换 checkpoint 时，同时修改入口里的路径、`RUN_NAME` 和 `SWANLAB_NAME` 中的恢复 step。
训练使用 W&B offline；master 上的旁路上传到 SwanLab，新输出目录用于创建独立曲线。

## 容器与网络要求

共享训练及 SwanLab venv、模型、数据、私有凭据路径需已挂载。
容器需能写 `/opt/cosmos-framework-pwm`；没有 FFmpeg 时，脚本通过 `apt-get` 安装，因此需要 root 权限及可用的软件源。
训练启动前会检查 TorchCodec 等媒体依赖以及每节点 8 张可见 GPU。

`NCCL_IB_HCA='=mlx5_101'` 仅为当前 d39afa19 节点上验证过的 RDMA 绕行配置；换节点后必须重新核验。

## 与历史启动链的关系

`launch_pretrain_state_h1_58src_192gpu.sh` 保留从旧 52 源 checkpoint 扩展的用途；
52 源入口和 `launch_pretrain_state_h1_96gpu.sh` 仍被其他入口依赖。
本入口复用后者的模型资产及 torchrun 启动逻辑，文件名中的 96 不决定实际 GPU 数。
