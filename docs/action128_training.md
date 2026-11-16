# Action128 统一训练接入

## 修订后的约定

- 一个五列 Parquet catalog 描述一个发布数据集，统一使用 `adapter="unified_action128"`，不再按 DROID、RoboTwin、RoboMIND 创建不同 reader。
- 仅取发布的 `data_role="training" && training_eligible=true`。不再依据 caption 的 outcome 筛选。
- action 始终为发布的 128D；仅按发布的 sidecar 做 affine 缩放与截断，不重排关节、不重新处理夹爪。
- 默认预测有效 action（包括观测状态派生目标）；control-only 额外应用 `action_control_mask`，仍使用发布的统计参数。
- 完整 `action_domain` 用于归一化及逆变换，当前 40 类；batch 中的 `normalization_domain_id` 对应它。普通夹爪和 rot6d 按 sidecar 保持原值。
- Projection 独立路由：所有 EEF 共用 head 0，Joint 按 embodiment 使用 head 1–31，固定初始化 64 个槽位；旧编号 0–21 不变。显式映射见 `cosmos_framework/data/generator/action/projection_routing.py`，不按训练数据子集重新编号。
- batch/packed sequence 中原有 `domain_id` 表示 projection head ID。原位 `action2llm/llm2action` 使用新 128D 参数，旧 projection 实现已删除。基础 DCP warm-start 跳过旧投影，精确 resume 保留训练后的投影与契约；归一化 buffer 位于无训练参数的 `action_normalization` 模块。
- 默认 2 秒窗口、2 秒 stride，不跨 episode。尾部右 padding，并携带 temporal/action/空间有效 mask；全空监督窗口由完整统计文件排除。
- 使用 episode 的原生 FPS，不对 action 降采样。视频按目标 timestamp 读取最近帧，右侧补到 `4n+1`。action mRoPE 使用实际 episode-local timestamp。
- 相机按 `view_mapping/video_views` 关联，等比 cover resize 后 center crop 到布局框，不做空间 padding。双路中间横线、三路横线和下半部竖线均覆盖 4 像素黑边（两侧各覆盖 2px），不改变画布尺寸；黑边参与视频监督。缺少 main 时按发布的 auxiliary role 选择，损坏的视频引用明确报错。
- caption 保留完整任务：60% task+全计划，30% 额外加窗口起点对应的 current_subtask，10% task only。始终写入 `embodiment=episode.embodiment_name`；多路的 `camera_layout` 直接为数组，每项为 `[role, position, width, height]`，宽高指输出布局框（包含被黑边覆盖的像素）。
- 整体画布尺寸只写顶层 `resolution`，`camera_layout` 不再包含 `size` 或 `views` 对象层。`task_title` 保留发布值，训练 `command` 读取 VLM 的 `source_task_description`，`subtasks` 提供 `task_plan`；缺少有效 `source_task_description` 立即报错，不回退到发布的拼接 `command`。

### RoboTwin 的更正

此前指出的短片段不是“缺失计划”：它删除了 5 秒前置 idle，保留了完整的单一放杯子 subtask。按用户提供的全量核查，10 条切分 representation episodes 对应 5 条物理轨迹，均在 idle→有效动作边界切分，没有切进非 idle subtask。

因此不把这份数据视为接入阻塞。未来**离线发布器**应拒绝落在非 idle subtask 内部的过滤切分边界；这不限制训练时普通的 2 秒窗口，也不能只凭已重映射的 caption 反推出原始切分情况。本轮不改写离线发布数据。

## 入口

每个发布根必须同时带有 `meta/action_normalization.json`、状态为 `passed` 的
`meta/a1_structural_gate.json`，并由 `meta/info.json` 与 `manifest.json` 引用。它们是训练读取的强制契约：
缺失、版本不符或 A1 未通过时立即报错，不提供 raw action fallback，也不在 Cosmos 内重新拟合统计量。
Dataloader 只使用发布的 domain-slot q01/q99 affine transform，将 Parquet 中保存的 raw action 动态映射到
`[-1, 1]`；mean/std 仅为 sidecar 诊断信息，不叠加第二次标准化。

入口脚本：`examples/launch_sft_action128_wam_nano.sh`；配套配置：`examples/toml/sft_config/action128_wam_nano.toml`。七份 Parquet 的路径和比例都在 TOML 中。默认 Nano two-tower WAM、8 卡、GA=1、global batch=8、1000 steps，七源 sequence ratio 为 1:1:1:1:1:1:1；W&B offline，EMA/compile 关闭。训练 forward/backward 尚未验收。

```bash
cd /share/project/liyue/cosmos-framework
source /root/.venv/bin/activate
export LD_LIBRARY_PATH=''
bash examples/launch_sft_action128_wam_nano.sh
```

## 验收清单

Caption 的 `fps` 表示模型输入视频的采样频率，`action_frequency_hz` 表示 episode 的 action 频率，均使用数值类型（例如 `24`、`30`）。视频采样优先使用 catalog 的 `data.fps`；该字段缺失或为 null 时，先使用 `[dataloader_train].action_video_fps_default`；配置也未设置时，使用 episode 的 `fps`（action 频率），不再默认采用主相机的 `video_views[].fps`。各路相机在统一的视频目标时间点取最近帧。Action 保留逐行 timestamp，不随视频重采样；两者通过实际时间对齐到共同的 RoPE 时间轴。Batch 分别传递 `conditioning_fps` 和 `conditioning_fps_action`，padding 不计入有效监督。采样规则变化已更新 loader fingerprint，旧规则的 loader 游标不应作为相同数据流继续恢复；该回退不限制高频 action 对应的视频帧数。

1. 合成数据：training 集合、原值、tail/control/dim mask、caption、注册表、meta 初始化、masked loss。
2. TOML→catalog loader→collator→packed topology 的实际遍历；旧 packing/resume 回归。
3. 三份真实发布数据：各读取两个窗口，检查视频、action 和 native FPS。
4. 校验全部发布 sidecar、A1 门禁和 domain 注册表；全空监督窗口在读取时按 drop-sample 契约处理。
5. 本轮交付边界为真实 dataloader 构建及遍历；模型 forward/backward、checkpoint 恢复和 CP=2 训练留待 GPU 空闲后验证。

CPU 验收入口（读取原 TOML，默认将 worker 数设为 0、关闭 pin memory，避免占用 GPU）：

```bash
export LD_LIBRARY_PATH=''
export HF_MODEL_PATH=/share/project/eai_pwm/models/Cosmos3-Nano-VLM
/root/.venv/bin/python -m cosmos_framework.scripts.inspect_action128_dataloader \
  --toml examples/toml/sft_config/action128_wam_nano.toml \
  --output outputs/action128_acceptance --batches 14
```

每条结果保存 detokenize 后的真实 caption、head/normalization ID、action/mask 摘要和拼接首帧。

实现不自动 stage、commit 或 push。

## 路由追加更新（2026-10-09）

当前分支已有 0–28 保持原编号，追加以下 Joint heads；2026-10-12 追加 head 31，现共 32 个已分配 head、64 个总槽位，32–63 暂不分配。编号注册不代表数据已加入训练，当前七源 TOML 不自动扩展。

| Head | `embodiment_name`（路由 key 前缀为 `joint128/`） | 数据集 | 发布状态（按数据侧提供的清单） |
| ---: | --- | --- | --- |
| 22 | `agibot_g2__parallel_grippers` | AgiBotWorld2026 | R2 已发布 |
| 23 | `galaxea_r1lite_mobile_bimanual__parallel_grippers` | Galaxea | R2 已发布 |
| 24 | `galaxea_r1pro_mobile_bimanual__parallel_grippers` | Galaxea | R2 已发布 |
| 25 | `agilex_split_aloha__parallel_grippers` | InternData-A1 | R2 已发布 |
| 26 | `arx_lift2__parallel_grippers` | InternData-A1 | R2 已发布 |
| 27 | `genie1__parallel_grippers` | InternData-A1 | R2 已发布 |
| 28 | `panda_omron_mobile__parallel_gripper` | RoboCasa365 | 发布准备中，先注册编号 |
| 29 | `paxini_df2__faas_bimanual_hands` | Paxini-Omnisharing | R2 已发布 |
| 30 | `sharpa_north__wave_bimanual_hands` | Sharpa-North | R2 已发布 |
| 31 | `agilex_piper_x__parallel_gripper` | 2026-10-12 catalogs | 新增本体 |

- Head 0 仍覆盖所有 `eef20`：AgiBotWorld-Beta、AgiBotWorld2026、DROID、FastUMI、InternData-A1、InternData-M1、RoboMIND-v1、RoboMIND2.0、RoboTwin2.0、LIBERO，以及待发布的 RoboCasa365 EEF。
- Head 8 的 Franka Joint 来源增加 InternData-A1，原 RoboMIND-v1、RoboMIND2.0、RoboTwin2.0 不变。Split Aloha 不与 ALOHA 合并，Panda Omron 不与裸 Franka 合并，Galaxea 不与 Galbot 合并。
- HumanTouch 的 `action128/xspark_humantouch_bimanual` 与 T-Rex 的 `joint128/trex_vega1__sharpa_wave` 暂不注册；未知 key 继续报错，不自动回退。
- Head ID 和 normalization domain ID 独立，仍按完整 `action_domain` 读取发布 JSON；新增本体不在 loader 中新增统计拟合、动作重排或手部变换。
- Head 14、17、18 的手部契约更新后，必须重新构建配置和 dataloader（包括 worker），重读 metadata、逐帧 mask 和归一化文件。不支持运行中覆盖数据并热更新旧 worker。每次构建重新读 sidecar，新的 spec/source digest 与 episode metadata 指纹会使旧数据流恢复状态失效；新 reader 不复用旧进程内 Parquet 缓存。
- 只追加 head 不改变 64-slot 参数形状。恢复分为下面两种情况，不通过重置 heads 绕过契约校验。

### 2026-10-12：22 源模型权重迁移

`action128_wam_nano_160gpu.toml` 和同名 launcher 保留原文件名，默认 18×8 GPU
（shard16×replica9）；设置 `NNODES=20` 可使用 20×8 GPU（shard16×replica10）。
22 份 EEF/Joint catalog 以 sequence 百分比 ×100 为基础权重，LIBERO EEF/Joint
均设为 100，权重合计 10194；归一化后各约 0.981%，合计约 1.962%。
每 rank 22 个 workers（每节点 176 个）。

这批数据相对旧 run 的归一化契约也发生变化，使用显式模型权重迁移：

```toml
[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
load_training_state = false
action128_load_mode = "warm_start"
keys_to_skip_loading = []
strict_resume = true
```

- 加载旧 checkpoint 的模型权重及全部 64 个 Action heads，使用新数据的归一化表。
- 不恢复 optimizer、scheduler、step、grad scaler、RNG 或数据 cursor；step 从 0 开始。
- 校验源契约 digest、旧 head 编号、槽位数、空间 RoPE 及模型参数形状；允许 domain
  增删和归一化参数变化，不使用 EMA 覆盖训练权重。
- `BASE_CHECKPOINT_PATH` 必须指向已训练 Action128 的 `checkpoints/iter_*`，不能继续
  使用原 E2aux 模型初始化目录。源 checkpoint 需有契约 metadata 或原 run 的 `config.yaml`。
- 使用全新的 `OUTPUT_ROOT`；同一新 run 后续自动恢复仍按正常逻辑恢复完整状态。
- 原 `extend` 仍仅支持旧契约不变的追加式完整状态续训，其限制未放宽。

### Checkpoint 恢复与追加数据

| 情况 | 设置与行为 |
| --- | --- |
| 数据、编号、归一化均不变 | `action128_load_mode="auto"`：正常恢复模型、optimizer、scheduler、step、数据流 cursor。 |
| 只追加本体/数据源，旧契约不变 | `action128_load_mode="extend"`：保留全部 64 个投影槽和 optimizer/scheduler/step；使用新归一化表；数据流从头开始。这是继续训练，不是精确续训。 |

追加时，保持原 catalog 的顺序，将新 catalog 加在末尾。使用新的输出目录，从原训练 checkpoint 加载：

```toml
[checkpoint]
load_path = "/path/to/old_run/checkpoints/iter_000001000"
load_training_state = true
action128_load_mode = "extend"
strict_resume = true
keys_to_skip_loading = []
keys_not_to_resume = []
```

- 按完整 domain 名逐项比较旧的本体、head、发布 transform 和 source digest；允许新增 domain 改变归一化表行号，不允许修改/删除旧规则、重编 head 或改变空间 RoPE。
- 检查已有数据流的 fingerprint；不允许替换/删除旧数据源。仅新增数据源可以调整混合比例。此模式内部强制严格加载模型及训练状态，不重置新启用的预留槽，也不将 EMA 覆盖到训练权重。
- 新 checkpoint 将完整 Action128 契约写入 trainer metadata。旧 checkpoint 没有该字段时，读取原 run 的 `config.yaml`，并用 checkpoint 中的 digest 核实。须保留原配置；找不到或不匹配就报错，不猜测旧契约。旧 run 应使用新输出目录继续，以免原配置被启动流程覆盖。
- 追加后的 checkpoint 再次恢复时，若契约和数据未变，即使仍设置 `extend`，也正常恢复新数据流 cursor，不会重复清零。
- `reset_heads` 仍仅供原有的显式 model-only 重启；不属于上述两种恢复策略，`extend` 不会自动调用它。

## 历史验证记录（2026-09-22）

- Action128、collator 和 resume 回归覆盖 center crop、4px 黑边、60/30/10 caption、EEF 共享 head 与独立逆变换。
- 原 TOML 构建七源 dataloader，每源读取两个窗口，共 14 个 batch，通过本地 Nano tokenizer、视频解码、collator 与 q01/q99 变换检查；归一化槽均在 `[-1,1]`。注册表包含 40 个 normalization domain、21 种本体、22 个已分配 projection head，模型固定预留 64 个 slots。
- 本轮真实样例与反解 caption：`outputs/action128_acceptance_20260922/summary.json`；同目录保存各窗口 JSON 与拼接首帧 PNG。
- 训练入口切换到 `action128-full-20260922` 的七份 catalog；全局 domain 和发布变换从 sidecar 自动合并。LIBERO 无 `main` 标记时显式以 `auxiliary_0` 作为 primary，metadata 保留 fallback 标记。
- 真实模型 forward/backward、DCP 恢复及 CP=2 尚未验收：检查时 8 卡各已有约 75 GB 显存占用，未打断现有任务。

## 三塔融合与后续验收补充（2026-09-28）

本节更新上述2026-09-22的two-tower默认配置及待验收状态；原始数据契约和历史记录保留。

### 当前训练与恢复

- 标准入口仍为`examples/launch_sft_action128_wam_nano.sh`，现复用main已有`width_only_1b` ACTION三塔WAM；模型消费端适配`image_size=[H,W]`与整数raw_action_dim，不改源数据契约。
- 默认七源等比、GA1、CP1、1000步、LR2e-5/warmup100、在线VAE、full AC/VIDEO Flash3、routed heads；EMA关闭。沿用Action-1B容量：固定UND11776（短桶5120）/GEN62592（VIDEO59648+ACTION2944），按token容量打包（样本上限92），分别检查VIDEO/ACTION限额，启用静态language compile和本机/tmp缓存。延长训练同步调整scheduler。
- base须为已转换的`Cosmos3-Nano-action-1b-v022/width_only_1b`三塔checkpoint；可覆盖BASE_CHECKPOINT_PATH、HF_MODEL_PATH、WAN_VAE_PATH，不可直接使用原始two-tower Nano DCP。
- `checkpoint.action128_load_mode=auto`仅对旧base初始化新heads；已训练Action128保留heads并校验归一化/空间契约。显式`reset_heads`重建heads/契约，不能同时加载完整训练状态。
- 新目录完整续训：BASE_CHECKPOINT_PATH指向Action128 DCP并加`checkpoint.load_training_state=true`，skip列表须为空；同目录同job自动恢复model/optim/scheduler/trainer/dataloader。
- 归一化scale/bias保持发布FP32值，旧BF16 buffers仅在契约验证后重建；固定验证保留布局、频率与监督mask，fixture的held-out属性须自行保证。
- 本体统计由归一化registry识别机器人，默认每100步按样本总和/计数聚合，覆盖GA并单列混合本地batch未归属样本；不改变采样配比。

```bash
# 仓库根目录，先激活训练环境并设置本机模型路径；全新验收使用独立输出目录。
export LD_LIBRARY_PATH=''
# 本地8卡：显式覆盖拓扑与小容量；使用独立OUTPUT_ROOT避免恢复旧实验。
NNODES=1 DP_SHARD=8 MAX_ITER=7 WANDB_MODE=offline OUTPUT_ROOT=outputs/action128_local_check \
bash examples/launch_sft_action128_wam_nano.sh \
  dataloader_train.max_samples_per_batch=4 dataloader_train.max_sequence_length=40960 \
  model.config.max_num_tokens_after_packing=40960 \
  model.config.fixed_sequence_length.und_tokens=8192 model.config.fixed_sequence_length.gen_tokens=32768 \
  model.config.fixed_sequence_length.action_tokens=1024 model.config.action_tower_und_short_tokens=2048
# 正式128卡：每个节点执行；平台提供MASTER_ADDR与节点RANK（或NODE_RANK）。
bash examples/launch_sft_action128_wam_nano.sh
```

默认16节点128卡/shard16/replica8、W&B online（需已配置凭证）；本地8卡/shard8与小容量仅由命令行覆盖。MAX_ITER覆盖训练步数，OUTPUT_ROOT覆盖输出目录，WANDB_MODE覆盖日志模式，尾部CLI优先；默认输出按8/128卡分目录，同目录自动resume，不同实验须设不同OUTPUT_ROOT。

### 拼接缝RoPE与解码优化

- 标准配方启用`action128_seam`：仅两/三路vision H改为`seam_pixels/S-row`，W仍为column；标准S=32，使用patch左上角并保留float32分数，上正下负；单路及其他模态不变。
- 旧模式用CLI `model.config.diffusion_expert_config.vision_spatial_position_mode=canvas`；空间模式不兼容时拒绝恢复。先拼接再VAE不保证跨缝patch或VAE感受野的相机隔离。
- 延迟非选中帧RGB转换，按实际packet身份复用解码前缀，七源各1worker/prefetch1；保留最近帧、超时及回退语义，七源画布/mask/metadata等价检查通过。
- 七源有界解码时限为300秒，解决已观察到的RoboMIND 120秒超时；不跳过样本，不保证全量数据永不超时。

### 验收结论与边界

- 标准8卡七源7步、外部恢复7→8、同目录8→9均exit0；72rank batches的loss/梯度、网络RoPE与FP32常量检查通过，两类恢复8rank投影哈希一致；固定验证跨进程总loss差约0.0023%，不宣称逐位一致。
- 空间/packing阶段251项回归通过；后续修复47/18/4/7项测试组均通过（有交集，不累加）。证据：`outputs/action128_rope_20260926/`与`outputs/action128_review_fixes_20260928/`。
- 匹配性能窗口耗时309.440→31.639秒（约9.78倍），完成8GPU nsys对照；该结果早于新增本体统计启用，不代表当前监控配置或长程吞吐。证据：`outputs/action128_opt_20260928/`。
- 正式T2V入口`examples/launch_sft_vision_nano_usr_128gpu.sh`单机8卡保留480/96K/GA2/compile，12600→12602→12603均exit0；48rank microbatches/188样本、5个实际source，loss/梯度正常，恢复投影一致，89项回归通过。
- T2V测试覆盖当前checkout的SFT_TOML_FILE、shard8、独立输出目录、W&B offline和本机/tmp编译缓存；初次共享盘缓存失败已记录。证据：`outputs/t2v_regression_20260928/`。
- 上述验证不代表128卡通信、长程收敛或机器人成功率验收；当前三塔配方仍限制CP=1。

- 小容量覆盖实测：在线VAE+pack4+固定长度+静态compile，8卡七源7步及同目录7→8均exit0；64rank batches/256样本、loss/梯度/RoPE检查通过，8rank恢复投影一致。配置/接口41项与最终拓扑9项回归通过（有重叠）；128卡仅验证shard16/replica8配置。证据：`outputs/action128_optimized_recipe_20260928/`。

## 160 卡 pilot（20 节点 × 8 H100）

> **⚠ 临时方案，不是正确的数据时间轴修复！** 当前160卡配方强制训练视频30 FPS，
> 仅绕过高帧率样本触发的256帧解码上限。原始频率尚未核实；低帧率视频会重复取帧，
> action频率、数值与时间戳保留发布值。不能将此实验当成正确物理时间轴的验收。
> TODO：完成源数据时间轴审计、明确采样策略后移除临时方案。

配置项 `[dataloader_train].action_video_fps_default = 30.0` 仅在 parquet catalog 的
`data.fps` 缺失或为 null 时生效；显式 FPS 优先，不生成或改写 catalog。
优先级：catalog FPS → 配置缺省 FPS → episode action FPS。配置未设置时保留 action 频率回退。
job名称为 `action128_wam_nano_160gpu_TEMP_video30`，应使用新的 OUTPUT_ROOT，
避免与旧采样规则混淆；使用30 FPS时两秒窗口61帧，action仍按原频率读取。

入口 `examples/launch_sft_action128_wam_nano_160gpu.sh`，配置
`examples/toml/sft_config/action128_wam_nano_160gpu.toml`。沿用七源等比、在线 VAE、
单卡 token 容量和 LR2e-5，使用 shard16/replica10、CP1、GA1；训练 1000 步、
warmup100、每100步保存。92 是每个 pack 的样本上限，实际全局样本数随 packing 变化。
此预算用于首轮实验，不表示全量 epoch 或收敛；160 卡通信与吞吐仍需线上验证。

每个节点运行相同命令，平台需提供 MASTER_ADDR 与不同的 NODE_RANK（0–19；也接受平台节点 RANK）：

```bash
# 先激活完整训练环境；所有节点使用同一共享输出目录。
export OUTPUT_ROOT=/share/project/eai_pwm/outputs/action128_wam_160gpu_run01
bash examples/launch_sft_action128_wam_nano_160gpu.sh
```

160 卡入口与 USR 平台入口一样，使用 `MASTER_ADDR:MASTER_PORT` 建立 c10d 会合；
任务 ID 取 JOB_ID，未设置时读取 `/etc/downwardapi/annotations` 中的
`scheduling.k8s.io/group-name`。覆盖平台预置的 RDZV_ENDPOINT/RDZV_ID，拒绝 `JOB_ID=none`。
默认会合等待7200秒；显式定制用 `ACTION128_RDZV_CONF`，不沿用平台 RDZV_CONF。
没有上述 annotations 的平台须给所有节点设置相同、此次启动独有的 JOB_ID。
新实验换 OUTPUT_ROOT；同目录同 job 自动恢复。
支持与 USR 一致的空格分隔 `EXTRA_TAIL_OVERRIDES`；尾部命令行参数优先级最高，
含空格的单个值应通过单独引用的命令行参数传递。
`$OUTPUT_ROOT/logs/action128_bootstrap_node<N>.log` 追加记录启动配置、预检查和训练输出；
`action128_node<N>.log` 保存本次 torchrun 输出。bootstrap 中包含实际拓扑、会合参数和 tail overrides。
W&B 默认 entity 为 `ybainlp`（可用 WANDB_ENTITY 覆盖）。凭证优先使用 WANDB_API_KEY；
未设置时读取 WANDB_API_KEY_FILE，默认 `/share/project/eai_pwm/.secrets/cosmos_wandb_api_key`。
该共享文件应仅对作业用户可读，不纳入版本控制，内容不写入日志；文件不存在时仍允许 SDK 使用已有登录。
线上镜像没有登录且无凭证时，online 初始化会失败；不会自动降级成 offline。
编译缓存默认位于本机 `/tmp/cosmos_compile_cache/action128_wam_160gpu/rank<LOCAL_RANK>`，
并发实验应各设 ACTION128_COMPILE_CACHE_ROOT。模型、七源数据和发布 sidecar 必须在每台机器可见。
本地检查曾因缺少 lerobot 无法构建 dataloader；应使用完整训练镜像。数据发布目录已有更新，
正式运行前固定发布版本并刷新 catalog provenance；验证集须按物理轨迹独立划分。
延长训练时同时覆盖 `trainer.max_iter` 与 `scheduler.cycle_lengths`，
并根据实验目的设置 `scheduler.warm_up_steps`；仅 MAX_ITER 不会更改学习率周期。

VAE AOT 保留默认融合优化（`emulate_precision_casts=false`），不要求与 BF16 eager
逐位一致。latent 保持原有数值阈值；缓存只允许极稀疏、有限且幅度受限的离群误差，
并必须通过使用各自缓存计算的后续两个 chunk latent 检查。shape/dtype、非有限值、
密集偏差或大幅离群仍拒绝，`require_full_coverage=true` 保持启用。
编译产物 schema 为 3；旧版本产物须重建，以免继续使用保留舍入的慢包。
这些检查是兼容性与传播检查，不等同于 FP32 精度认证；真实视频连续编码用于离线验收。

### 13 源配方的静态 compile 八卡验证（2026-10-09）

`launch_sft_action128_wam_nano_160gpu.sh` 对应配方现使用
`model.compile.compile_dynamic=false`、`compiled_region="language"`；decoder
按 `fullgraph=True` 编译。`trainer.compile_config.recompile_limit=128` 替代默认 8，
`use_duck_shape=false`。固定 token 容量不代表只有一张图：不同 pack 的
`_small_masks.max_q` 和 `cumulative_q` 长度仍会触发静态特化。
提高上限允许保留这些图，不会将模型切回动态 compile。
持久化 compile cache 管理仍关闭；job 名后缀为 `staticcompile`，与旧动态实验分开。

本机实测发现，104 个数据 worker 的视频大数组分配可触发同步透明大页整理，
表现为 CPU 内核时间高、内存压力高、GPU 等待，即使系统仍有大量空闲内存。
仅设置 `NUMPY_MADVISE_HUGEPAGE=0` 不足以解决所有分配路径。
入口默认设置 Linux `PR_SET_THP_DISABLE`，仅作用于当前作业及其子进程，
不改系统全局设置；`ACTION128_DISABLE_THP=0` 可保留主机原策略。

本地八卡命令（保留 13 源比例、在线 VAE、完整 token 容量和 1B ACTION tower）：

```bash
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29649 \
DP_SHARD=8 MAX_ITER=30 WANDB_MODE=disabled \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false TORCH_LOGS=recompiles \
OUTPUT_ROOT=/tmp/action128_static_compile_8gpu_new_run \
bash examples/launch_sft_action128_wam_nano_160gpu.sh
```

使用新的输出目录，防止自动恢复旧实验。可用尾部参数
`trainer.compile_config.recompile_limit=<正整数>` 调整限制。
短程八卡验证不等同于 160 卡或 100000 步训练验收，也不保证覆盖所有数据形状。

实测结果：8×H100 80GB、PyTorch 2.10.0+cu130，30 步正常退出（exit 0），
保存 `iter_000000030`；240 个 rank-step、2052 个样本，实际选中 7 个源。
诊断启用 `fail_on_recompile_limit_hit=True`，全部 rank/step 的 graph break 为 0，
最终各 rank 累计 11–14 张图（包括 mask 构建图），所有 loss 均为有限值。
除首步外，按每步最慢 rank 统计的 wall time 中位数 22.03 秒，范围 11.72–44.42 秒；
包含在线 VAE 和重编译，不能视为稳态吞吐。最大 allocated/reserved 显存为 57.32/66.70 GiB。
21 项配置/入口测试、配方 dryrun、Ruff 和 shell 语法检查通过。
本地证据：`outputs/action128_static_compile_8gpu_20261009/README.md`、
`summary.json`、`performance/` 和 `console.log`；诊断原始输出与 checkpoint 位于
`/tmp/action128_static_compile_8gpu_20261009_run05/`。
