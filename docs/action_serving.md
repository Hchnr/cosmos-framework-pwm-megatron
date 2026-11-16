# Action128 模型 HTTP serving

此服务用于 `examples/toml/sft_config/action128_wam_nano_160gpu.toml` 对应的
Action128 WAM checkpoint，包括 three-tower / width_only_1b action tower。
服务沿用 Cosmos 原有通用 WAM serving 协议：`POST /generate` 接收
`OmniSampleOverrides`，返回 `SampleOutputs`。WAM 原字段保留，必要 Action128
扩展放在已有 `extra.action128`。原 `cosmos_framework.inference.ray.submit` 客户端可直接使用。
输入是当前观测图像/视频和任务文本，返回动作及可下载的未来 `vision.mp4`。
本训练 recipe 没有 state/history 条件，不套用旧 DROID/Libero 的 pose/gripper 变换。

## 1. 显式选择 checkpoint（推荐）

先将实现提交同步到 serving 使用的代码目录（本次工作目录与线上训练仓库是两个独立 checkout）。
在 **有空闲 GPU 的 serving 节点**执行，使用已安装仓库及 inference/serve 依赖的 Python 环境：

```bash
cd /share/project/eai_pwm/repos/cosmos-framework-pwm
source /root/.venv/bin/activate
export LD_LIBRARY_PATH=''

export CHECKPOINT_RUN_DIR=/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_e2aux345_run01/cosmos3_action/action128/action128_wam_nano_1b_e2aux345_12src_no_robomind2_160gpu_videocap30_static_vae_aot_run1
export CHECKPOINT_ITER=iter_000006000
export CHECKPOINT_PATH="$CHECKPOINT_RUN_DIR/checkpoints/$CHECKPOINT_ITER/model"
export CUDA_VISIBLE_DEVICES=0
export SERVING_GPUS=1
export SERVING_PORT=8000
export SERVING_OUTPUT_DIR=outputs/action128_serving_1gpu
bash examples/launch_action128_serving.sh
```

服务依赖 `fastapi`、`uvicorn`、Pillow；训练环境若缺少 HTTP 依赖，可执行
`python -m pip install fastapi uvicorn`。完整可选 serving 环境见项目 `serve` extra。
服务本身不需要 Ray、Gradio 或 OpenPI。

推荐用 `CHECKPOINT_PATH` 固定具体 checkpoint，服务不会替换成其他步数。
2026-10-12 此 run 已核验可选 `iter_000005600`、`iter_000005800`、`iter_000006000`；
修改上面 `CHECKPOINT_ITER` 即可切换。目录名固定9位，不需要修改模型代码或客户端。
`CHECKPOINT_PATH` 优先于旧的 `CHECKPOINT_ROOT`，避免 shell 里遗留 ROOT 导致选错版本。
CLI 同时支持 `--checkpoint-path` 与兼容别名 `--checkpoint-root`。

如果希望每次启动自动追踪该 run 的最新保存版本，显式改用以下方式：

```bash
unset CHECKPOINT_PATH
export CHECKPOINT_ROOT="$CHECKPOINT_RUN_DIR"
bash examples/launch_action128_serving.sh
```

自动选择方式读取训练写完后发布的 `latest_checkpoint.txt`，不是扫描最大步数目录。
`CHECKPOINT_ROOT` 可填写：

- 训练命令的 `OUTPUT_ROOT`（其下必须只有一个训练 run）。
- 包含 `config.yaml` 和 `checkpoints/` 的具体 run 目录。
- 具体 `checkpoints/` 目录。
- 指定已完成的 `iter_000006000` 或它的 `model/` 目录。

服务读取训练写入的 `latest_checkpoint.txt`，只加载已完成的 DCP model 权重和该 run 的
`config.yaml`，不需要先 export，也不加载 optimizer/trainer 状态。
不会选取正在写入的更大 `iter_*` 目录。指定历史 iter 时需要 `.save_complete` 标记；
没有该标记的老 checkpoint 只有在 latest 指针仍指向它时才可加载。
若 OUTPUT_ROOT 包含多个 run，启动报错，改为填写具体 run。

rank0 在启动时解析一次，并广播同一快照给所有 rank。后续训练产生新 checkpoint 不影响
当前服务；**重启服务**后选择新的 latest。成功启动后，`/info` 与
`SERVING_OUTPUT_DIR/serving_snapshot.json` 记录实际加载路径。
训练自动清理可能在加载期间删除旧 checkpoint；需要长期固定旧版本时，先复制完整
run 的配置及相应 checkpoint 到独立 serving 目录，再从该副本启动。

### 先检查路径（不加载模型、不占用 GPU）

```bash
python -m cosmos_framework.scripts.action128_server \
  --checkpoint-path "$CHECKPOINT_PATH" --dry-run
```

`--dry-run` 只检查 checkpoint 路径、完成证据、分片文件和保存配置是否存在，输出计划；
不代表权重加载或 GPU 推理已经通过。参数帮助：

```bash
python -m cosmos_framework.scripts.action128_server --help
```

## 2. 按卡数部署

每个 GPU 一个 torchrun rank；权重使用 FSDP 按服务 world size 重分片。
`CP=1`、`CFG parallel=1`，全部 rank 执行同一请求，只有 rank0 开 HTTP 端口。
服务卡数不必与训练的 144 卡（18×8）一致；训练 launcher 名字中的 `160gpu` 不决定服务卡数。

| 部署卡数 | CUDA_VISIBLE_DEVICES 示例 | SERVING_GPUS |
| --- | --- | --- |
| 1 | `0` | `1` |
| 2 | `0,1` | `2` |
| 4 | `0,1,2,3` | `4` |
| 8 | `0,1,2,3,4,5,6,7` | `8` |

多卡分片（包括多节点）启动时需追加 `--no-reuse-small-action-metadata`，
因为请求内元数据复用目前只支持单卡独立副本；默认开启时，多 rank 启动会明确报错。

卡数参数表示权重分片，单副本一次处理一个请求，不代表并发副本数。
单卡能否运行取决于整模型和生成过程的显存需求；action tower 为 1B 不代表整模型只有 1B。
本版本默认 `flash3 + compile + metadata-only`，适用于 native `width_only_1b` 三塔模型，
需要已安装 FlashAttention 3。单卡启动无需额外传入这三个优化开关。
可分别用 `--attention-backend sdpa`（或 `flash3_video`）、`--no-compile`、
`--no-reuse-small-action-metadata` 覆盖默认值。恢复原 SDPA/eager 服务配置：

```bash
bash examples/launch_action128_serving.sh \
  --attention-backend sdpa --no-compile --no-reuse-small-action-metadata
```

服务 launcher 的 `SERVING_GPUS`、`SERVING_NNODES`、`SERVING_MASTER_PORT` 独立于训练的
`NPROC_PER_NODE`、`NNODES`、`MASTER_PORT`，默认单节点单卡，避免继承 18 节点训练设置。
不要在训练仍使用的 GPU 上启动。

### 多节点

每个节点相同仓库、Python 环境和共享 checkpoint/资源路径，GPU 数量一致。
所有节点设置相同的主节点地址、rendezvous 端口、节点总数和模型参数；节点 rank 不同：

```bash
export CHECKPOINT_PATH=/shared/path/to/run/checkpoints/iter_000006000/model
export SERVING_NNODES=2
export SERVING_GPUS=4
export SERVING_MASTER_ADDR=10.0.0.10
export SERVING_MASTER_PORT=29571
export SERVING_NODE_RANK=0   # 第二个节点设为 1
bash examples/launch_action128_serving.sh --no-reuse-small-action-metadata
```

总分片卡数为 2×4=8，HTTP 地址是 node rank 0 的 `SERVING_PORT`。
节点间需要 NCCL/Gloo 网络连通，多节点 GPU 推理目前未实测。

### 移动模型依赖路径

默认沿用训练保存配置中的 Wan VAE 与文本 tokenizer 路径。部署到其他机器时可以覆盖：

```bash
bash examples/launch_action128_serving.sh \
  --vae-path /share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth \
  --tokenizer-path /share/project/eai_pwm/models/Cosmos3-Nano-VLM
```

需要与训练兼容的 tokenizer，不能任意换其他模型的词表。服务设置 eval 模式，关闭训练用固定 token padding bucket（及依赖它的 short UND bucket）和 EMA，默认开启 decoder 的 torch.compile，
加载训练普通权重（当前 recipe 的 EMA 本来就关闭），保留 Action128 source registry、
projection routing 和空间位置编码 contract。

## 3. 原有 Cosmos WAM HTTP 协议

接口类型直接复用 `cosmos_framework/inference/args.py` 的 `OmniSampleOverrides`
和 `cosmos_framework/inference/common/args.py` 的 `SampleOutputs`，与
`cosmos_framework/inference/ray/serve.py` / `submit.py` 的协议一致。

| 方法/路径 | 含义 |
| --- | --- |
| `POST /generate` | 原 Cosmos 请求与返回类型，执行 WAM 生成 |
| `GET /models` | 已部署模型名，默认 `["action128"]`，可用 `--model-name` 修改 |
| `GET /outputs/{path}` | 原 Cosmos 输出文件下载接口 |
| `GET /info` | 模型列表、output_dir、实际 checkpoint/config 路径、分片参数 |
| `GET /healthz` | 模型加载完成后返回 `{"status":"ready"}` |
| `GET /domains` | 当前 checkpoint 的 source action_domain 列表与 contract fingerprint |
| `GET /docs` | OpenAPI 文档 |

之前开发阶段的独立 cameras/action_fps/duration_seconds 请求和自定义 `/predict`
协议已撤换。请使用下面的原 WAM 结构。

### 原 WAM 字段保持不变

| 字段 | Action128 serving 中的含义 |
| --- | --- |
| `name` | 样本名，建议必填；省略时服务生成默认名 |
| `model` | 可选模型路由名，与 `/models` 一致 |
| `model_mode` | 必须为 `"wam"` |
| `prompt` / `prompt_path` | 原有任务文本；JSON prompt 可保留 command/task_title |
| `vision_path` | 原有本地路径或 URL，图像/视频的第一帧作为当前观测；多相机需预先拼成一个 canvas |
| `domain_name` | 保留旧客户端标签（如 libero/droid_lerobot）；Action128 实际路由以 extra 中的完整 source domain 为准，不套用旧整数 domain ID |
| `image_size` | 原有目标图像高度；省略保留输入高度并对齐 VAE 的16像素倍数，填写时须为16的倍数；拼接布局随缩放同步更新 |
| `action_chunk_size` | 原有动作步数 T；窗口时长=T/action_fps，当前训练 recipe 支持≤2秒 |
| `fps` | 原有视频 FPS，Action128 上限30；省略沿用原 WAM 默认24 |
| `num_frames` | 原有视频帧数；省略按动作窗口和视频 FPS 推导4k+1；显式填写时必须与该窗口一致 |
| `raw_action_dim` | 原字段，Action128 为128；有效槽数由扩展 mask 描述 |
| `seed` | 原随机种子；省略时服务生成并回显实际 seed |
| `num_steps` / `guidance` / `shift` | 原采样字段，省略使用原 WAM defaults（30 / 1.0 / 10.0） |
| `guidance_interval` / `normalize_cfg` / `sigma_max` / `negative_prompt` | 使用原字段和默认解析 |
| `video_save_quality` | 原视频输出质量参数 |
| `extra` | 原扩展容器，其他团队的 extra 字段照常保留 |

当前专用部署每个请求支持 `num_outputs=1`，串行接收；不支持 sound、prompt upsampling、
state/history，也不能通过 `condition_frame_indexes_vision` 引入额外条件帧。输出保存目录由服务生成。
HTTP 中的相对 vision_path 按服务器工作目录解析；原 submit 客户端从 JSON 文件加载时，
路径按 JSON 所在目录解析。URL 媒体沿用原 download 逻辑下载到本次请求的 inputs 目录。

### 必要的 Action128 扩展：extra.action128

| 字段 | 是否必需 | 含义 |
| --- | --- | --- |
| `action_domain` | 必需 | `/domains` 中完整的 source domain；确定 projection 与 source normalization 两套 ID |
| `action_dim_mask` | 必需 | 发布数据的128个有效槽位 bool，至少一槽有效；不能使用 normalization_mask 替代 |
| `action_fps` | 视频/动作频率不同时必需 | 独立动作频率；省略回退原 fps；时间戳间隔为1/action_fps |
| `selected_roles` | 多相机时必需 | canvas 中真实相机角色顺序；单相机默认 `["main"]` |
| `camera_layout` | 多相机时必需 | 与训练一致的 `[role,location,width,height]`，必须描述输入 canvas 的实际像素几何；单相机为null |
| `task_title` | 可选 | 保留发布 caption 的标题；可直接使用 JSON prompt 中的 task_title，未提供时回退样本名 |

多相机输入沿用原 `vision_path`，由客户端准备拼接观测。两相机排列是 top/bottom；
三相机是 top/bottom_left/bottom_right。拼接顺序、裁剪与接缝应与训练数据一致，
包括黑色分隔线。服务校验 camera_layout，并在 image_size 缩放时更新 seam 几何。
`action128_seam` 位置编码读该真实布局，不按 latent 尺寸猜测接缝。

### 请求示例（原格式增加 extra.action128）

以真实 Libero 两相机观测为例：640×960 的 canvas，上下区域各640×480，
20Hz 视频、20Hz 动作、2秒窗口，输出40行动作和41帧视频：

```json
{
  "name": "libero_action128_example",
  "model_mode": "wam",
  "prompt": "Pick up the cup and place it on the plate.",
  "vision_path": "/shared/observations/current_stitched.png",
  "domain_name": "libero",
  "image_size": 960,
  "action_chunk_size": 40,
  "fps": 20,
  "num_frames": 41,
  "seed": 42,
  "num_steps": 35,
  "guidance": 1.5,
  "shift": 5,
  "extra": {
    "action128": {
      "action_domain": "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_franka_link8_rz90_eef10_to_action128_v1",
      "action_dim_mask": [
        true, true, true, true, true, true, true, true,
        true, true, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false
      ],
      "action_fps": 20,
      "selected_roles": ["auxiliary_0", "left_wrist"],
      "camera_layout": [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    }
  }
}
```

动作50Hz、视频15Hz、2秒时，原字段设置 `action_chunk_size=100`、`fps=15`、
`num_frames=33`，扩展设置 `action_fps=50`；不再引入顶层 duration_seconds/action_fps。
示例 mask 对应本次 Libero 样本的前10槽有效；其他来源必须使用其实际发布 mask。
相同 embodiment 的两个 source domain 可能使用同一 projection head，却有不同统计量，
必须使用与原发布数据一致的完整 action_domain 与 slot mask。

### 原 submit 客户端

把已填好真实 mask/path 的请求保存为 wam_action128.json，原客户端命令不变：

```bash
python -m cosmos_framework.inference.ray.submit \
  -i wam_action128.json -o outputs/action128_client \
  --host 127.0.0.1 --port 8000 --timeout 600
```

也可以直接 POST：

```python
import json
from pathlib import Path
import requests
from cosmos_framework.inference.common.args import SampleOutputs

url = "http://127.0.0.1:8000"
request = json.loads(Path("wam_action128.json").read_text())
response = requests.post(f"{url}/generate", json=request, timeout=600)
response.raise_for_status()
result = SampleOutputs.model_validate(response.json())
assert result.status == "success", result.message
action = result.outputs[0].content["action"]
print(len(action), len(action[0]))
for remote_file in result.outputs[0].files:
    file_response = requests.get(f"{url}/outputs/{remote_file}", timeout=600)
    file_response.raise_for_status()
    Path(remote_file.name).write_bytes(file_response.content)
```

### 返回结构保持原样

```json
{
  "args": {"name": "libero_action128_example", "model_mode": "wam", "...": "解析后的原参数"},
  "status": "success",
  "message": null,
  "stack_trace": null,
  "outputs": [{
    "content": {
      "action": [["T×128，raw physical/source units"]],
      "action128": {"action_domain": "...", "action_units": "raw_physical_or_source_units", "...": "诊断信息"}
    },
    "files": ["generate_<id>/vision.mp4"]
  }]
}
```

`outputs[0].content.action` 是原读取位置，动作已按 checkpoint 的 source transform
反归一化；仅 robust-quantile 槽位执行 `(normalized-bias)/scale`，其余保持来源单位，
mask=false槽位归零。不会再改变 gripper、姿态坐标或绝对/相对动作约定。

新增 `outputs[0].content.action128` 包含 normalized_action、action_domain、action_dim_mask、
action_fps、action_timestamps_seconds、seed、checkpoint_path、elapsed_seconds，供诊断；
原读取 action/files 的客户端可以忽略它。elapsed_seconds 包含采样、VAE解码和视频保存。
视频文件通过原 `/outputs/` 下载。slot布局和控制量选择仍遵循发布数据合同，
接口/数值通过不能替代机器人闭环任务评测。

### 错误

- 422：请求、Action128 extension、domain/mask/布局/窗口/媒体等不合法，生成前拒绝。
- 404：model不存在，或使用已撤换的独立 `/predict` 路径。
- 503 + Retry-After：副本正在执行，可待当前请求结束后重试。
- GPU/OOM/collective异常：服务退出，torchrun终止其余rank，避免在失步进程组继续服务。
- 首次权重加载完成前不监听HTTP。

服务默认绑定0.0.0.0，无鉴权；本机调用可设置 SERVING_HOST=127.0.0.1。

## 4. 验证与复现

Action128 的 `raw_action_dim=128` 只表示容器宽度；有效槽由请求的
`action_dim_mask` 决定。服务将该布尔 mask 传入模型，在初始噪声、每次网络输入和
预测 velocity 中清零无效槽。固定步采样重新加噪后也将这些槽固定为零。
这个约束独立于训练的 `action_loss_mask`，不能用最终返回值归零代替。

针对这一约束的 CPU 回归使用真实打包、模型推理方法及 UniPC/固定步采样器，
检查逐步状态、稀疏槽、有效槽数值、条件动作、请求隔离与旧宽度裁剪兼容性：

```bash
LD_LIBRARY_PATH='' python -m pytest \
  cosmos_framework/model/generator/action128_inference_mask_test.py \
  cosmos_framework/inference/action_serving_test.py -q --num-gpus=0
```

```bash
LD_LIBRARY_PATH='' python -m pytest cosmos_framework/inference/action_serving_test.py -q --num-gpus=0
LD_LIBRARY_PATH='' python tools/validate_action128_serving.py \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --request-file wam_action128.json \
  --gpu-counts 1 2 4 8 \
  --output-dir outputs/action_serving_acceptance
```

CPU验证覆盖原类型/from_files加载、HTTP请求/返回、source与projection映射、帧布局和时间戳、
反归一化、串行准入、2/3 rank Gloo请求顺序、不同卡数launcher和单卡torchrun store回归。
真实线上 iter_000006000 的保存配置和小型DCP contract tensor校验已通过。

2026-10-12 使用实际 `iter_000006000`、真实 Libero episode 518 的预拼接观测与发布
mask，最终代码通过原 `cosmos_framework.inference.ray.submit.submit` 客户端在
1/2/4/8 张 H100 80GB 上完整加载、生成并下载视频：

| GPU 数量 | 启动就绪（秒） | 本次生成含视频保存（秒） | 最繁忙卡采样显存（GiB） |
| --- | --- | --- | --- |
| 1 | 45.04 | 26.58 | 43.35 |
| 2 | 42.04 | 30.89 | 34.16 |
| 4 | 40.04 | 31.45 | 27.32 |
| 8 | 45.04 | 31.36 | 26.24 |

条件：SDPA、CP=CFGP=1、35 steps、guidance=1.5、shift=5、seed=42、2秒、20Hz。
所有返回动作 `[40,128]` 有限、无效槽位归零、source反归一化及时间戳检查通过；
视频为640×960、41frames、20FPS。四种部署相同seed的动作最大差为0。
显存是每0.5秒采样nvidia-smi得到的观测最大值，非torch allocator精确峰值；
rank0负责视频解码，其显存高于其他rank。此分片方式主要降低每卡权重显存，不改善本例延迟。
这些数据仅适用于本次观测/参数，不是机器人任务成功率，也不覆盖其他分辨率/频率的峰值。

CPU最终29项测试通过，ruff/format及定向pyrefly检查通过。最初独立协议和统一协议
中间结果单独留档，最终原客户端验收结果为
`plans/v0.3.0b_action_serving/gpu_validation.json`。

验收命令可追加 `--original-client`，直接使用未修改的原ray.submit客户端。
工具只停止自身进程组，完成后已释放验收GPU；使用前确保GPU空闲。
多节点NCCL与其他注意力后端尚未实测。完整过程、决策和问题见该目录的 `record.md`。

## 6. Action128 工程优化部署 profile（2026-10-15）

单卡可容纳本轮 1B action tower checkpoint，优先部署**一张卡一个独立 HTTP 副本**。
副本之间不共享请求或 UND 状态；单副本仍以非阻塞锁拒绝并发请求（503），
调用端应选择空闲副本，避免积压过期观测。现阶段未提供自动负载均衡器。

- `latency` 默认组合：单 GPU，已默认开启 `--attention-backend flash3 --compile --reuse-small-action-metadata`，请求显式设置 `extra.action128.return_video=false`。
- `throughput`：在不同 GPU 上启动同一 profile 的独立副本，每个副本有独立端口及 output_dir；分发独立观测。卡数增加的是并发容量，单请求延时须单独测量。
- 兼容参考：显式传入 `--attention-backend sdpa --no-compile --no-reuse-small-action-metadata` 使用原 SDPA/eager 组合。默认 `return_video=true` 保留视频解码、文件保存和 `SampleOutputs`；动作直返仍计算模型中的 VIDEO 分支，仅省略最终视频解码和保存。

`--reuse-small-action-und` 目前只支持单卡副本，启动多 rank 时显式报错。
`--fuse-qkv` 仅为实验 opt-in，本轮出现多窗口累计误差超过工程门槛的情况，
不纳入推荐组合。UND 缓存仅在一次采样内复用，cond/uncond 分离；每个新请求重新建立缓存，
不按相似图像或文本命中跨请求视觉缓存。QKV buffer 在 checkpoint 加载后生成，
仅用于只读 inference 权重；模型权重变化后必须重启服务。
`/healthz` 的 `ready` 只表示模型加载完成，**不表示固定 shape 已预热**。
上线前使用目标相机布局、caption 长度和 horizon 发真实 `/generate` 请求预热；
首个请求可能包含编译和 Graph 捕获，须与稳态延时分开统计。

同节点两个独立副本示例（在两个 shell 分别执行，GPU 和端口对应更改）：

```bash
export LD_LIBRARY_PATH='' COSMOS_TRAINING=0 PYTHONPATH=. OMP_NUM_THREADS=1
# 直接 python 启动独立副本时，清除外层训练 shell 遗留的 distributed 环境；
# 正常 torchrun 启动时保留由 torchrun 设置的环境。
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE TORCHELASTIC_USE_AGENT_STORE
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m cosmos_framework.scripts.action128_server \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --host 127.0.0.1 --port 18104 --output-dir outputs/action_infer_opt/serving/gpu4
# 第二个 shell 使用 CUDA_VISIBLE_DEVICES=5、port=18105、output-dir=.../gpu5。
```

服务端 profile 名称是本文部署建议，并非新增 CLI `--profile` 参数。
Graph 开关表示启用编译器的 CUDA Graph 支持，实际捕获需另以 profiler 确认；
不能仅凭该开关声称所有算子都在 Graph 中执行。

HTTP 验证工具连接两个已经运行的 loopback 服务，不自行启动/终止子进程：

```bash
LD_LIBRARY_PATH='' .venv/bin/python tools/benchmark_action_serving_http.py \
  --requests outputs/action_infer_opt/serving/request_0.json outputs/action_infer_opt/serving/request_1.json \
  --ports 18104 18105 --output-dir outputs/action_infer_opt/serving/validation \
  --summary plans/v0.3.0d_action_infer/assets/opt_serving_summary.json
```

两个 request JSON 必须来自真实不同窗口的当前观测，显式动作直返，
保留实际相机 canvas、caption、domain、slot mask 和 action fps；不能使用随机图像代替。
本轮输入来自 `assets/input.json` 的 Libero 窗口 index 0/1，seed 42/43，
640×960 两相机布局、41帧/40行动作、20Hz、5步、guidance=1.5。
原始首帧、请求和完整 HTTP 响应保存在 `outputs/action_infer_opt/serving/`。
工具检查每个副本 A/B/A 重放、有限值、无效槽清零、domain/mask/时间戳、
默认视频下载以及双副本并发；跨副本工程数值门槛独立记录为 normalized
有效槽 `max_abs<=0.05, RMSE<=0.01`，不将 HTTP 契约通过等同于数值或闭环质量通过。

2026-10-15 的 H100×2 短测：HTTP 合约与两副本各自 A/B/A 逐位重放通过，
本地 HTTP 动作直返三次稳态约 2.29s；默认视频约 4.53s，并成功下载 MP4。
两副本预热后并发处理不同窗口 wall time 为 2.31s。
这些数字来自实验性的含 QKV 融合组合，仅证明服务能力；该组合在窗口 B 的
跨副本 normalized 有效槽误差为 max_abs=0.17135、RMSE=0.03747，
超过预先固定工程门槛，**不能作为已通过数值验收的部署推荐**。
不同 shape 首次请求分别出现约 21–22s 和 9s 的编译/捕获等待。
CPU 服务合约回归为 31 passed；GPU 短测没有覆盖闭环成功率、持续负载、
10,000 请求 p99 或控制器 deadline。
HTTP 时间包含准备输入、传输/序列化和文件操作；不能与离线 sample_seconds
直接计算同口径收益，baseline→最终组合的工程收益以主报告的配对离线测量为准。

补充独立 baseline 诊断：GPU4/5 各自仅启用 FA3+compile 时，窗口 A 跨副本
normalized 有效槽 max_abs=0.02517、RMSE=0.00775，窗口 B 为逐位一致。
同一窗口 B 的 Graph+UND（不融合 QKV）与 baseline 比较仍为
max_abs=0.12737、RMSE=0.01417，未通过门槛；因此不能将这些优化候选的
窗口 B 失败简单解释为 baseline 的独立编译波动。UND/Graph 组合保留实验 opt-in。
新增 `--reuse-small-action-metadata` 仅复用请求内布局/mask/RoPE，
独立于 UND hidden/KV 复用，现已默认开启；显式关闭使用
`--no-reuse-small-action-metadata`，部署指纹会记录实际开关值。
HTTP 层的 34 个 CPU 回归覆盖 metadata/UND 四种开关组合的参数传递和动作直返契约；
是否采用该新候选仍以配对 GPU 数值结果为准。

### 显式启动预热（单卡）

在上述最终组合启动命令添加 `--warmup-input /absolute/path/warmup_requests.json`。
文件可为单个 Cosmos `/generate` 请求对象，或覆盖多个目标 bucket 的非空对象数组；
相对 vision_path 按该 JSON 所在目录解析。推荐请求显式动作直返，避免无必要的视频保存。
服务在开启 HTTP listener 前，对每个请求执行两次完整 prepare/predict；
`/info` 与 `serving_snapshot.json` 的 `warmup` 记录 completed、输入 SHA256、
各 bucket 的 shape/seed/guidance 和每次耗时。预热失败则启动失败，避免发布未验证的 ready。
未传此参数时保持原有加载完成即 ready 的行为。

此选项只预热声明的输入，不建立固定 shape admission，不拒绝后续新输入，
也不保证长文本/新布局不会触发重编译。两个独立副本应分别预热，
不能用另一个副本的健康状态或编译缓存命中替代本副本的实际生成验证。

最终 HTTP 收口采用 **Graph off + metadata-only + 显式动作直返**。
metadata-only 在 GPU5 的两窗口（含 seed43）各 2 warmup+3 repeats 与同卡
FA3+compile baseline 的 normalized 动作逐位一致。Graph+metadata 在独立加载的
HTTP 副本 B 出现 max_abs=0.06118、RMSE=0.01165，故 Graph 不进入 HTTP 推荐。
这与同一已加载模型下的离线逐步数值测试不同，不能将离线通过扩展为跨加载 HTTP 已验收。

已核对 startup warmup 与 HTTP 的请求对象、保存的 prompt/name/vision_path、
PNG SHA256、经实际 prepare_wam_request 构造的 caption 字符串及首帧像素均一致。
输出 Action128 metadata 与启动 warmup 状态新增 `conditioning_caption_sha256`，
便于核查实际服务条件；没有通过修改 caption 或减步绕过数值门槛。

### 最终 HTTP 验证结论

最终默认组合为 `flash3 + compile + metadata-only`，**Graph off、UND off、QKV off**；
动作直返由请求显式启用。GPU4/5 各自独立进程已通过真实 `/generate`，
两副本各自 A/B/A 逐位重放、mask/domain/时间戳、默认视频下载均通过。
GPU5 启动时先在主线程预热两个 bucket，随后 HTTP 工作线程的两窗口动作
与同卡 baseline 逐位一致；warmup/HTTP 的实际 caption SHA256 相同。
跨副本 normalized 有效槽 A max_abs=0.02344、RMSE=0.00721，
B max_abs=0.03862、RMSE=0.00834，均通过固定 0.05/0.01 工程门槛。
两个不同观测并发完成的 wall time 为 2.386s；这仅是短并发能力检查。

下表为**同一张 GPU5**、5步、CFG=1.5、同 checkpoint/观测/seed，
每种模式预热后 3 次独立 HTTP 请求的客户端中位数：

| 返回内容 | FA3+compile reference | 最终 metadata-only、Graph off |
| --- | ---: | ---: |
| **无视频，仅动作（WAM机器人控制主口径）** | **2.361s** | **2.331s** |
| 默认视频+动作（辅助数据） | 4.576s | 4.477s |

**主比较两侧均为动作直返：1.013×、延时降低约1.31%，没有显著新增提速。**
完整视频 reference → 最终动作直返的1.964×仅是辅助输出模式对比，不计入无视频WAM主口径优化收益。
无视频指不decode/编码/保存预测视频；WAM内部VIDEO分支仍参与动作预测。
请求发送的是已在服务器本地的 PNG 路径，不包含相机采集/上传或控制器执行。
这里 reference 是当前实现关闭优化开关的服务；历史 frozen baseline 的验证
以主报告为准，不能用本表替代。没有闭环质量、10,000 请求 p99 或 deadline 验收。

服务 CPU 最终回归 **38 passed**。结果入口是
`plans/v0.3.0d_action_infer/assets/opt_serving_summary.json` 顶层的
`final_recommended_profile`、`final_acceptance`、`final_vs_baseline`；
此前失败的候选明确收纳在 `experimental_history`。
`assets/opt_serving_nograph_final.json` 保存最终双副本原始摘要，
`assets/opt_serving_final.json` 是此前 Graph 候选失败记录，不能误作最终配置。
