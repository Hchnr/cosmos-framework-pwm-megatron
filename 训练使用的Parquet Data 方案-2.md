# 训练使用的Parquet Data 方案

## 任务类型

三大类训练任务：

```text
understanding:
  - image-to-text
  - image-to-text multi-turn (for interleave data)
visual_generation:
  - text-to-image (T2I)
  - text-to-video (T2V)
  - text-to-video with assistant reasoning
  - target-prefix text+image-to-video (TI2V/I2V)
  - target-prefix video-to-video continuation (V2V)
action:
  - text-to-video-to-action (T2V+Action/WAM)
```

数据统一采用 Qwen3-VL 的 instruct/chat template 语义，generation task 也包装成 `messages`。Parquet 物理上固定为 `id / type / dataset / data / provenance` 五列，不存在独立 `media` 列。下面为了可读性，将 `data` 和 `provenance` 列中的严格 JSON 字符串展开为 JSON 对象展示。

# 任务样例

以下样例展示推荐字段组织；具体 ID、路径和 metadata 以实际数据为准。

### 2\.1 `understanding`：image\-to\-text 普通 VQA

```json
{
  "id": "vqa_v2:000001",
  "type": "understanding",
  "dataset": "vqa_v2",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "/datasets/vqa_v2/images/000001.jpg",
            "width": 640,
            "height": 480
          },
          {
            "type": "text",
            "text": "What color is the car?"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "Red."
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "000001",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

### 2\.2 `understanding`：image\-to\-text Interleave

```json
{
  "id": "ego4d:000021",
  "type": "understanding",
  "dataset": "ego4d",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Observe the current state."
          },
          {
            "type": "image",
            "image": "/path/to/image1",
            "width": 640,
            "height": 480
          },
          {
            "type": "text",
            "text": "What action should happen next?"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "Place the cucumber in the bag."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Observe the current state after step."
          },
          {
            "type": "image",
            "image": "/path/to/image2",
            "width": 640,
            "height": 480
          },
          {
            "type": "text",
            "text": "What action should happen next?"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "Place the tomato in the bag."
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "event_window_21",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

### 2\.3 `visual_generation`: text\-to\-image

```json
{
  "id": "coco:000042",
  "type": "visual_generation",
  "dataset": "coco",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "{\"caption\": \"Man wearing a suit standing in front of a black car.\"}"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "gen_image",
            "gen_image": "/datasets/coco/images/train2017/000000000042.jpg",
            "width": 640,
            "height": 480
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "000000000042",
    "source_path": "/path/to/coco/data",
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

### 2\.4 `visual_generation`：纯 text\-to\-video（T2V）

```json
{
  "id": "action100m:000107",
  "type": "visual_generation",
  "dataset": "action100m",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "{\"caption\": \"A man is kicking a soccer ball on green grass.\"}"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "gen_video",
            "gen_video": "/datasets/video_gen/clips/000107.mp4",
            "width": 640,
            "height": 480,
            "num_frames": 160,
            "trim_head_frames": 10,
            "fps": 24.0
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "000107",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

该样例不显式写 `conditioning_latent_frames`，因此按 TOML 的 `conditioning_config` 权重在 T2V/TI2V/V2V 中采样。纯生成样本没有 assistant 文本，但仍使用 `<|im_end|>` 关闭空的 assistant turn；这个 `<|im_end|>` 只作为结构边界，text loss mask 仍全为 0，不计算 CE loss。逻辑序列为：

```text
<|im_start|>user
instruction
<|im_end|>
<|im_start|>assistant
<|im_end|>
<|vision_start|>
[VIDEO]
```

### 2\.5 `visual_generation`：T2V + assistant reasoning（同时计算 text/video loss）

```json
{
  "id": "action100m:000108",
  "type": "visual_generation",
  "dataset": "action100m",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Generate a video of a robot placing a red cup on a table. Explain the motion plan first."
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "The robot should approach the cup, grasp it, move above the table, and release it steadily."
          },
          {
            "type": "gen_video",
            "gen_video": "/datasets/video_gen/clips/000108.mp4",
            "width": 1280,
            "height": 720,
            "num_frames": 93,
            "trim_head_frames": 0,
            "fps": 24.0,
            "conditioning_latent_frames": 0
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "000108",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

assistant `content` 中的顺序就是训练序列顺序：先文本，再 `gen_video`。渲染后的逻辑序列为：

```text
<|im_start|>user
instruction
<|im_end|>
<|im_start|>assistant
reasoning / plan / description
<|im_end|>
<|vision_start|>
[VIDEO]
```

assistant 文本和结束它的 `<|im_end|>` 计算 CE loss；`<|vision_start|>` 只是模态切换 token，不计算 CE loss；video latent 计算 video loss。训练时先检查 text loss mask：全 0 则跳过 CE，非全 0 则计算 CE。

### 2\.6 `visual_generation`：target\-prefix TI2V/I2V

```json
{
  "id": "action100m:000109",
  "type": "visual_generation",
  "dataset": "action100m",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Continue the scene as the robot arm picks up the object."
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "gen_video",
            "gen_video": "/datasets/video_gen/clips/000109.mp4",
            "width": 1280,
            "height": 720,
            "num_frames": 93,
            "trim_head_frames": 0,
            "fps": 24.0,
            "conditioning_latent_frames": 1
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "000109",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

这里不需要额外保存 `source.jpg`。`gen_video` 的第 1 个像素帧同时就是 image condition，对应第 1 个 clean target latent；其余 target latent 计算 video loss。

### 2\.7 `visual_generation`：target\-prefix V2V continuation

```json
{
  "id": "action100m:000110",
  "type": "visual_generation",
  "dataset": "action100m",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Continue the motion while preserving the scene and camera trajectory."
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "gen_video",
            "gen_video": "/datasets/video_gen/clips/000110.mp4",
            "width": 1280,
            "height": 720,
            "num_frames": 93,
            "trim_head_frames": 0,
            "fps": 24.0,
            "conditioning_latent_frames": 2
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "000110",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

这里同样不保存 paired source video。`gen_video` 的前 5 个像素帧作为已知 continuation condition，对应前 2 个 clean target latent；后续 target latent 计算 video loss。

#### `gen_video` 字段说明

| 字段 | 含义 | 是否必填 |
|---|---|---|
| `gen_video` | 已物化的 target video 绝对路径。TI2V/V2V 也从该 target 的开头取 clean-prefix，不需要额外 `source` 路径。 | 是 |
| `width` / `height` | target 像素尺寸，用于后续 sequence length 与训练量估算。 | 否；data processor 能获取时应尽量填写 |
| `trim_head_frames` | 按解码顺序必须丢弃的文件头帧数；`0` 合法，缺失不等价于 `0`。 | 是 |
| `valid_frames` | trim 后按目标 fps 重采样得到的训练帧数；当前 Wan VAE 要求 `4n+1`，Edge 通常为 `93`。 | 是 |
| `fps` | 训练目标采样率，不是源文件播放帧率；例如源 60 fps、训练 24 fps 时按 PTS 约每 2～3 个源帧采一帧。 | 是 |
| `num_frames` | 文件含 trim 头部在内的总像素帧数，只用于审计，不作为训练输出长度。 | 否；能获取时应尽量填写 |
| `source_fps` | 源文件报告的帧率，只用于审计；实际重采样以解码帧 PTS 为准。 | 否；能获取时应尽量填写 |
| `conditioning_latent_frames` | 从 target 开头取多少个 clean latent 作为 DiT 条件：`0`=T2V，`1`=TI2V/I2V（第 1 个像素帧），`2`=V2V（前 5 个像素帧）。 | 否；缺失或 `null` 时走 TOML 采样 |

显式 `conditioning_latent_frames` 优先于 TOML 配置。不写或写 `null` 时，generation 样本按 TOML 的权重在 `0/1/2` 中选择：

```toml
conditioning_config = { "0" = 0.7, "1" = 0.2, "2" = 0.1 }
```

如果要强制按纯 T2V 训练，可以显式写 `"conditioning_latent_frames": 0`；这会 override TOML 采样结果，不使用 clean target prefix，全部 target video latent 都计算 video loss。如果 target 过短，有效 clean-prefix 应自动降级，至少保留 1 个需要预测的 video latent。

### 2\.8 `visual_generation`：直接读取预编码 VAE latent

当 target video 已离线编码为 Wan VAE latent 时，仍使用 `type: "gen_video"`，但用 `vae_latent` 替代像素视频字段 `gen_video`：

```json
{
  "id": "action100m:-t_n_U9btPM#734400-744410@w0",
  "type": "visual_generation",
  "dataset": "action100m",
  "data": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Generate a video that follows the described motion."
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "gen_video",
            "vae_latent": {
              "path": "/share/project/eai_pwm/pipeline_store/demo/vae_demo_10k/latents/action100m_wan2pt2_tokenizer-t480-20eb789667fa_v1/out/chunk_00001.safetensors",
              "tensor_key": "action100m/-t_n_U9btPM#734400-744410@w0",
              "latent_shape": [48, 61, 30, 52]
            },
            "fps": 24.0,
            "gen_video": null,
            "width": null,
            "height": null,
            "num_frames": null,
            "trim_head_frames": 0
          }
        ]
      }
    ]
  },
  "provenance": {
    "source_id": "-t_n_U9btPM#734400-744410",
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

`gen_video` content item 必须且只能选择一种 target 存储方式：

- 像素输入：写 `gen_video`，训练时执行视频解码、重采样、几何处理和 VAE encode。
- latent 输入：写 `vae_latent`，训练时直接读取 tensor 并传给模型，不再执行视频解码、重采样、resize、crop、VAE encode 或 latent normalization。

`vae_latent` 的必填字段：

| 字段 | 含义 |
|---|---|
| `path` | `.safetensors` 容器的绝对路径。Data Processor 应把产物索引中的 `shard_relpath` 解析为该字段，训练不依赖旁边的 `chunk_*.jsonl`。 |
| `tensor_key` | 容器内 tensor 的精确 key。一个 safetensors chunk 可以保存多条样本，因此 `path` 只定位容器，`tensor_key` 才定位容器中的目标 tensor。 |
| `latent_shape` | `[C, T, H, W]`。训练可直接用 `T×H×W` 估算 sequence length，无需先加载 tensor；当前固定 `C=48`、时间压缩倍率为 4。 |

latent dtype 由训练约定固定为 `bfloat16`，不逐行保存。像素等价帧数如后续需要，可由固定时间压缩倍率直接计算为 `4 × (T - 1) + 1`，因此不保存 `valid_frames`。空间尺寸已经体现在 `H/W` 中，训练不再做任何像素空间处理，因此不保存 `canvas_h/w` 或 `content_h/w`。VAE、代码版本和其他 lineage 信息由数据集整体管理，不逐行写入 Parquet。

`clip_id`、`window_index`、`record_id`、源视频信息和生产过程统计只用于审计，可放入 `provenance` 或不写，不是训练加载必需字段。外层 `id`、`dataset` 已提供稳定样本标识和数据集名称，因此不在 `vae_latent` 中重复保存。

直接读取 latent 不改变任务语义：`conditioning_latent_frames` 仍放在 `gen_video` content item 中、与 `vae_latent` 平级。缺失时按 TOML 权重采样；显式为 `0/1/2` 时分别强制 T2V、TI2V/I2V、V2V。assistant reasoning、`<|im_end|>` 的 CE mask 和 `<|vision_start|>` 的处理也与像素视频格式完全相同。

> **兼容性说明：** 本节示例暂时保留 `gen_video`、`width`、`height` 等旧字段，并按字段含义设置为 `null` 或 `0`，仅用于兼容现有 DataLoader，不作为 latent 输入的有效像素视频元数据。后续可移除 DataLoader 对 `valid_frames` 字段的依赖，将像素域帧数统一以 `num_frames` 表示；对于直接读取 latent 的样本，其时间维长度以 `vae_latent.latent_shape` 中的 `T` 为准，必要时按上述压缩倍率推导像素等价帧数。`fps` 字段仍须保留，用于 Cosmos 的 RoPE 时间位置编码及 `position_id` 计算。

### 2\.9 text\-to\-video\-to\-action 格式样例



Action 数据保留 LeRobot 原生目录，不把 episode、segment 或滑动窗口重新物化到本 Parquet 中。



一个 Action Parquet 代表一个逻辑数据集；其中一行描述一个 LeRobot root 及其数据解释方式。通常一个 Action Parquet 只有一行。如果同一个 LeRobot root 需要采用不同的 `chunk_length`、`sample_stride` 或过滤规则训练，可以生成多个 Action Parquet，并在各自的 `data` 中写入不同配置。



数据集之间的混合比例不保存在 Parquet 中，由训练入口根据多个 Parquet 路径单独配置。



```json
{
  "id": "droid:wam:train:v1",
  "type": "action",
  "dataset": "droid",
  "data": {
    "format": "lerobot",
    "adapter": "droid",
    "root": "/datasets/lerobot/droid",
    "mode": "wam",
    "fps": 15.0,
    "chunk_length": 32,
    "sample_stride": 1,
    "action_space": "joint_pos",
    "viewpoint": "concat_view",
    "use_state": true,
    "action_normalization": null,
    "use_success_only": true,
    "use_filter_dict": false,
    "filter_dict_path": null
  },
  "provenance": {
    "source_id": "droid",
    "source_path": "/datasets/droid/raw",
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```



`data` 字段说明：



|字段|含义|一般是否调整|
|---|---|---|
|`format`|底层存储格式。Action 当前使用 `lerobot`。|基本不调整。|
|`adapter`|选择数据集对应的解析和预处理代码，例如 `droid`、`droid_merged` 或 `libero`。同一种字段布局和 Action 定义的数据集复用同一个 adapter。|更换数据集类型时调整。|
|`root`|LeRobot 数据集根目录。目录内继续使用 LeRobot 自己的 `info.json`、`tasks.parquet`、`episodes.parquet`、逐帧 data parquet 和视频文件。|每个数据集必须设置。|
|`mode`|定义 Vision 和 Action 的 condition/target 关系。当前 text\-to\-video\+action 训练使用 `wam`：首个 Vision latent 和可选 initial state 为 condition，后续 Video 与 Action 都计算生成 loss。|当前任务通常固定为 `wam`。|
|`fps`|DROID 窗口内 observation/action 的目标时间采样频率。对于能够从 LeRobot metadata 读取原生 FPS 的 adapter，可以优先使用数据集原生值。|偶尔调整。|
|`chunk_length`|一个训练窗口预测的 Action step 数。通常对应 `T` 个 Action 和 `T+1` 帧 observation。|较可能调整。|
|`sample_stride`|相邻窗口起点间隔的帧数。`1` 表示每帧都作为一个新起点；增大后可减少长 episode 中高度重叠的窗口。|数据量较大或重叠严重时调整。|
|`action_space`|Action 的物理表示，例如 DROID 的 `joint_pos` 或末端执行器 pose。改变它会改变监督张量的含义和维度。|很少调整；改变时应视为新的数据集变体。|
|`viewpoint`|输入视频使用的相机视角，例如腕部视角、第三人称视角或 `concat_view`。LIBERO adapter 中对应字段可命名为 `camera_mode`。|偶尔调整。|
|`use_state`|是否把窗口起始时刻的机器人 state 放在 Action 序列开头作为 condition。|通常不调整。|
|`action_normalization`|Action 数值的归一化方式；`null` 表示不在 dataset adapter 中归一化。|通常不调整。|
|`use_success_only`|是否仅使用标记为成功的轨迹。主要用于支持 success/failure 划分的 adapter。|数据筛选策略变化时调整。|
|`use_filter_dict`|是否启用额外的 episode 区间过滤；与现有 Cosmos DROID factory 参数保持一致。|只有使用清洗区间文件时调整。|
|`filter_dict_path`|可选的区间过滤文件。用于只保留长 episode 中的有效区间；`null` 表示不额外过滤。|重新清洗数据时调整。|



不同 adapter 可以拥有少量专用字段。例如 LIBERO 通常还需要：



```json
{
  "format": "lerobot",
  "adapter": "libero",
  "root": "/datasets/lerobot/libero_10",
  "mode": "wam",
  "chunk_length": 16,
  "sample_stride": 1,
  "camera_mode": "concat_view",
  "image_size": 256,
  "action_space": "frame_wise_relative",
  "rotation_space": "6d",
  "pose_coordinate_frame": "native",
  "action_normalization": "quantile_rot",
  "action_stats_path": null
}
```



其中 `rotation_space` 和 `pose_coordinate_frame` 决定位姿 Action 的旋转表达与坐标系；`action_stats_path` 指向可选的外部 Action 归一化统计文件。没有自定义统计文件时置为 `null`。



Action Parquet 不保存 `messages`。训练时 adapter 从 LeRobot 的 `tasks.parquet` 读取任务文字，并动态构造与其他任务一致的 instruct chat template。对于 `wam`，逻辑上的 assistant 目标同时包含 `gen_video` 和 `gen_action`，但这些目标不作为普通文字 token 写入 Parquet。

## 核心数据结构

通用 Row 外层：

```json
{
  "id": "dataset:stable-sample-id",
  "type": "understanding",
  "dataset": "ego4d",
  "data": {},
  "provenance": {}
}
```

字段职责：

其中：

```text
type    = 如何训练
dataset = 数据来自哪里
```

同一 `type` 可以包含多个不同 dataset。

## Provenance 原则

当前训练读取的是物化后的独立 clip；原始长视频信息仅用于追溯。

```json
{
  "provenance": {
    "source_id": "video_8f3a",
    "source_path": "/datasets/ego4d/raw/video_8f3a.mp4",
    "source_start_sec": 10.0,
    "source_end_sec": 15.0
  }
}
```

没有来源信息时显式置空：

```json
{
  "provenance": {
    "source_id": null,
    "source_path": null,
    "source_start_sec": null,
    "source_end_sec": null
  }
}
```

`source_start_sec/end_sec` 不参与当前媒体读取，也不替代 target 或 condition 的物化媒体文件。

## Parquet、Annotation 与校验原则

Parquet 约束：

- 使用严格 UTF\-8 JSON。

- 不允许 NaN、Infinity 等非标准 JSON 值。

- 一个 snapshot 内使用统一 codec。

最低校验要求：

- ID 全局唯一。

- JSON 可正确 decode。

- `type` 与 `data` 字段匹配。

- path 是合法绝对路径，且指向预期媒体类型。

- 视频 path 指向独立物化 clip。

- LMDB 内不存在媒体 bytes。

- provenance 缺失字段显式为 `null`。

- 多 worker 和 checkpoint resume 的遍历顺序稳定。
