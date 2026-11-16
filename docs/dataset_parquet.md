# Unified training Parquet catalogs

New training data sources use an exact five-column schema:

| Column | Meaning |
| --- | --- |
| `id` | Stable sample or dataset-catalog identifier. |
| `type` | `understanding`, `visual_generation`, or `action`. |
| `dataset` | Logical source name used for retries and observability. |
| `data` | Strict JSON object. Understanding/T2V store `messages`; Action stores one dataset-catalog object. |
| `provenance` | Strict JSON source/revision metadata. |

There is no top-level `media` column. Image/video paths and their metadata live
inside `data.messages`. Online Action rows point at a LeRobot root and do not duplicate
episode rows. Offline DROID Action uses one row with
`data={"adapter":"droid_offline","manifest_path":"/absolute/path/manifest.json"}`;
the manifest maps observation windows to latent shards and raw action metadata.
See [Offline VAE preparation](action_policy_1b.md#offline-vae-data).
Parquet producers must use at most 2,048 rows per row group.
Ordinary image/video decode failures are logged and dropped; schema, chat grammar,
dimension, and Action feature errors fail immediately.

Configure one catalog per dataset and put ratios in the training recipe:

```toml
[[dataloader_train.datasets]]
path = "/path/to/catalog.parquet"
ratio = 3

[[dataloader_train.datasets]]
path = "/path/to/another.parquet"
ratio = 2

[dataloader_train.media]
decode_timeout_sec = 120
max_sampled_frames = 256
max_frame_pixels = 16777216
max_window_bytes = 1073741824
max_decode_frames_per_seek = 2048
parquet_handle_cache_size = 8
video_handle_cache_size = 8

[dataloader_train.visual_generation]
# Used only by text-only gen_video rows that omit conditioning_latent_frames.
conditioning_config = {"0" = 0.7, "1" = 0.2, "2" = 0.1}

[model]
# Required when visual-generation rows may contain assistant text.
predict_text_tokens = true
text_loss_weight = 1.0
```

Positive integer ratios use the exact periodic schedule: `3:2` contains exactly
three batches from the first source and two from the second in every five-batch
cycle. The cycle order is deterministically shuffled from the recipe seed.

## Task grammar

- Understanding calls the Qwen3-VL native chat template. Assistant content and
  its terminating `<|im_end|>` receive text CE; role/template/media-placeholder
  and modality-control tokens are masked.
- Pure T2I/T2V closes the empty assistant turn with an unsupervised
  `<|im_end|>`, then puts the encoded `<|vision_start|>` before the DiT visual
  tokens. Its text target mask is empty, so no CE kernel is run.
- When a T2I/T2V assistant message contains a `text` item before `gen_image` or
  `gen_video`, the exact order is assistant prefix, assistant text,
  supervised `<|im_end|>`, unsupervised `<|vision_start|>`, then DiT tokens.
  Set `model.predict_text_tokens=true` to enable this optional LM-head loss.
- Understanding and visual generation share the same full Qwen3-VL processor.
  A user `image`/`video` is retained as VLM context and produces native media
  placeholders plus vision tensors; it is independent of DiT target-prefix
  conditioning. A multimodal user prompt whose assistant target omits
  `conditioning_latent_frames` therefore remains a zero-prefix T2V target.
- T2V+Action uses one generation region containing video then Action tokens with
  no VLM delimiter between them. Video and Action diffusion losses are enabled;
  text CE is disabled.

## Target-prefix T2V / TI2V / V2V

Training uses the beginning of the target itself; it does not store a paired
source video or train arbitrary source-to-target editing:

```json
{
  "type": "gen_video",
  "gen_video": "/data/target.mp4",
  "width": 1280,
  "height": 720,
  "num_frames": 93,
  "fps": 24,
  "conditioning_latent_frames": 2
}
```

`conditioning_latent_frames=0` is T2V, `1` conditions the first pixel frame (TI2V/I2V), and `2`
conditions the first five pixel frames (V2V continuation). Explicit values
override the TOML mixture. For text-only rows, an omitted value is sampled from
the TOML mixture once per epoch with deterministic resume. For rows containing
user media, omission means zero target-prefix latents. If a target is too short,
the effective prefix is reduced to leave at least one generated latent.

`width`, `height`, `num_frames`, and `fps` are optional producer metadata for
future token/epoch-size estimation. The loader does not use them to choose the
task and producers need not probe media solely to fill missing values.

## Canonical DROID

New DROID roots are recognized by the canonical feature signature in
`meta/info.json`, not by directory basename. Required fields are:

```text
observation.image.wrist_image_left
observation.image.exterior_image_1_left
observation.image.exterior_image_2_left
observation.state.joint_positions
observation.state.cartesian_position
observation.state.gripper_position
action.joint_position
action.gripper_position
```

Legacy named Cosmos profiles remain supported. New datasets should be converted
to this signature rather than adding aliases in the training loader.
