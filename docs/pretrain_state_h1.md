# Pretraining with one current State128 observation

The generic reader supports mixed MP4 and indexed JPEG TAR episodes. JPEG views
declare `storage="indexed_jpeg_tar_v1"`; the dataset publishes a uint64 offset
index, SHA/shape contract in `meta/image_storage.json`, and a matching passed
`training_validation.json`. JPEG reads use the normal bounded decode helper and
nearest source frames, including when the requested sampling rate differs from
the camera rate. Action timestamps remain independent. No dataset-specific
sampler is required. RoboDojo and RoboCasa365 use pretraining-owned media paths;
historical SFT paths in provenance are not runtime inputs. Episode metadata must
be published under `meta/episodes/chunk-*`, alongside the normal numeric chunks.

Based on upstream `main` at `8ec9484b`, including the shared-prefetch dataloader.
This does not import or merge the older one-second-history branch.

```toml
[dataloader_train]
action_use_state = true
first_frame_color_jitter = true
num_workers = 4
prefetch_factor = 4
```

`None`/omitted follows each catalog's `use_state`; `false` disables State.
The State override applies to the unified Action128 reader only. Existing no-State
recipes and their fingerprints remain unchanged. No source Parquet is rewritten.

## Contract

- Read the episode-local observation at or immediately before window start.
  This is **one frame**, not one second. Never cross an episode or use future State.
- Read `observation.state` and `observation.state_mask`, both fixed-width 128.
  An all-masked current observation produces **no token** and does not fall back
  to an older observation. Keep the video/Action training sample unchanged;
  do not shift the window to obtain a future valid State. Partial invalid
  dimensions are zeroed. This applies to entire no-State sources as well.
- Require the published `meta/state_normalization.json` reference. Apply its
  fixed per-domain affine transform and clipping; never fit statistics or use
  Action's transform for State. Missing/malformed contracts fail explicitly.
- Prepend State to the Action stream. EEF absolute State uses input-only slot 32,
  independently of EEF delta Action slot 0. Joint absolute State and Action share
  the existing embodiment slot. Domain IDs can therefore vary per token. State is clean,
  gets no diffusion noise, has no loss, and is excluded from output projection.
  No parameter names, head assignments, or checkpoint tensor shapes change.
- Use real relative timestamps and existing `seconds * base_fps / VAE_temporal_ratio`
  RoPE. State(t0) and Action(t0) may share a time; duplicates within the action
  stream are still rejected. Video and Action windows/stride are unchanged.
- State consumes one Action token of packing budget when present. The enabled
  policy and State normalization content enter the dataset fingerprint. Rebuild
  the dataloader cursor when enabling it; do not claim exact no-State resume.

## Configurations and checks

### 52-source local acceptance configuration (2026-10-29)

`examples/toml/sft_config/action128_pretrain_state_h1_52src_8gpu.toml`
uses all 52 catalog paths and integer ratios from the supplied
`Action128_完整采样配比_20261029.xlsx`, sheet `全部catalog明细`, rows 7–58.
Ratios sum to 10,000 and select packed sequences, not raw windows.
The paired launcher is `examples/launch_pretrain_state_h1_52src_8gpu.sh`.

- Local 8 GPUs, GA1, four shared workers, prefetch four.
- Packing: UND 11,776 / GEN 62,592 / ACTION 2,944; at most 92 samples,
  with admission limited by the token budget.
- One current State when valid; missing State keeps the window without a State
  token. Ordinary first-frame ColorJitter is enabled.
- Model-only warm start from `pretrain23src_iter17200_eef_state32_from0`;
  no old optimizer or dataloader cursor. All outputs use mixed storage.
- Acceptance defaults: 12 steps, save at step 12, keep one checkpoint,
  W&B disabled, decoder and VAE compilation disabled. This is not a long-run recipe.
- All 52 sources passed real sample decoding and State/Action mask checks.
  Window-uniform audit: 128 windows per source, 6,656 in total. Only RoboTwin
  Joint had missing State (128/128); the other 51 sources had 0/128 each.
  This is a sample audit, not a full-data absence guarantee.
- Local 8-GPU acceptance completed 12 optimizer steps and saved a full DCP at
  `iter_000000012`. All observed losses/gradient norms were finite; rank-0 final
  loss was 0.2026. Typical non-save steps took 14–18 seconds, with one 32-second
  step; final synchronous save brought step 12 to about 200 seconds.
  Focused regression suite: 81 passed. These checks establish execution, not
  convergence or model quality across the new mixture.

Use a new `OUTPUT_ROOT` for another warm-start test. Reusing the same output
directory may auto-resume its checkpoint instead.

`examples/toml/sft_config/action128_wam_nano_state_h1.toml` preserves the main
23-source recipe and their relative sequence ratios, adding State and using mixed-storage catalog
paths. A 24th source, `action_tactile_pretraining_20261026/catalogs/legacy_unified_action128.parquet`,
adds the two-second Joint/State data without tactile, with user-selected ratio 526.
Its sequence share is 526/10526 (approximately 5%); the original ratios still sum
to 10000. Shared worker/prefetch remain 4/4. Use a new output/data cursor when
switching from the 23-source mixture. Set checkpoint/output paths explicitly before a production launch.
Launch with `examples/launch_pretrain_state_h1_96gpu.sh`.

`examples/launch_pretrain_state_h1_smoke.sh` runs twenty updates on eight GPUs with
DROID EEF, RoboDojo Joint, AgiBot Joint, Galaxea Joint, FastUMI EEF and LIBERO EEF
pretraining catalogs, four shared physical workers, prefetch four,
two samples per pack, and the slot-32 initialized iter17200 model. Optimizer/cursor
start fresh (`warm_start`); fixed lengths, language compilation and full-canvas
VAE AOT are enabled.

The runtime must expose both `/share/mixed/projset` and `/share/mixed/project`
as aliases of the matching `/share-mixed` mounts: published video symlinks use
both prefixes. Both launchers use `examples/_pretrain_state_paths.sh` to create
missing aliases and check existing paths without replacing data.

The default model-only checkpoint supplies its contract via the original
`config.yaml` beside `model/`; complete checkpoints prefer trainer metadata.

Ordinary first-frame ColorJitter is enabled in both new recipes: brightness,
contrast and saturation factors in [0.8, 1.2], hue in [-0.05, 0.05]. Only the
current RGB condition image is modified; future target frames and numerical
State/Action are unchanged. There is no stronger mode. The same parameters apply
to the composed camera canvas and the four-pixel black seams are restored.
Its configuration is included in the stream fingerprint. Other recipes retain
their existing behavior unless the augmentation is explicitly enabled.

With the launcher's model/VAE environment variables set:

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. python tools/validate_pretrain_state_h1.py
LD_LIBRARY_PATH='' python -m pytest cosmos_framework/data/test/pretrain_state_test.py
```

The validation reads actual EEF/Joint samples, traverses the shared loader,
and checks clean/loss masks and the exclusion of State from prediction indexes.

## Validation on 2026-10-28

- All 23 production catalogs passed State normalization contract loading.
- Six real sources passed direct sample and shared-loader traversal (12 packed
  batches), including first-frame-only augmentation assertions.
- State and ColorJitter tests: 16 passed. Shared-prefetch/resume and State tests:
  58 passed. The broader Action/model test selection had 155 passes and one
  pre-existing assertion expecting two workers in the main production recipe,
  which currently specifies four; that unrelated test/config was not changed.
- Eight-GPU forward/backward smoke uses `172.24.232.11`. Logs and checkpoints:
  `/share-mixed/projset/liyue/model_ckpts/pretrain_state_h1_main_6src_jitter_smoke_20261028/`.
  This test uses two samples per pack, not full production packing capacity.
- Before splitting EEF State routing into slot 32, completed all 12 updates and exited successfully. All six sources trained
  twice. Rank-0 loss stayed finite (0.1062–0.3470, final 0.1358); post-initialization
  steps took 19–24 seconds excluding checkpoint save. Model/optimizer/scheduler,
  trainer and dataloader state saved to `iter_000000012`; saving took 206 seconds.
  This validates the training path, not convergence or downstream task quality.
- After splitting EEF absolute State (32) from delta Action (0), repeated the
  six-source sample/packed-loader checks successfully. Joint State/Action retain
  their embodiment slot. The previous GPU smoke is not validation of this routing
  update; model input/output routing is covered separately by unit tests.

- Final slot-32 validation used the production launcher/TOML and default checkpoint:
  3 GPUs, 20 updates, fixed UND4096/GEN12288/Action1024, GA1, one sample per pack,
  short-UND disabled. State, ColorJitter, static language compilation and full VAE
  AOT stayed enabled (32/32 variants per rank). Training exited 0 and saved step 20;
  trainer iteration and model contract digest passed readback checks.
  All 23 sources/ratios were retained; 9 sources were sampled. Production
  96-GPU/74368-token/GA3 capacity, full packing and dual UND buckets remain untested.
  Command/results: `outputs/pr76_gpu_validation/run_compiled.sh` and
  `outputs/pr76_gpu_validation/compiled_result.json`.
