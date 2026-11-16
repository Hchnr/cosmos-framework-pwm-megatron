# Six-source Ego R2 training

The six published catalogs use the existing `unified_action128` reader. No
source-specific dataset, rewritten numeric Parquet, or slow-motion video copy
is required. The framework does not repair invalid publication metadata.

## Contract

- Action head **63**, current-State input head **62**; six independent
  normalization domains. State and Action use their own published transforms.
- Numeric timestamps, Action, State and captions are already on the published
  slow-motion timeline. Do not resample these or multiply their values by two.
- Camera `fps` is the effective published FPS, independent of the 30 Hz numeric
  grid. Do not halve it a second time.
- Video queries first locate the nearest published numeric row. Source time
  priority is explicit `pts_column`, frozen main-camera `source_media_pts`, then
  `from_timestamp + row_timestamp * time_scale`. Legacy cameras without a
  mapping retain their existing direct timestamp behavior and scale 1.
- Subtract `source_clock_origin_s` once, then snap to the declared frozen
  timeline, checking half-frame tolerance. Pad queries repeat the final valid
  query inside the episode. Numeric tail loss masks retain `pad_mask` semantics.
- Native JSON is expanded before non-null structured fields override it.
  MCAP topics and Zarr stream keys are not discarded during view selection.

`media_timeline.py` owns time/frame mapping; `native_video.py` provides MCAP and
Zarr decoding inside the existing timeout-killable media helper. MCAP uses its
indexed topic log timestamps and integer nanosecond origin, bounded H.264 GOP
pre-roll, and nearest-frame selection. Zarr-v3 JPEG arrays are read by native
frame index directly from directories or TAR members, including indexed shards
with CRC32C checks. Caches and decompression/request sizes are bounded.

No per-topic native frame index is declared for the MCAP views in the handoff.
The reference case file repeats the main-camera `source_frame_id` for them;
that ID is not used for MCAP decoding. Each topic is selected independently by
its epoch time. Validation records both the unused reference ID and actual
decoded time rather than claiming a main-camera ID identifies another camera.

## Validation commands

Activate the training environment, set `LD_LIBRARY_PATH=''`, and run from this
repository. Dependencies include `mcap`, `mcap-protobuf-support`, `zstandard`
and `google-crc32c` in the training extra.

The training extra and lockfile include these packages. Existing environments
still need to install the new native-media dependencies if they were created
before this update. Install them explicitly in the training environment when
needed:

```bash
python -m pip install mcap mcap-protobuf-support zstandard google-crc32c
```

```bash
PYTHONPATH=. python tools/validate_ego_media.py \
  --reference-code /share-mixed/project/liyue/Project/action-preprocess \
  --output outputs/ego6_validation/reference_cases.json
```

The reference-code option is only for validation. Training has no dependency on
the preprocessing repository. This command checks all 19 published cases,
including first/middle/last episodes and nonzero cut boundaries: per-view
mapped times/native indexes, actual pixels against the reference reader,
initial and final partial-window decoding, VAE-compatible temporal lengths,
and the normal multi-camera canvas.

```bash
bash examples/launch_pretrain_state_h1_ego6_8gpu.sh
```

The paired `action128_pretrain_state_h1_ego6_8gpu.toml` uses all six original
catalogs, **8 GPUs / GA1 / 4 shared workers / prefetch 4**, current State h1,
first-frame ColorJitter and the existing full packing budgets. It runs 12 steps
and saves at step 12. Equal ratios are for source coverage testing, not a
proposed production mixture. The existing 52-source production weights are not
changed. Model-only warm start uses the existing iter17200 State32 backup;
new Ego heads are taken from that checkpoint, not copied at runtime.

Default training output:

```text
/share-mixed/projset/liyue/model_ckpts/pretrain_state_h1_ego6_8gpu_20261030
```

Set a new `OUTPUT_ROOT` for an independent repeat. Existing output directories
may auto-resume. The changed video timeline policy is included in the dataset
fingerprint for frozen-timeline publications; an old cursor is not silently
reused under changed video semantics.

## Acceptance evidence (2026-10-30)

- `outputs/ego6_validation/reference_cases.json`: all 19 published cases pass;
  35 camera views / 105 checked frames match the preprocessing reference pixels
  exactly. Start and padded-tail windows also pass the framework canvas path.
- `outputs/ego6_validation/numeric_cases.json`: all 19 cases pass numeric,
  normalization, mask and Action-63 / State-62 routing checks. An empty current
  State mask omits the State token; it does not discard a valid Action sample.
- `outputs/ego6_validation/regression_final.log`: 172 passed, 2 skipped across
  Action128, pretraining State, media decode, JPEG TAR, MCAP/Zarr and timeline
  tests. Subsequent strict-input checks pass all 9 native/timeline tests.
- `outputs/ego6_validation/train_launch.log`: real eight-GPU training on the
  original full six catalogs, not synthetic or reduced metadata catalogs.
  All 12 steps completed (two six-source cycles), with finite loss and gradient
  norms; rank-0 loss ranged from 1.3428 to 1.4531. The launcher exited with code
  0. Step 12 saved a complete model/optimizer/scheduler/trainer/dataloader
  checkpoint, including `.save_complete`, under the default output directory:
  `cosmos3_action/pretrain_state/state_h1_ego6_8gpu/checkpoints/iter_000000012`.

These checks validate representative cases and training integration, not every
frame in the full publication. No published data was modified. Schema and
timeline violations fail explicitly; existing no-eligible-Action drops and
ordinary media decode retry policies remain unchanged.

Startup currently scans 75,874 episode metadata partitions and took about
23 minutes for these six full datasets. After startup, observed non-save step
times range from about 13 to 59 seconds. This smoke test is not a throughput
benchmark, and native decoding performance is not claimed to match MP4.

## Workbook mixture: 58 catalogs

`examples/toml/sft_config/action128_pretrain_state_h1_58src_8gpu.toml`
copies paths and integer ratios from the updated workbook's
`全部catalog明细!K7:K64` and `I7:I64` exactly. Ratios sum to 100,000:
the six new Ego sources account for 40%, and the existing 52 catalogs for 60%.
The h1 State override, first-frame ColorJitter, four shared workers, prefetch 4
and packing budgets remain unchanged. The old 52-catalog recipe is retained.

```bash
bash examples/launch_pretrain_state_h1_58src_8gpu.sh
```

This is a 12-step, eight-GPU GA1 acceptance recipe, not a long production run.
All 58 catalogs passed schema/path checks and initialized successfully. The
mixed run completed 12 steps, saved a complete checkpoint, and exited with
code 0. Loss and gradients were finite, with no OOM or decode exception.
The short weighted run does not sample every low-ratio catalog.

Log: `outputs/ego6_validation/train_58src_launch.log`.
Checkpoint:
`/share-mixed/projset/liyue/model_ckpts/pretrain_state_h1_58src_8gpu_20261030/cosmos3_action/pretrain_state/state_h1_58src_8gpu/checkpoints/iter_000000012`.

## Production continuation: 24 nodes / 192 GPUs

`examples/launch_pretrain_state_h1_58src_192gpu.sh` uses
`examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml` and
reuses the 52-source production launch chain, including node-local staging,
platform rendezvous and per-rank load-balance traces. The original 52-source
recipe and launcher remain available.

The production recipe copies all 58 catalog paths and ratios from the 8-GPU
workbook recipe, in the same order: the original 52 streams occupy 60% and
EgoDex / EgoLive / EgoPro / EgoStandard / EgoVerse / Lightwheel occupy 40%.
RoboCasa365 and RoboTwin2.0 also use the updated workbook weights. It retains
the production shard16, GA3 and 100,000-step budget, with 32 workers / prefetch 32,
450-second worker timeout, media-error skipping, language/VAE compilation,
packing budgets, online W&B and checkpoint retention. There are 58 logical
streams mapped onto 32 physical workers with fixed affinity; 26 workers handle
two streams each and six handle one. Prefetch 32 is the global per-rank limit.

The checkpoint mode is `extend` with full training-state loading. It validates
old domains and dataset fingerprints, retains model/head weights, optimizer,
scheduler and training step, and resets data cursors once when the six streams
are appended. Later restarts of the unchanged 58-source run restore its new
cursors normally. It does not start again from the model-only iter17200 backup.

Keep the existing environment setup and 24-node launch command, replacing the
output-directory assignment and final launch line with:

```bash
export OUTPUT_ROOT="$PWD/outputs/pretrain58src_state_h1_nano_192gpu_ga3_run01"
bash examples/launch_pretrain_state_h1_58src_192gpu.sh
```

By default the launcher reads `checkpoints/latest_checkpoint.txt` from the
original 52-source run under
`outputs/pretrain52src_state_h1_nano_192gpu_ga3_from17200_run01/cosmos3_action/action128/action128_wam_nano_52src_state_h1`.
It requires `.save_complete`, model/optimizer/scheduler/trainer metadata and
dataloader state before launching. Set `ACTION128_SOURCE_RUN` to choose another
source run, or set `BASE_CHECKPOINT_PATH` after the existing `unset` commands to
pin one complete checkpoint identically on every node. Pin the path if the
source job may save another checkpoint while the new nodes start.

Use a distinct 58-source output directory to preserve the old run and its
configuration. The default new W&B job name is
`pretrain58src_state_h1_nano_192gpu_ga3_run01`. Native dependencies must be installed in
the shared training environment; the launcher checks imports before launch.
The `cosmos-framework-pwm-cu130-train` environment was supplemented with `mcap`
and `mcap-protobuf-support` for this migration; `zstandard` and `google-crc32c`
were already available. Other environments still need the packages listed
above. The lockfile includes the native-media dependencies while retaining
existing protobuf, lz4 and google-crc32c versions.

Offline validation checked the TOML schema, all 58 catalogs, exact workbook
ratios, retained production settings and append-only compatibility with the
saved 52-source run's model contract. All 24 targeted checkpoint-extension and
native-media/timeline tests passed. The 19 handoff cases' canvas sizes are all
covered by the existing production VAE compilation sizes; no size was added.
This does not constitute a 192-GPU training or full-checkpoint restore test;
switching the running job requires launching the new command.
