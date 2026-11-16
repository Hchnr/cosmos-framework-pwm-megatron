# USR v2.1 RF03 VLM restart (2026-09-30)

Keep the existing online command, TOML path and launcher. Change only
`export NNODES=16` to `export NNODES=24`; keep `NPROC_PER_NODE=8` for **192 GPUs**.
The TOML retains shard degree 16 and replicate degree -1, which resolves to 12
replicas at 192 GPUs (8 at 128 GPUs). No launcher changes are required.

- Source: the online run's `checkpoints/latest_checkpoint.txt` pointed to
  `iter_000025000` when preparing this restart. The full source path is below.
- Replacement VLM: `rf03_m2-141_format-align_small_hq_iter104_dcp`.
- Resume: `/share/project/eai_pwm/models/Cosmos3-Nano-USR-v2.1-iter25000-RF03-iter104-Resume-DCP`.
- New job name: `nano-usr-v2.1-rf03-iter104-resume25000-ga2-r480-fixed-static-und3k-gen96k-cfg01-cond541`.
  This prevents the previous run's auto-resume from bypassing the replacement VLM.
  Output root and project/group remain unchanged.
- Generation weights, optimizer moments, scheduler and trainer step come from
  step **25000**; RF03 optimizer/scheduler/step are not used. The next training
  step is **25001**, next periodic save **25500**, and max_iter remains **100000**.
- Per-rank dataloader cursors are omitted for the 128-to-192 topology change;
  samples may be revisited. This is not an exact continuation of sample order.
  Subsequent resumes at the same topology restore the new run's saved cursors.
- Dataset, ratios, GA2, LR, token budgets, frozen VLM, compile settings and all
  other training settings are unchanged. With GA2 unchanged, global batch/token
  capacity per optimizer step increases by 192/128 = **1.5x**.
- `load_vlm_visual=false` is retained: replace the VLM language weights used by
  this recipe; the source's extra visual encoder weights are not loaded.

The existing preparation tool creates the self-contained checkpoint, refuses to
overwrite it, compares every model tensor after saving, and SHA-256 verifies the
copied training state. Its report is `merge_verification.json` in the output.
Reproduction command (run once, only if that output does not already exist):

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. OMP_NUM_THREADS=8 \
python examples/prepare_nano_usr_v2_checkpoint.py \
  --generation /share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v21/cosmos3-t2v/nano-128gpu/nano-usr-v2.1-r7-1056-resume23000-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541/checkpoints/iter_000025000 \
  --understanding /share/project/eai_pwm/models/cosmos-private-trained-models/understanding_training/rf03_m2-141_format-align_small_hq_iter104_dcp \
  --output /share/project/eai_pwm/models/Cosmos3-Nano-USR-v2.1-iter25000-RF03-iter104-Resume-DCP
```

## Historical USR v2.1 production restart (2026-09-29)

Use `examples/launch_sft_vision_nano_usr_128gpu.sh` on each platform node.
The launcher now resolves its default TOML from its own repository root.

- Catalog root: `/share/project/liyue/data/parquet-data/train-usr-v2.1-20260928/full-r3-80-r6b-5-r6-15-epochs4-seed0`.
- Resume: `/share/project/eai_pwm/models/Cosmos3-Nano-USR-v2.1-iter23000-Resume-DCP`.
- Source: `/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v2/cosmos3-t2v/nano-128gpu/nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541/checkpoints/iter_000023000`.
- The restart directory contains verified hardlinks to all 516 files in the source
  `model`, `optim`, `scheduler`, and `trainer` components. Trainer iteration was
  read through DCP and verified as **23000**. `resume_manifest.json` records the source.
  These links survive deletion of the original checkpoint paths; treat both copies as immutable.
- Old `dataloader` state is intentionally omitted because the catalog fingerprints
  changed. `load_training_state=true` restores model and training state; data starts
  at the beginning of the new catalog. Subsequent same-run resumes restore the new
  dataloader state normally. No permanent `keys_not_to_resume` override is used.
- New job name: `nano-usr-v2.1-r7-1056-resume23000-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541`.
- Default output root: `outputs/usr_v21`. The separate job name prevents the old
  run's automatic resume pointer from taking precedence over step 23000.
- Dataset ratios, 128-GPU topology, GA2, token budgets, compile settings, LR and
  conditioning remain unchanged. The next periodic checkpoint is step **23500**.
- Existing explicit `SFT_TOML_FILE`, `USR_DATASET_ROOT`, `OUTPUT_ROOT` and CLI
  overrides still take precedence over defaults; remove stale deployment overrides.

## Historical USR v2 restart (2026-09-22)

The existing online command continues to use
`examples/launch_sft_vision_nano_usr_128gpu.sh` and
`examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml`.

- Catalog root: `/share/project/liyue/data/parquet-data/train-usr-v2-20260922/full-r3-80-r6b-5-r6-15-epochs4-seed0/`.
- All 29 supplied dataset weights are recorded verbatim in the TOML (total 4246463).
- T2V/TI2V/V2V conditioning weights: **5:4:1** (`0/1/2` prefix-frame choices).
- Save every **500 optimizer steps**. From step 12600, the next periodic save is step 13000.
- Same 16 × 8 GPUs, GA2, HSDP shard degree 16, UND/GEN budgets 3072/98304,
  LR 1e-4, CFG dropout 0.1, frozen VLM, and generation-only optimizer selection.
- New job name: `nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541`.
- Launcher output root defaults to `outputs/usr_v2`. The new job name prevents
  an old run's `latest_checkpoint.txt` from overriding the assembled checkpoint.

## Assemble once before the platform restart

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. OMP_NUM_THREADS=8 \
python examples/prepare_nano_usr_v2_checkpoint.py
```

The script reads these sources without modifying them:

- Generation and training state: `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000012600`.
- VLM: `/share/project/eai_pwm/models/cosmos-private-trained-models/understanding_training/r7_premix_unfrozen_samplemean_ga2_4node_20260921_r01_1056/model`.
- Output: `/share/project/eai_pwm/models/Cosmos3-Nano-USR-v2-iter12600-R7-1056-Resume-DCP`.

All 399 VLM tensors used by T2V are replaced; all 405 other model tensors are
preserved. The 351 extra visual-encoder tensors in the VLM source are unused
because this recipe has `load_vlm_visual=false`. Unexpected non-visual keys or
incompatible VLM shapes/dtypes cause preparation to fail.

The script reads back all 804 model tensors and compares them exactly, then
copies and SHA-256 verifies the original optimizer, scheduler, and trainer files.
The output is self-contained and is published only after verification; existing
outputs cannot be overwritten. `merge_verification.json` records sources, hashes,
tensor counts and restored iteration.

`checkpoint.load_training_state=true` continues from step **12600**, including
generation optimizer moments, scheduler, and trainer/RNG state. No new LR warmup
is introduced. The assembled checkpoint deliberately omits old-catalog dataloader
state because dataset names and ratios changed. Subsequent checkpoints from this
new run save and restore the new catalog state normally; no permanent
`keys_not_to_resume=["dataloader"]` override is used.

## Launch

Use the existing platform command on each of the 16 nodes. It needs no changes to
`SFT_TOML_FILE` or the launcher path. The supplied
`EXTRA_TAIL_OVERRIDES='job.wandb_mode=online trainer.compile_cache.path=/tmp/nano-usr-v1'`
only overrides W&B mode and compiler cache location; it does not override the
new data, conditioning, checkpoint, or save frequency.

Optional cache isolation for this run:

```bash
export EXTRA_TAIL_OVERRIDES='job.wandb_mode=online trainer.compile_cache.path=/tmp/nano-usr-v2/rank${oc.env:LOCAL_RANK,0}'
```

`USR_DATASET_ROOT` and `OUTPUT_ROOT`, when explicitly exported, still override
their defaults. All model/data changes are in the same TOML used by production.

## Preparation validation (2026-09-22)

- Assembled checkpoint published successfully: 804 tensors read back exactly,
  387 training-state files verified by SHA-256, trainer iteration 12600.
- All 29 catalog directories contain Parquet files; the first file in each was
  checked for the five-column schema and a readable visual-generation row.
- Full training config dry-run passed with the supplied online overrides;
  resolved conditioning is 5:4:1, save interval 500, and full-state resume enabled.
- Six checkpoint preparation/merge tests passed; Ruff, Bash syntax, and diff
  whitespace checks passed. No 128-GPU training job was launched during preparation.
