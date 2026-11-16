# Nano USR Parquet experiment

> Historical USR v1 configuration and validation. The GA2 production recipe and
> launcher now default to the [USR v2 restart](README_usr_v2.md), including the
> new catalog, 5:4:1 conditioning, 500-step saves, and step-12600 resume checkpoint.

Entry point: `examples/launch_sft_vision_nano_usr_128gpu.sh`.
Recipe: `examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml`.

Defaults: 16 nodes × 8 GPUs, token-budget multi-sample packing, gradient accumulation
**2**, LR `1e-4`, CFG text dropout `0.1`, checkpoint every
200 optimizer steps, evaluation disabled. T2V/TI2V/V2V weights are `7:3:1`
(normalized to `7/11`, `3/11`, `1/11`). Explicit per-row
`conditioning_latent_frames` still takes precedence. EMA is disabled. Fixed UND/GEN
budgets are 3072/98304 (101376 total), with HSDP shard degree 16. LR warms up for 50 steps, then stays at 1e-4.
Static language compilation and optimizer parameter selection follow the reference.

The launcher now selects
`examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml`
by default. `SFT_TOML_FILE` remains available for an explicit recipe override.
The older `vision_sft_nano_usr_128gpu_fixed_static.toml` GA4 profile uses the
same Qwen3-VL-8B-Instruct tokenizer and hybrid model-only warm-start.

The default data root is
`/share/project/liyue/data/parquet-data/train-usr-v1-20260908/full-r3-90-r6b-5-r6-5-seed1`.
The 12 dataset weights follow the supplied full-data packed-sequence counts
(total 473262). Set `USR_DATASET_ROOT` to select another root with the same dataset
subdirectories. Training reads existing Parquet directly, retaining the captions.

## Hybrid model initialization

Run once before launching the distributed job:

```bash
bash examples/prepare_nano_usr_hybrid_checkpoint.sh
```

The existing `cosmos_framework.scripts.merge_understanding_dcp` tool validates
exact Understanding key/shape/dtype compatibility and saved architecture identity
when available,
then overlays the slim Understanding checkpoint onto the full Nano DCP.
The preparation script refuses to overwrite an existing output.

Current default sources:

- Understanding/VLM: `/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct-DCP`
- Generation/DiT + Action + sound: `/share/project/eai_pwm/models/Cosmos3-Nano-DCP`
- Output: `/share/project/eai_pwm/models/Cosmos3-Nano-Qwen3-VL-8B-Instruct-Hybrid-DCP`

Preparation source/output paths can be overridden with
`GENERATION_CHECKPOINT_PATH`, `UNDERSTANDING_CHECKPOINT_PATH`, and
`HYBRID_CHECKPOINT_PATH`.

The current TOML loads the assembled Qwen3-VL-8B-Instruct/Nano hybrid with
`checkpoint.load_training_state=false`, so it starts a new optimizer, scheduler,
trainer iteration and dataloader state at iteration 0. Override
`checkpoint.load_path` and set `checkpoint.load_training_state=true` only when
resuming a checkpoint produced by this run.

The hybrid contains 1165 tensors: 750 Understanding tensors from Qwen3-VL-8B-Instruct,
415 Generation/Action/sound tensors from original Nano. The latter include
`*_moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`, and Action projection weights;
copying only the action heads would not retain the original video DiT.
The optimizer retains `keys_to_select = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]`,
so the new VLM is frozen and the video generation branch adapts to it. Action
weights are retained in the model checkpoint, but the video recipe does not
activate Action training. Evaluation remains disabled; a recovery-time conclusion
requires subsequent evaluation of trained checkpoints.

## Platform launch

Run once on every node; the platform supplies node `RANK`, `MASTER_ADDR`, and
`MASTER_PORT`. `JOB_ID` comes from Kubernetes downward API annotations or an
explicit identical value on every node.

```bash
NNODES=16 JOB_ID=<shared-job-id> \
USR_DATASET_ROOT=/path/to/full-parquet-root \
bash examples/launch_sft_vision_nano_usr_128gpu.sh
```

`HF_MODEL_PATH`, `OUTPUT_ROOT`, and `SFT_TOML_FILE` are overridable. W&B runs
online by default; the platform launcher retains its configured credential.

The current HSDP shard degree is 16, so the production recipe requires at least
16 GPUs. A smaller smoke requires an explicit, separately validated memory profile;
do not reduce the shard degree and treat it as equivalent to the production run.

The compiler cache defaults to node-local
`/tmp/cosmos_compile_cache/nano_usr_fixed_static/rank<LOCAL_RANK>`, avoiding shared
filesystem races. `USR_COMPILE_CACHE_ROOT` can point to another local persistent
directory. `[model].resolution="480"` controls the diffusion schedule; offline latents
are never resized or re-encoded.

## Defaults and override precedence

Training defaults are defined in the TOML: GA2, max_iter=100000, resolution=480,
local Qwen3-VL-8B-Instruct tokenizer path, the model-only hybrid warm-start, and the USR data paths.
The script supplies no fallback training-parameter overrides. Editing the TOML
or selecting another `SFT_TOML_FILE` therefore takes effect directly.

- `MAX_ITER` and `TRAIN_RESOLUTION` add overrides only when explicitly set.
- `HF_MODEL_PATH`, `USR_DATASET_ROOT`, and
  `USR_COMPILE_CACHE_ROOT` are optional `${oc.env:NAME,default}` interpolations in
  the TOML; their default paths are not duplicated in the training launcher.
- Precedence for explicit training overrides: TOML < convenience environment
  flags < `EXTRA_TAIL_OVERRIDES` < quoted positional arguments.
- The optional NVTX environment block is appended last for NVTX callback fields.
- The script retains launch/runtime settings: NNODES, NPROC_PER_NODE, rendezvous,
  NCCL/thread settings, output/log paths and environment activation. These do not
  duplicate training knobs in the TOML.
- `job.project=cosmos3-t2v`, matching the reference T2V W&B project. The GA2 job
  name creates its checkpoint/output directory under that project.

## Offline input contract

Each target has `type="gen_video"`, a `vae_latent` object with absolute safetensors
`path`, `tensor_key`, `[48,T,H,W]` `latent_shape`, and sibling `fps`. Legacy
`vae_latent.target_fps` remains supported. Null pixel metadata is accepted.
Selected tensors must be bf16 and match their declared shape. Training neither
reads producer JSONL indexes nor constructs the VAE. Producer lineage sidecars
are optional; if supplied through `vae_manifest_path`, they are validated.
Standard catalog `manifest.json` files are not producer VAE lineage manifests.

## Historical validation (2026-09-08)

These results predate the current 3072/98304 GA2 recipe and do not validate its
memory profile. New shapes compile on their first use.

- All 768 smoke rows processed with the local Cosmos3-Nano tokenizer and their
  actual safetensors read back with bf16/shape checks. Maximum text length: 421;
  maximum patchified generation length: 28,400, within the fixed budgets.
- Eight H100 GPUs, GA4, unrestricted multi-sample packing, static compilation:
  two optimizer steps completed with exit 0. Logged video batch sizes ranged
  from 2 to 6; all rank losses were finite. Final `iter_000000002` checkpoint saved.
- Log: `/tmp/cosmos_usr_ga4_packing/logs/nano_usr_node0.log`.
  Artifacts: `/tmp/cosmos_usr_ga4_packing/cosmos3-usr/nano-128gpu/usr-ga4-packing-8gpu/`.
- The resolved-config test asserts GA4 and no sample-count cap. Single-node and
  16-node mocked launcher checks passed, including EXTRA overrides and NVTX.
  Bash syntax, Ruff and diff-whitespace checks passed. Actual 128-GPU execution
  has not been tested on this single-node host.

### Hybrid checkpoint smoke

The hybrid DCP was created and all 1165 tensors read back and compared exactly
with their designated source (750 trained Understanding, 415 original Nano
Generation/Action/sound). The output's `merge_verification.json` records source
paths, source metadata hashes and verification counts. Existing merge unit tests
passed (3 tests).

Eight H100 GPUs then completed two optimizer steps using the hybrid checkpoint,
GA4 and multi-sample packing. Logs confirm a model-only warm start at iteration 0;
all losses are finite, checkpoint `iter_000000002` saved, exit 0.
Log: `/tmp/cosmos_usr_hybrid_ga4/logs/nano_usr_node0.log`.
Artifacts: `/tmp/cosmos_usr_hybrid_ga4/cosmos3-usr/nano-128gpu/usr-qwen8b1684-hybrid-smoke/`.
This smoke verifies initialization and trainability, not recovery of DiT quality.

## Differences from the production reference

Compared with `launch_sft_vision_nano_t2v_128gpu.sh` (including its 8-GPU wrapper)
and `vision_sft_nano_t2v_8gpu_fixed_static.toml`:

- Required experiment changes: offline VAE instead of online, CFG dropout 0.1,
  prefix weights 7:3:1 instead of pure T2V, and 12 USR catalogs
  weighted by the supplied packed-sequence counts instead of Action100M/Koala at
  16:25. The supplied counts use text=3072/video=98304; they are sampling weights
  and do not change the current model token budgets.
- UND/GEN allocation is 3072/98304 (101376 total).
- HSDP shard degree is 16, with automatic replication.
- Same model, unrestricted multi-sample packing, bf16,
  language static compilation, full activation checkpointing, selected
  optimizer parameters, LR/betas/weight decay,
  100000 steps, worker/prefetch settings, and disabled previews.
- EMA is disabled; the reference enables it. The current schedule warms up for
  50 steps then holds LR constant, replacing the reference's repeated warmup/cosine cycles.
- The current TOML saves every 200 steps; the reference saves every 500 steps.
- Offline training removes VAE weights/warmup requirements and producer-sidecar
  requirements. The latent shape is used directly; no geometry processing.
- Compiler cache uses per-rank node-local storage because the shared cache failed
  in the first smoke. Cache enablement and static compilation remain enabled.
- Job names/project, output/log directories and dataset/checkpoint defaults are
  specific to this experiment. `NNODES` defaults to 16 (the reference's actual
  shell default is 1 despite its 128-GPU comment). Single-node mode supports
  launching without platform rendezvous variables; multi-node c10d settings match.
- The environment activation order, NCCL settings, compile threads, W&B settings,
  per-node bootstrap logs, `EXTRA_TAIL_OVERRIDES`, and optional detailed NVTX flags
  match the reference. The launcher clears inherited `LD_LIBRARY_PATH` for this container, overridable through
  `LD_LIBRARY_PATH_OVERRIDE`, and accepts quoted positional overrides / `MAX_ITER`.
- Both wrappers source `_sft_launcher_common.sh`; the USR wrapper directly sources
  it because the old 8-GPU wrapper requires the two old datasets and VAE sidecars.

## Known scheduling limitation

The exact-periodic joint loader expands the current integer dataset weights into
473262 entries and currently copies/shuffles that list on every selection. The
weights represent the requested packed-batch proportions, but caching the shuffled
order per period remains a follow-up performance improvement. Changing ratios
also changes the checkpointed schedule identity; use a fresh run for the new mix.

The current 3072/98304 profile and shard degree 16 have not been GPU-validated here.
