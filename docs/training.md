# Post-Training (Supervised Fine-Tuning)

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Optional FP8 training on H100](#optional-fp8-training-on-h100)
- [Step 1 - Prepare data and config](#step-1---prepare-data-and-config)
- [Step 2 — Prepare checkpoint](#step-2--prepare-checkpoint)
- [Step 3 — Run training](#step-3--run-training)
  - [Option A (recommended): the paired launch shell](#option-a-recommended-the-paired-launch-shell)
    - [Overriding the defaults](#overriding-the-defaults)
  - [Option B: raw `torchrun`](#option-b-raw-torchrun)
- [Outputs](#outputs)
- [Reuse torch.compile artifacts across jobs](#reuse-torchcompile-artifacts-across-jobs)
- [Export checkpoint to Hugging Face safetensors](#export-checkpoint-to-hugging-face-safetensors)
  - [ViT / vision tower (Cosmos3-Edge)](#vit--vision-tower-cosmos3-edge)
- [Convert to Diffusers](#convert-to-diffusers)
- [Config](#config)
  - [Common Hydra tail overrides](#common-hydra-tail-overrides)

______________________________________________________________________

<!--TOC-->

Fine-tune a pre-trained Cosmos3 model on your own dataset using supervised fine-tuning (SFT). Tested on 8× H100 (80 GB).

For the unified five-column training catalog, bounded Parquet access, exact
dataset ratios, task chat grammar, and canonical Action schema, see
[Unified training Parquet catalogs](dataset_parquet.md).

Prerequisites:

- [Setup](../README.md#setup)
- [Environment Variables](./environment_variables.md)
- [FAQ](./faq.md) — troubleshooting (OOM during SFT, defaults), common pitfalls.

The runnable artifacts (TOML recipes, paired launch shells, inference helpers) live in [`examples/`](../examples/README.md).

## Optional FP8 training on H100

Nano vision SFT can opt into TorchAO FP8 GEMMs for the generation MLP. This is
a training compute policy, separate from inference `--quantization-method`.
It is disabled by default; keep `model.precision="bfloat16"` and the existing
optimizer/parameter storage policy.

```toml
[model.fp8_training]
enabled = true
backend = "torchao"
recipe = "rowwise"
include_regex = ['^language_model\.model\.layers\.\d+\.mlp_moe_gen\.(gate_proj|up_proj|down_proj)$']
exclude_regex = []
```

Supported recipes are `rowwise` and `rowwise_with_gw_hp`; the latter keeps the
weight-gradient GEMM in high precision. Both use dynamic scaling. Only Nano dense
Qwen3VL generation MLP Linear modules are supported initially. The understanding
branch, attention, normalization, VAE and adapters keep their original compute
precision. Selection does not change the optimizer's trainable parameter filter.
Empty include lists, zero matches, unsupported modules or incompatible settings
raise errors rather than silently falling back to BF16.

The training path supports FSDP/HSDP, GA, full activation checkpointing and static
`torch.compile`, with ordinary-precision communications. FP8 all-gather, LoRA,
EMA, CP/CFG parallelism, three-tower/action/sound training, non-Nano MoE,
distillation and inference PTQ combined with training FP8 are not supported.

For an isolated 8-H100 smoke using the USR iter23000 checkpoint:

```bash
export LD_LIBRARY_PATH=''
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
IMAGINAIRE_OUTPUT_ROOT=outputs/fp8_validation \
torchrun --standalone --nproc-per-node=8 -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/vision_sft_nano_usr_fp8_smoke.toml
```

The smoke keeps 3K UND/96K GEN and GA2, but uses 8-way rather than production
16-way sharding. `max_iter=23010` means ten **additional** updates. Checkpoint and
USR catalog paths must be accessible. Expandable segments avoid the observed
96K compiled-backward fragmentation OOM on 8×80GB H100; FP8 scaling/backward
still create large temporary tensors and do not guarantee lower peak memory.
`vision_sft_nano_usr_fp8_128gpu.toml` retains the production 16-way shard / 8-way
replicate topology; single-node tests do not validate that topology.

Use different job names, output directories and compile caches for BF16 and each
recipe. Keep data, token budgets, optimizer and seeds identical; exclude startup
compilation and checkpoint saves from steady-state timing.

Checkpoint behavior:

- Old BF16 DCP supports weights-only warm-start and `load_training_state=true`.
- Same-job resume uses `latest_checkpoint.txt`, restoring the model, optimizer,
  scheduler, step, per-rank RNG and available dataloader state.
- When catalog dataloader state was actually restored, iterator reconstruction
  preserves the restored CPU training RNG. PyTorch otherwise draws fresh worker
  base seeds even with `num_workers=0`, shifting diffusion timesteps on restart.
  Fresh runs, zero-step checkpoints and checkpoints without dataloader state retain their original
  initialization behavior. Worker-local random augmentation needs its own
  resume validation; this fix does not make arbitrary worker transforms exact.
- Optional trainer metadata records the actual FP8 recipe, target shapes and
  software versions. Same-job changes are rejected. For an explicit precision
  migration (including FP8→BF16), use a **new job/output directory** with
  `checkpoint.load_path` and the appropriate `load_training_state` value.
- Old checkpoints without precision metadata remain loadable. The supplied
  iter23000 restart intentionally omits old-catalog data state; subsequent
  checkpoints restore the new catalog state.
- Inference/CPU export construct ordinary Linear modules even when the source
  YAML enables training FP8. HF export strips training-only policy/build fields.

The small real-GPU contract harness exercises backward, GA2, full AC, optional
compile, FSDP/HSDP and DCP independently of video quality:

```bash
torchrun --standalone --nproc-per-node=8 -m cosmos_framework.scripts.validate_fp8_training \
  --output-root=outputs/fp8_contract --run=split --recipe=rowwise \
  --shard-size=2 --compile --async-save --steps=5
# Re-run with --steps=10 to resume in a fresh process.
# Compare against --run=continuous starting directly with --steps=10.
```

Per-rank `loaded_rank*.pt` / `result_rank*.pt` record model/optimizer states,
LR/step and final RNG/losses. This synthetic MLP harness does not establish video
quality or convergence. Keep FP8 opt-in until BF16-controlled quality and
workload-specific performance checks pass.

Run the automated fresh-process recovery/migration checks on eight idle H100s:

```bash
pytest -q --manual --levels=1 --num-gpus=8 \
  cosmos_framework/model/generator/fp8_training_distributed_test.py
pytest -q --num-gpus=1 cosmos_framework/model/generator/fp8_training_test.py
```

For inference from this **offline-VAE, non-EMA** smoke checkpoint, use its
`checkpoints/iter_<N>/model` directory with the saved `config.yaml`, pass
`--no-use-ema-weights`, and supply these `--experiment-overrides`:

```text
model.config.load_vision_tokenizer=true
model.config.vision_vae_mode=online
model.config.tokenizer.vae_path=/absolute/path/to/Wan2.2_VAE.pth
model.config.fixed_sequence_length.enabled=false
```

Use the same overrides for its exported HF directory. No training FP8 conversion
is needed for inference. Initial 8-H100 tests exercised both recipes, ordinary
HF/DCP generation and exact distributed recovery; they do **not** establish
128-GPU speedup or long-run quality. See the implementation evidence and open
production gates in [the validation record](../plans/v0.2.5_t2v_fp8/record.md).

## Step 1 - Prepare data and config

Some datasets are license gated — visit the repository page and accept any terms, and authenticate with `uvx hf@latest auth login` (or set `HF_TOKEN`).

The download commands below stage source media under `examples/data/<dataset>/`
and the VAE at `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth`. Set
`DATASET_PATH` to the corresponding five-column Parquet catalog before launch.
See [Step 3 → Option A](#option-a-recommended-the-paired-launch-shell) for path overrides.

Select one of the following recipes:

<details open><summary><b>Vision SFT (Cosmos3-Nano)</b></summary>

T2I/T2V/TI2V/V2V SFT reads the unified five-column Parquet catalog. Set `$DATASET_PATH` to one catalog file or catalog directory; see [Parquet Training Dataset](./dataset_parquet.md).

Launch shell: `examples/launch_sft_vision_nano.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
    --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 \
    --local-dir examples/data/BridgeData2-Subset-Synthetic-Captions --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

</details>

<details><summary><b>Vision SFT (Cosmos3-Edge)</b></summary>

Full fine-tune of the Nemotron-2B-Dense-VL backbone (Cosmos3-Edge), on the same Bridge dataset as **Vision SFT (Cosmos3-Nano)**. Step 2 must convert the Cosmos3-Edge checkpoint, not Cosmos3-Nano.

Launch shell: `examples/launch_sft_vision_edge.sh`

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Edge

# Defaults match the launcher (see Step 3 → Option A to override).
uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
    --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 \
    --local-dir examples/data/BridgeData2-Subset-Synthetic-Captions --quiet
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir examples/checkpoints/wan22_vae --quiet
```

</details>

<details><summary><b>Reasoner Alignment SFT with LLaVA-OneVision (vfm-vlm)</b></summary>

Alignment SFT for the Reasoner variant on an Understanding Parquet catalog. Skips Step 2: by default the backbone `Qwen/Qwen3-VL-8B-Instruct` is fetched from the HF Hub by the model downloader at startup. To instead start from a merged Cosmos3 reasoner snapshot, build it with `convert_model_to_vlm_safetensors` (see [Step 2](#step-2--prepare-checkpoint)) and point `VLM_SAFETENSORS_PATH` at it.

Launch shell: `examples/launch_sft_llava_ov.sh`

```shell
# DATASET_PATH points to an Understanding Parquet catalog file or directory.
export DATASET_PATH=/path/to/understanding.parquet
#
# (optional) VLM_SAFETENSORS_PATH starts training from a local pre-converted
# Qwen3-VL safetensors snapshot (e.g. Cosmos3-Nano LM merged with the Qwen3-VL
# visual tower) instead of the public HF backbone:
# export VLM_SAFETENSORS_PATH=$PWD/examples/checkpoints/Cosmos3-Nano-VLM
```

</details>

<details><summary><b>Reasoner Alignment SFT with VideoPhy-2 (Cosmos3-Nano)</b></summary>

Reasoner alignment SFT for 1–5 physical-plausibility scoring using an Understanding Parquet catalog. `[job].task = "vlm"`. Bootstraps from `Cosmos3-Nano`'s language-model weights merged onto the public Qwen3-VL-8B-Instruct visual tower; the merged HF directory is consumed via `[model.backbone].safetensors_path` (plumbed by `VLM_SAFETENSORS_PATH`).

Launch shell: `examples/launch_sft_videophy2_nano.sh`

```shell
export DATASET_PATH=/path/to/videophy2.parquet
```

</details>

<details><summary><b>Reasoner Alignment SFT with VideoPhy-2 (Cosmos3-Edge)</b></summary>

Edge-tier counterpart of the VideoPhy-2 recipe above: same 1–5 physical-plausibility scoring on [videophysics/videophy2_train](https://huggingface.co/datasets/videophysics/videophy2_train), but on the compact **Nemotron-2B-Dense-VL** reasoner (SigLIP2 vision tower, `model_type = "cosmos3_edge"` — native HF metadata, no remote code; the model classes are registered in-framework) instead of Qwen3-VL. `[job].task = "vlm"`. The vision tower is frozen; the projector + LM train at a uniform LR. `[model.backbone].model_name` is the **public, ungated** omni release `nvidia/Cosmos3-Edge`, which supplies the arch/config/tokenizer **and** the reasoner weights: the training loader follows the snapshot's root safetensors index into its weight shards, so weights load directly from the download — no conversion step and no required weights env var. Like the nano/super recipes, `VLM_SAFETENSORS_PATH` is an optional override for loading from a local safetensors directory instead. The reasoner weights shipped inside `nvidia/Cosmos3-Edge` are the canonical Edge reasoner weights — there is no separate reasoner repo to download. Only 2B, so it runs on a 4-GPU (e.g. GB200x4) or 8-GPU allocation.

Launch shell: `examples/launch_sft_videophy2_edge.sh`

```shell
export DATASET_PATH=/path/to/videophy2.parquet
```

</details>

<details><summary><b>Action-Policy Post-Training (1B scheme D)</b></summary>

Use [the action-policy guide](./action_policy_1b.md) for the DROID WAM catalog,
converted three-tower checkpoint, production launch and local CLI overrides.
The single entry is `examples/launch_sft_action_policy_1b.sh`.

</details>

## Step 2 — Prepare checkpoint

Convert the base checkpoint to [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) format. `cosmos_framework.scripts.convert_model_to_dcp` ships in the unified `cosmos_framework/` package, so this step runs from the repo root (with the environment activated per [Setup](./setup.md)).

Set `BASE_CHECKPOINT_NAME` to the value from the recipe block you picked in Step 1 (`Cosmos3-Nano` or `Cosmos3-Edge`):

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano   # or Cosmos3-Edge — match the recipe in Step 1

# Default output dir matches the launcher (see Step 3 → Option A to override).
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/$BASE_CHECKPOINT_NAME \
  --checkpoint-path $BASE_CHECKPOINT_NAME
```

`$BASE_CHECKPOINT_NAME` is a registered name in the checkpoint catalog; the converter downloads the matching repo from the Hugging Face Hub and writes the DCP into `examples/checkpoints/$BASE_CHECKPOINT_NAME`.

**Reasoner Alignment SFT with LLaVA-OneVision (vfm-vlm):** Skip this step — the Reasoner alignment SFT loads `Qwen/Qwen3-VL-8B-Instruct` from the HF Hub at startup (no DCP conversion required). To start from a merged Cosmos3 reasoner snapshot instead, build one with `convert_model_to_vlm_safetensors` (see the VideoPhy-2 note below) and pass it via `VLM_SAFETENSORS_PATH`.

**Reasoner Alignment SFT with VideoPhy-2 (Cosmos3-Nano):** Use `cosmos_framework.scripts.convert_model_to_vlm_safetensors` instead.

```shell
# Step 2 (VLM checkpoint): merge Cosmos3-Nano LM onto the Qwen3-VL visual tower.
# Replaces the convert_model_to_dcp step used by the VFM recipes above.
python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
    --checkpoint-path Cosmos3-Nano \
    -o examples/checkpoints/Cosmos3-Nano-VLM
```

**Reasoner Alignment SFT with VideoPhy-2 (Cosmos3-Edge):** Skip this step — the recipe loads the reasoner weights directly from the (ungated) `nvidia/Cosmos3-Edge` snapshot at startup: the training loader follows the repo's root safetensors index into its weight shards, so no conversion or extraction is required. To start from a local safetensors directory instead, point `VLM_SAFETENSORS_PATH` at it (optional, same mechanism as the nano recipe).

## Step 3 — Run training

**Weights & Biases (optional):** every recipe TOML defaults to `job.wandb_mode = "disabled"`. To log a run to W&B, flip that field to `"online"` in the TOML and export `WANDB_API_KEY` in your environment before launching.

### Option A (recommended): the paired launch shell

Each recipe ships as a `examples/toml/sft_config/<recipe>.toml` (validated against the pydantic schema at [`cosmos_framework/configs/toml_config/sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py)) paired with `examples/launch_sft_<recipe>.sh`; the full catalog is indexed in [`examples/README.md`](../examples/README.md). Each `.sh` sources [`examples/_sft_launcher_common.sh`](../examples/_sft_launcher_common.sh) and forwards into `cosmos_framework.scripts.train --sft-toml=<recipe-toml>`. From the repo root, run the launch shell paired with the recipe you set up in Step 1. The wrapper resolves `DATASET_PATH`, `BASE_CHECKPOINT_PATH`, and `WAN_VAE_PATH` from the default locations under `examples/` (populated by Step 1 + Step 2), so no env-var setup is required (see [below](#overriding-the-defaults) to override):

```shell
# from the repo root, after Step 1 + Step 2:
bash examples/launch_sft_vision_nano.sh
```

Each launcher's default paths come from the `DATASET_PATH` + `BASE_CHECKPOINT_PATH` defaults declared at the top of its `.sh` (each uses `: "${VAR:=…}"` so any value you `export` in the shell before launching wins over the default):

| Launch shell                   | Post-Training Task | Default $DATASET_PATH (under examples/data/)               | Default $BASE_CHECKPOINT_PATH (under examples/checkpoints/)        |
| ------------------------------ | ------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| `launch_sft_vision_nano.sh`    | Generator SFT      | `BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge` | `Cosmos3-Nano`                                                     |
| `launch_sft_vision_edge.sh`    | Generator SFT      | `BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge` | `Cosmos3-Edge`                                                     |
| `launch_sft_llava_ov.sh`       | Understanding SFT  | (set `DATASET_PATH` to a Parquet catalog)                  | (none; backbone fetched at startup, or set `VLM_SAFETENSORS_PATH`) |
| `launch_sft_videophy2_nano.sh` | Understanding SFT  | (set `DATASET_PATH` to a Parquet catalog)                  | (none; set `VLM_SAFETENSORS_PATH` env)                             |
| `launch_sft_videophy2_edge.sh` | Understanding SFT  | (set `DATASET_PATH` to a Parquet catalog)                  | (none; weights load directly from `nvidia/Cosmos3-Edge`)           |

`WAN_VAE_PATH` defaults to `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` for every non-reasoner recipe.

Cosmos3-Edge is only 2B and also fits a 4-GPU node — e.g. `NPROC_PER_NODE=4 bash examples/launch_sft_vision_edge.sh` (tested on 4×GB200).

**Reasoner Alignment SFT with VideoPhy-2 (Cosmos3-Nano):**

```shell
# Step 3 (launch): supply the catalog and optional merged checkpoint.
export DATASET_PATH=/path/to/videophy2.parquet
export VLM_SAFETENSORS_PATH=$PWD/examples/checkpoints/Cosmos3-Nano-VLM
bash examples/launch_sft_videophy2_nano.sh
```

**Reasoner Alignment SFT with VideoPhy-2 (Cosmos3-Edge):**

```shell
# Same as the nano recipe, but no VLM_SAFETENSORS_PATH is needed: the reasoner
# weights load directly from the (ungated) nvidia/Cosmos3-Edge snapshot.
# Only 2B — on a 4-GPU node set NPROC_PER_NODE=4.
export DATASET_PATH=/path/to/videophy2.parquet
bash examples/launch_sft_videophy2_edge.sh
```

#### Overriding the defaults

If you'd rather put data or checkpoints on a different filesystem (e.g. a faster SSD or shared mount), download to your chosen path in Step 1 / convert the DCP to your chosen path in Step 2, then export the matching env var(s) before launching:

```shell
# Example: data on /scratch, base DCP on /nfs/ckpts.
export DATASET_PATH=/scratch/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/nfs/ckpts/Cosmos3-Nano
export WAN_VAE_PATH=/nfs/ckpts/wan22_vae/Wan2.2_VAE.pth
bash examples/launch_sft_vision_nano.sh
```

Each env var falls back to its default if unset, so you only need to export the ones you're moving. The downloads / `convert_model_to_dcp` commands in Step 1 + Step 2 just need their `--local-dir` / `-o` argument pointed at the same path you export here. `.gitignore` excludes `examples/data/` and `examples/checkpoints/` so the multi-GB downloads aren't tracked when you keep the defaults.

### Option B: raw `torchrun`

If you'd rather not use the paired launch shell, invoke `torchrun` directly with the recipe's TOML. Unlike Option A, **raw `torchrun` does not auto-resolve `DATASET_PATH` / `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH` from `examples/`** — they have to come from your shell:

- `BASE_CHECKPOINT_PATH` and `WAN_VAE_PATH` are read via `${oc.env:BASE_CHECKPOINT_PATH}` / `${oc.env:WAN_VAE_PATH}` at the TOML's `[checkpoint].load_path` / `[model.tokenizer].vae_path` keys.
- `DATASET_PATH` is referenced by the TOML's `[[dataloader_train.datasets]]` entry.

You have two options to fill them in (pick either, not both):

1. **Export them in the shell before `torchrun`** (whether they point at the default `examples/...` paths from Step 1+2 or your own overrides) — shown below.
2. **Edit the TOML by hand** — open `examples/toml/sft_config/<recipe>.toml` and replace the `${oc.env:DATASET_PATH}`, `${oc.env:BASE_CHECKPOINT_PATH}`, and `${oc.env:WAN_VAE_PATH}` placeholders with literal paths. Useful if you want a self-contained TOML you can hand to a colleague or commit alongside an experiment record.

Run from the repo root (the directory containing `pyproject.toml` and `examples/`); the snippet uses `$PWD` to absolutize the relative paths.

```shell
# This example uses the vision_sft_nano recipe end-to-end (same recipe as
# Option A). To switch recipes, swap TOML_FILE + DATASET_PATH per the table in
# Option A, and choose the matching Nano or Edge checkpoint.
TOML_FILE="examples/toml/sft_config/vision_sft_nano.toml"

# Match the launcher's defaults — or substitute your own paths.
export DATASET_PATH="$PWD/examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Nano"
export WAN_VAE_PATH="$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"

IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. \
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
    --sft-toml=$TOML_FILE
```

To resume from the latest in-progress checkpoint, point `BASE_CHECKPOINT_PATH` at the run's `checkpoints/iter_<N>/` directory under `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/` (see [Outputs](#outputs) below for the full layout).

## Reuse `torch.compile` artifacts across jobs

Compile-cache reuse is opt-in. PyTorch owns graph identity, guards, compatibility, cache I/O, and hit/miss decisions. Cosmos only resolves the configured path and sets `TORCHINDUCTOR_CACHE_DIR` before model construction; there is no artifact serialization, staging directory, or framework-managed publication step.

Add or override the following TOML block:

```toml
[trainer.compile_cache]
enabled = true
path = ".compile_cache"
```

The default and every other relative path resolve from the repository root, independent of the launch command's current working directory. The default therefore resolves to `<repo>/.compile_cache`. Absolute paths are used directly, and environment variables and `~` in the configured value are expanded. The directory must be writable and shared across machines if cross-machine reuse is required.

Example raw commands, with paths and base recipe environment prepared as described above:

```shell
# The first run populates the native cache; later runs use the same tree.
torchrun --nproc-per-node=4 -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/vision_sft_edge.toml -- \
  trainer.compile_cache.enabled=true \
  trainer.compile_cache.path=.compile_cache \
  model.config.parallelism.data_parallel_shard_degree=4 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  model.config.parallelism.context_parallel_shard_degree=2
```

Run the same command again from another GPU group or machine to reuse the cache. A native miss compiles normally and writes its results into the same tree. When cache configuration is disabled, Cosmos does not modify `TORCHINDUCTOR_CACHE_DIR`, so setting that environment variable directly remains supported. When configuration is enabled, the configured path replaces any inherited value. Compile-cache configuration is incompatible with `--deterministic`, which disables model compilation.

## Outputs

Outputs land under `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`:

1. `config.yaml`, `config.pkl`: Finalized resolved config (YAML for inspection, pickle for re-instantiation).
1. `launch_info.yaml`, `job_env.yaml`: Job metadata and captured launch environment.
1. `checkpoints/`:
    1. `latest_checkpoint.txt`: Pointer file containing the latest checkpoint directory name (e.g. `iter_000000200`).
    1. `iter_<iter>/`: DCP checkpoint saved every `[train.ckpt].save_freq` iterations (zero-padded 9-digit, e.g. `iter_000000200/`):
        1. `model/`: model weights (sharded `.distcp`).
        1. `optim/`: optimizer state.
        1. `scheduler/`: LR scheduler state.
        1. `trainer/`: training state — includes the `iteration` counter and per-rank `rng_state_<i>` (numpy + random + torch + torch_cuda).
        1. `dataloader/`: optional per-rank pickle shards (`rank_<i>.pkl`) — only present for dataloaders that implement `has_state()`.
1. `<callback_name>/`: Callback outputs, one directory per registered callback (e.g. `DeviceMonitor/`, `EveryNDrawSample/`, `norm_monitor/`).
1. `wandb/`, `wandb_id.txt`: Wandb run files — only present when `[job].wandb_mode` is `online` or `offline`.

The shorthand `$RUN_DIR` used in the rest of this page refers to `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>`. For example, with `IMAGINAIRE_OUTPUT_ROOT=outputs/train` and the `vision_sft_nano` recipe, `$RUN_DIR` is `outputs/train/cosmos3/sft/vision_sft_nano`.

## Export checkpoint to Hugging Face safetensors

Export the DCP checkpoint produced in Step 3 to a Hugging Face safetensors checkpoint:

```shell
RUN_DIR=$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>

CHECKPOINT_ITER=$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=$RUN_DIR/checkpoints/$CHECKPOINT_ITER

python -m cosmos_framework.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file $RUN_DIR/config.yaml \
  -o $RUN_DIR/model
```

The exported safetensors land at `$RUN_DIR/model` and can be used in [Inference](../README.md#inference) commands by passing `--checkpoint-path $RUN_DIR/model`.

### ViT / vision tower (Cosmos3-Edge)

The plain export above is all any recipe needs — Edge included. For Cosmos3-Edge, `export_model` bundles the SigLIP2 vision tower into `$RUN_DIR/model/vision_encoder/` and the processor/tokenizer files into the export root, so the exported directory is **self-contained**: inference runs from it offline, with no Hub download (see [Inference → Reasoner](./inference.md#reasoner)). `export_manifest.json` records where the tower and processor came from. Nano/Super exports bundle their processor files the same way.

Optional flags:

```shell
# --no-vit: generation-only export (T2V / I2V / V2V / T2I), ~1 GB smaller.
#     Writes include_visual=false so the checkpoint loads cleanly everywhere.
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file $RUN_DIR/config.yaml \
  --no-vit \
  -o $RUN_DIR/model

# --vit-checkpoint-path: use a custom tower from a local directory
#     (a snapshot root or a vision_encoder/ directory).
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file $RUN_DIR/config.yaml \
  --vit-checkpoint-path /path/to/snapshot \
  -o $RUN_DIR/model

# --verify: smoke-test the export on this node's GPU (tiny reasoner query +
#     tiny generation). A failure exits non-zero but keeps the export.
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file $RUN_DIR/config.yaml \
  --verify \
  -o $RUN_DIR/model
```

## Convert to Diffusers

Convert the Hugging Face export from the previous step into a Diffusers pipeline (`transformer/`, `vae/`, `scheduler/`, `text_tokenizer/`, `model_index.json`, …):

```shell
python -m cosmos_framework.scripts.convert_model_to_diffusers \
  --checkpoint-path $RUN_DIR/model \
  -o $RUN_DIR/diffusers
```

The input is the Hugging Face safetensors directory produced by `export_model` above — not a raw DCP checkpoint, so run the export first. The result at `$RUN_DIR/diffusers` can be loaded with `diffusers` or passed to [Inference](../README.md#inference) via `--checkpoint-path $RUN_DIR/diffusers`.

A generation-only checkpoint with no reasoner vision tower (`include_visual` unset) is converted without a `vision_encoder/` sidecar automatically; pass `--skip-vision-encoder` to force-skip it even when a tower is present.

## Config

The recipe TOML is parsed against the pydantic schema [`SFTExperimentConfig`](../cosmos_framework/configs/toml_config/sft_config.py) at load time. Every top-level key listed below maps to a sub-model in that file; unknown keys raise a `ValidationError` before training starts (`extra="forbid"` on every sub-model). Values may use OmegaConf env interpolation `${oc.env:NAME}` — the recipe TOMLs use this for `BASE_CHECKPOINT_PATH` (`[checkpoint].load_path`) and `WAN_VAE_PATH` (`[model.tokenizer].vae_path`). `DATASET_PATH` is consumed the same way but inside the experiment-SKU Python (`cosmos_framework/configs/base/experiment/sft/<recipe>.py`), not in the TOML.

For the full field-by-field reference (every section, every default, every VFM/VLM applicability note, the `"???"` MISSING sentinel, env interpolation, the VFM↔VLM path-remap table, and how to extend the schema), see [SFT Structured-TOML Config Reference](./sft_config.md).

The commonly tuned knobs:

1. `[job]`
    1. `task` — `"vfm"` (generator recipes) or `"vlm"` (Reasoner alignment). Picks the base config: `cosmos_framework/configs/base/config.py` vs `…/vlm/config.py`. Also drives `PATH_REMAPS` in `toml_config_helper.py`.
    1. `experiment` — Registered experiment SKU name (e.g. `vision_sft_nano`). Each SKU is a Python file under `cosmos_framework/configs/base/experiment/sft/` that wires up dataloader, model variant, and recipe-specific defaults.
    1. `project`, `group`, `name` — Components of the run output dir `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`. Also flow to W&B as the project / group / run name.
    1. `wandb_mode` — `"online"` (logs to W&B; `WANDB_API_KEY` must be set), `"offline"` (logs locally, sync later with `wandb sync`), or `"disabled"`.
1. `[model]`
    1. `max_num_tokens_after_packing` — VFM token-packing target. `-1` disables the cap. VFM only; VLM uses `data_setting.max_tokens` (tail override).
    1. `joint_attn_implementation` — VFM attention layout: `"two_way"` / `"three_way"` (NATTEN) / `"flex"`.
    1. `attn_implementation` — VLM attention impl: `"cosmos"` / `"flash_attention_2"` / `"sdpa"` / `"eager"`. VLM only.
    1. `lora_enabled`, `lora_rank`, `lora_alpha`, `lora_target_modules` — LoRA adapter knobs for the generation pathway. Used by SUPER-tier recipes; NANO-tier leaves `lora_enabled=false`. VFM only.
1. `[model.ema]`
    1. `enabled`, `rate`, `iteration_shift` — Exponential moving average of generation-pathway weights. Full fine-tunes typically enable it; LoRA recipes leave it off.
1. `[model.parallelism]`
    1. `data_parallel_shard_degree` — FSDP shard degree. `data_parallel_shard_degree × data_parallel_replicate_degree` must equal `WORLD_SIZE`. `-1` autoselects from torchrun world size.
    1. `data_parallel_replicate_degree` — HSDP replicate degree (outer replicate loop over the shard topology).
    1. `context_parallel_shard_degree` — Context-parallel shard degree. `>1` splits the sequence dim across an overlay mesh and does not multiply world size.
    1. `cfg_parallel_shard_degree` — Classifier-free-guidance shard degree. Almost always `1` for SFT.
    1. `fsdp_master_dtype` — Master parameter / FSDP reduce dtype: typically `"float32"`.
1. `[model.compile]`
    1. `enabled` — Enable `torch.compile`. Improves speed at the cost of memory.
    1. `compile_dynamic` — Whether to compile with symbolic-shape (dynamic) kernels. `True` (default) is appropriate for training; AR inference may prefer `False` for stable shapes.
    1. `mix_order_reduction_split_size` — Optional TorchInductor mixed-order reduction split override. Leave unset normally; `32` avoids excessive per-block shared-memory use in affected long-static-sequence RMSNorm backward kernels.
1. `[model]`
    1. `precision` — Compute dtype for forward/backward: `"bfloat16"` / `"float16"` / `"float32"`. Master weights stay fp32 separately.
1. `[model.activation_checkpointing]`
    1. `mode` — `"none"` / `"selective"` (per-op SAC, MoT-only) / `"full"` (per-block checkpointing).
    1. `save_ops_regex` — Regex patterns for ops to keep saved under `mode="selective"`.
    1. `preserve_rng_state`, `determinism_check` — Recompute determinism plumbing.
1. `[model.tokenizer]`
    1. `vae_path` — Wan2.2 VAE `.pth` path. Recipe TOMLs use `"${oc.env:WAN_VAE_PATH}"`. VFM only.
1. `[optimizer]`
    1. `lr` — Base learning rate.
    1. `betas`, `eps`, `fused`, `weight_decay` — Standard AdamW knobs. `eps` is VFM-only.
    1. `keys_to_select` — Substring allowlist for trainable params. Empty list = train everything; `["lora_"]` = adapter-only fine-tune.
1. `[optimizer.lr_multipliers]`
    1. Inline table of `<substring> = <multiplier>` pairs that scale the LR of params whose name contains the substring. The shipped vision recipes leave this empty (Hydra default `{}` stands).
1. `[scheduler]`
    1. `cycle_lengths`, `warm_up_steps` — Cycle length and warmup duration (lists, one entry per cycle), in optimizer steps.
    1. `f_max`, `f_min`, `f_start` — LR multipliers at peak / trough / step-0 (ratios of `optimizer.lr`).
    1. `verbosity_interval` — Scheduler-side LR log frequency. VFM only.
1. `[trainer]`
    1. `max_iter` — Total optimizer steps.
    1. `grad_accum_iter` — Micro-batches per optimizer step. Effective global batch = `grad_accum_iter × per-rank batch × world_size`.
    1. `logging_iter` — Console / W&B scalar log frequency.
    1. `distributed_parallelism` — `"fsdp"` is the only supported value.
1. `[trainer.compile_cache]`
    1. `enabled` — Opt into direct PyTorch native compiler-cache reuse.
    1. `path` — Cache directory. Relative paths resolve from the repository root; default `.compile_cache`.
1. `[trainer.callbacks.compile_tokenizer]`
    1. `enabled`, `compile_after_iterations`, `warmup_resolutions` — Lazy `torch.compile` of the VAE tokenizer. VFM only.
1. `[trainer.callbacks.grad_clip]`
    1. `clip_norm` — Max global L2 norm of the gradient (steps with larger norm are rescaled).
    1. `force_finite` — Replace NaN/Inf grads with zero (default `true` on VFM, `false` on VLM).
1. `[checkpoint]`
    1. `load_path` — Base DCP checkpoint directory to resume from (Step 2 output, or a prior run's `checkpoints/iter_<N>/`). Recipe TOMLs use `"${oc.env:BASE_CHECKPOINT_PATH}"`.
    1. `save_iter` — Save a new DCP checkpoint every N optimizer steps.
    1. `keep_last_n` — Keep the newest N completed checkpoints after a successful save. Omit to keep all checkpoints. Local/shared filesystem DCP only.
    1. `keep_every_n_steps` — Also retain completed checkpoints at positive multiples of this optimizer-step interval, outside the newest-N limit. Must be a positive multiple of `save_iter`; omit to disable. For example, `save_iter = 200`, `keep_last_n = 3`, `keep_every_n_steps = 2000` keeps steps 2000, 4000, ... plus the newest three. Applies to synchronous and asynchronous saves, including existing checkpoints after resuming with this policy; already deleted checkpoints cannot be recovered. A running trainer must restart with the updated code/config to apply it.
    1. `keys_to_skip_loading` — Substring blocklist applied at load time. Used to mask EMA / LoRA tensors when warm-starting from a checkpoint that doesn't have them yet.

    DCP tensor shards are written independently by each rank. Its save-plan and metadata object collectives use a lazily created CPU/Gloo process group, so checkpoint coordination does not require temporary CUDA memory from the training NCCL group.
1. `[dataloader_train]` — Top-level scalars only; the dataloader's class (LazyCall) and pipeline wiring (datasets, packers, …) stay in the experiment Python.
    1. `max_samples_per_batch` — Per-micro-batch sample cap (remapped to `max_batch_size` on the VLM packer). `null` / omitted = no per-count cap.
    1. `max_sequence_length` — Per-packed-sequence token cap (remapped to `max_tokens` on the VLM packer).
    1. `seed` — Dataloader RNG seed (VFM only).

### Generation CFG dropout

Set `[dataloader_train.visual_generation].cfg_drop_ratio = 0.1` to drop text
conditioning for 10% of generation samples (default: `0.0`, valid range: `[0, 1]`).
This applies to T2V, TI2V, and TV2V in both online and offline VAE modes. Dropped
samples use an empty user caption and omit assistant explanation text and its CE
loss; system instructions, user image/video inputs, and clean target-prefix frames
are preserved. Sampling is deterministic for the dataloader seed, epoch, dataset
fingerprint, and sample ID, independent of worker ordering and prefix-mixture draws.
The `vision_sft_nano_t2v_8gpu_fixed_static_cfg_drop.toml` recipe is a 10-step,
8-GPU smoke-test copy with this ratio enabled.

### Offline VAE latents for Nano T2V

The fixed-static Nano T2V recipe supports a strict run-level switch at
`dataloader_train.visual_generation.vae_mode`. `online` is the default and keeps the existing pixel decode plus
Wan VAE encode path. `offline` accepts only `gen_video.vae_latent` targets and derives
`model.config.load_vision_tokenizer=false`; VAE compilation and in-training generation visualization callbacks are
removed automatically.

Offline catalog entries may set `vae_manifest_path` for legacy producer lineage validation. Standard Parquet
catalogs manage lineage outside training and do not require producer sidecars. Before DataLoader workers
start, supplied manifests are checked for matching `vae_name`, `vae_tag`, and `vae_weights_sha256`; the manifest `dataset`
must also match the catalog dataset. Latent rows must provide an absolute `.safetensors` path, tensor key,
`[C,T,H,W]` shape, and `fps` beside `vae_latent` (legacy `vae_latent.target_fps` is also accepted).
Null pixel placeholders are ignored for latent targets. Tensor dtype and actual shape are checked when the selected batch is materialized.

Use the launcher with the fixed-static recipe and explicit sidecars:

```shell
SFT_TOML_FILE=examples/toml/sft_config/vision_sft_nano_t2v_8gpu_fixed_static.toml \
T2V_VAE_MODE=offline \
ACTION100M_DATASET_PATH=/path/to/action100m-offline-catalog \
ACTION100M_VAE_MANIFEST_PATH=/path/to/action100m/MANIFEST.json \
KOALA_DATASET_PATH=/path/to/koala-offline-catalog \
KOALA_VAE_MANIFEST_PATH=/path/to/koala/MANIFEST.json \
bash examples/launch_sft_vision_nano_t2v_8gpu.sh
```

This T2V recipe supports T2V/TI2V/V2V target-prefix visual-generation catalogs. It does not mix pixel and latent
rows or support gen-image, Action, Understanding, user-media, or multi-view samples in the same run. Switch back to
`T2V_VAE_MODE=online` and pixel catalogs for rollback.

DROID Action training has a separate window-local Offline VAE producer and
`droid_offline` adapter; see [the action-policy guide](action_policy_1b.md#offline-vae-data).

### Common Hydra tail overrides

These knobs aren't part of the pydantic schema today; pass them as trailing `key.path=value` positionals after `--` (the `cosmos_framework.scripts.train` flow forwards them through OmegaConf):

- `model.config.policy.backbone.model_name` — VLM backbone HF identifier (e.g. `Qwen/Qwen3-VL-8B-Instruct`). Used by `launch_sft_llava_ov.sh`.
- `data_setting.max_tokens` — VLM token-packing cap (the VLM analogue of `[model].max_num_tokens_after_packing`). Used by `launch_sft_llava_ov.sh`.

The launchers wire these via `TAIL_OVERRIDES=(…)`; the helper appends `-- "${TAIL_OVERRIDES[@]}"` after the `--sft-toml=` argument.

## Global prefetch for mixed catalog datasets

Catalog training defaults to `execution_backend = "shared_prefetch"`. Each
rank's loader schedules the mixed microbatch sequence first and prepares only
that sequence's next batches in a single shared CPU worker pool:

```toml
[dataloader_train]
execution_backend = "shared_prefetch"
num_workers = 12
prefetch_factor = 2
pin_memory = true
# Preserve the previous source-local worker topology before tuning physical workers:
logical_workers = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
worker_timeout = 300.0
```

`prefetch_factor=2` means **two pending batches across the entire mixture**,
including running, IPC, pinning and ready results. The budget belongs to each
rank's loader, not to the whole multi-GPU job. It is independent of worker count
and source count. The trainer's current batch and each logical shard's packing
carry/pool are additional memory. Missing prefetch defaults to 1 in shared mode.
`num_workers=0` runs the same logical streams synchronously without background
prefetch.

With `pin_memory=true`, physical workers and CUDA available, one rank-local
background thread receives worker results and pins model input ahead of use,
as in PyTorch's multiprocessing DataLoader. Pinning and ready queues share the
same K task credits; they do not add another prefetch window. Checkpoint RNG
metadata remains ordinary CPU data. Synchronous execution pins inline. The pin
thread stops before its worker processes during shutdown.

A global K of 1 allows at most one future batch task to run even if many physical
workers exist. A larger pool alone cannot compensate for a shallow window. Tune
K together with physical workers using real training throughput and host memory;
preparing slow batches earlier may require a larger K than a steady-state average
would suggest. These knobs do not change source-local logical shards.

A batch still belongs to one source. Dataset ratios, seed, source-local shard
order, processing, token budgets and pack boundaries retain their semantics.
Results are delivered in scheduled order; a slow source can delay that order.
Increase global prefetch to prepare it earlier, or adjust physical worker count
and static binding. All those performance changes preserve the logical topology.

`logical_workers` records each source's **previous DataLoader worker count**, in
`datasets` order. A zero denotes one main-process logical shard. If omitted, the
old even split of `num_workers` across sources is used. Before reducing physical
workers, explicitly pin this list to the original topology. Checkpoints validate
it, the active shards, DP coordinates, schedule and dataset fingerprints. An
optional `worker_assignment` list assigns each active logical shard, in
source/shard order, to a physical worker. The default is round-robin assignment.
Do not change logical shards, ratios or packing settings to tune throughput.

Shared checkpoints store **completed** batch positions and logical-stream CPU
RNG, not speculative prefetched cursors. Restart regenerates the uncommitted
suffix and reconstructs deterministic packing lookahead. The catalog batchers
and collators are deterministic; private third-party RNG/stateful stochastic
batchers or collators require their own resume integration. Existing pool
`exact`/`at_most_once` restrictions remain in force.

Old catalog checkpoints remain readable: source cursors are retained and the
next source-local worker is recovered from the saved microbatch schedule. They
do not contain worker augmentation RNG, so that missing state cannot be
recovered. A legacy synchronous source with random transforms may also depend
on the training process's interleaved RNG; switching it to isolated workers
requires a separate alignment check. Deterministic Action processing and
sample/pack order are covered by alignment tests.

With context parallelism, a local batch is committed once at the end of its
complete CP data window. Save checkpoints at window boundaries; a midwindow
save is rejected because the partial model/data window cannot be reconstructed
from a batch cursor alone.

`execution_backend = "legacy_per_source"` retains the old per-source DataLoaders
for comparison, where `prefetch_factor` again means per-worker prefetch. Use the
appropriate configuration values when switching. Shared checkpoints cannot be
silently loaded into the legacy backend; roll back from a legacy checkpoint or
start an explicit weights-only continuation in a new output directory.

Workers use Linux `fork`, matching the previous catalog loaders, and execute CPU
operations only. The timeout applies while waiting for a scheduled result;
raise `worker_timeout` if a valid packed batch legitimately needs longer. Worker
failures/timeouts report the offending batch and close the pool. Call `close()`
or use the loader as a context manager in standalone tools and early-stop tests.

For the 23-source Action128 WAM production entry
`examples/launch_sft_action128_wam_nano_160gpu.sh`, the paired TOML uses a small
shared pool (`num_workers=4`, global `prefetch_factor=16`) and records the
post-OOM synchronous topology as 23 zero entries in `logical_workers`. Each zero
still denotes one logical shard. This recipe sets `action_gen=true` and trains
video plus Action128 with the three-tower model; the `job.task="vfm"` schema
category is also used by Action training. The global loader is on this Action
path. Keep the episode metadata column projection when merging these changes:
it controls a separate, substantial source of per-rank host memory.

### Skip slow microbatches in the shared catalog loader

The production `examples/launch_pretrain_state_h1_58src_192gpu.sh` recipe enables
worker-side microbatch deadlines. In `[dataloader_train]`:

```toml
microbatch_timeout_sec = 120  # 0 disables this feature (the general default)
microbatch_timeout_max_skips = 0  # retry without a count limit; consumer wait is still bounded
microbatch_skip_log_dir = "${oc.env:OUTPUT_ROOT}/logs/skipped_samples"
worker_timeout = 450
pin_memory = false
num_workers = 16
prefetch_factor = 8  # total scheduled batches per rank, not per worker
```

This production recipe keeps queued video pageable instead of pinning the full
prefetch window. The model still transfers the same batch tensors to CUDA when
it consumes them; host-to-device transfer behavior and host memory usage should
be measured together when changing this option.

The production recipe limits the physical pool and global prefetch window to
reduce concurrent decoding and the host working set. A smaller window also
provides less time to hide decoding, so compare exposed loader waits, memory
pressure and step times when tuning it; a larger window is not always faster.

The 58-source launch shell defaults `NUMPY_MADVISE_HUGEPAGE=0` before starting
Python. This prevents NumPy from requesting transparent huge pages for its
large CPU arrays, avoiding synchronous compaction observed on fragmented NUMA
memory during video decoding. It changes allocation advice for the launched
process tree, not the system THP policy. Set `NUMPY_MADVISE_HUGEPAGE=1` explicitly
to compare with NumPy's default behavior; compare both memory pressure and
throughput rather than inferring performance from free memory alone.
NumPy reads this flag at import time; see the
[NumPy documentation](https://numpy.org/doc/2.2/reference/global_state.html#madvise-hugepage-on-linux).

The same launch shell defaults `MIMALLOC_ALLOW_LARGE_OS_PAGES=0`, covering
PyArrow's mimalloc pool independently of NumPy. In the training environment
(PyArrow 21.0.0), isolated Arrow allocations and real Galbot/Lightwheel episode
reads produced huge-page-advised mappings with the default allocator settings;
this flag removed those mappings while preserving the mimalloc backend and
the episode rows, columns and content hashes. These tools-node comparisons do
not establish a production throughput improvement. Set the variable to `1`
explicitly for a controlled comparison; it takes effect in newly launched
processes and does not change the host THP policy or an already running job.
See the [mimalloc environment options](https://github.com/microsoft/mimalloc#environment).

The deadline starts when a physical worker begins preparing a scheduled batch;
queueing, IPC transfer and pinning are covered separately by `worker_timeout`.
The worker deadline uses Python's SIGALRM handler. Native calls that do not
return to Python can delay handling, so recorded `elapsed_sec` may exceed the
configured deadline; it is not a hard wall-clock cancellation guarantee.
The consumer watchdog still bounds training-side waiting independently.
For changing DP size while retaining committed Action128 sample cursors, see
[the virtual-shard migration procedure](action128_shrink_resume.md). Directly
loading the original per-rank cursors with a different DP size remains rejected.
On timeout, discard all candidates retained by that logical stream for the
unfinished batch (including buffered lookahead and the interrupted sample),
then rebuild the stream after the last discarded cursor. Refill the **same
source and global microbatch slot**, so DP ranks still execute the same number
of forward/backward calls and gradient accumulation steps. No empty batch is
sent to the model. A positive `microbatch_timeout_max_skips` limits replacements
per scheduled slot (the general default is 2). The production recipe uses 0:
continue replacing during prefetch, while `worker_timeout` still bounds how long
the trainer waits for its next batch. This avoids failing the experiment merely
because a future prefetched slot needed several replacements. It does not reset
the consumer watchdog on each skip. Timeout skipping
requires `shared_prefetch`, physical workers, resumable map/episode/parquet
sources and sequential batching. Configure
`microbatch_timeout_sec * (microbatch_timeout_max_skips + 1) < worker_timeout`;
allow additional room for worker queueing and cleanup.

Each worker writes a new JSONL file lazily on its first timeout under the fixed
`$OUTPUT_ROOT/logs/skipped_samples/` directory. File names contain UTC startup
time, a random run identifier, global rank, physical worker and PID. Repeated
launches and checkpoint resumes retain old files and create distinct new ones.
Each line identifies one discarded candidate: dataset, original sample index,
epoch/stream cursor, logical/physical worker, global microbatch ID, timestamp,
configured/observed duration and retry attempt. Action datasets additionally
record root, episode and window identity when available. Lines sharing a
`skip_id` belong to one discarded batch. Writes are flushed after every skip;
logging failures propagate rather than silently losing the audit trail.

Checkpoint cursors advance past skipped candidates when the replacement batch
is committed. Resuming an older checkpoint can encounter those candidates
again; these logs are evidence, not a persistent blacklist. Timing-dependent
skipping changes the effective training distribution and exact reproducibility.

Deadlines use a worker-local POSIX timer, interrupting Python and supervised
media-decode waits. An uninterruptible native call can delay Python signal
handling; the existing `worker_timeout` remains a fail-fast error for that case,
worker death, or blocked result transfer. This mechanism does not skip model
forward/backward calls or NCCL collectives.

### Action candidate and non-finite diagnostics

Action128 windows with no eligible target emit a `CosmosDataLoader: dropped
sample` warning with JSON `context`: source, root, episode, window, stream
cursor, rank and worker, plus a category (`no_rows`, `all_action_invalid`,
`no_valid_action_dimensions` or `no_controlled_action_dimensions`) and mask
counts. `Action128 candidate audit` INFO records give cumulative counts per
stream instance every 128 completed candidates and on orderly stream closure.
For each instance, use the latest record or attached `candidate_counts`, not the
sum of all cumulative records. `candidates` is the sum of `prepared`,
`no_target`, `other_drop`, `media_error`, and `excluded`; interrupted/timeout
attempts are tracked separately by timeout diagnostics. These are loader-side
counts: prefetch and replay mean they are not counts of samples consumed by the
model. Logs can end before a stream's next report on an abrupt termination.

`loss_nonfinite/steps`, `affected_rank_steps`, and `all_rank_steps` count
optimizer-step windows using the existing all-rank loss flag. An affected rank
had at least one non-finite loss among its accumulated microbatches; the same
rank may contribute once in each step. Partial-rank events include rank-local
warnings. `grad_nonfinite/[modality/]norm_steps_rank0` and
`sanitize_steps_rank0` report the observing rank's pre-clip norm and cleanup
counts. The per-rank warning includes NaN/Inf counts and example parameter
names for the local gradient shards; an `inspection_failures_rank0` value makes
failed local inspection visible. A DTensor norm can include other ranks, and a
non-finite norm can arise from reduction overflow even if local gradients are
finite. Gradient observations occur after accumulation and prior callbacks;
they do not prove every individual microbatch gradient was finite.
