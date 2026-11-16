# Action policy: 1B three-tower scheme D

Use [`launch_sft_action_policy_1b.sh`](../examples/launch_sft_action_policy_1b.sh)
with [`action_policy_1b.toml`](../examples/toml/sft_config/action_policy_1b.toml)
for production and local checks. Environment setup is in [setup.md](setup.md);
general training and export instructions are in [training.md](training.md).

## Model and defaults

Scheme D keeps the original UND and VIDEO towers and adds an independent
1,010,059,264-parameter ACTION tower (`width_only_1b`): 36 layers, hidden1280,
MLP3840, 32 Q heads, 8 KV heads and head dimension128. VIDEO and ACTION exchange
attention bidirectionally without K/V adapters; UND is causal and frozen.
The DROID WAM input predicts video and actions, without robot state or action history.

| Setting | Default |
| --- | --- |
| Hardware / parallelism | 1×8 H100 80GB; DP shard = total GPUs, replicate1, CP1 |
| Execution | BF16, full activation checkpointing, static language compile |
| Attention | VIDEO Flash3 varlen; UND/ACTION sample-block Flex (`flash3_video`, Hopper required) |
| Batch | 92 samples/rank, GBS736, accumulation1 |
| Packing capacities | UND11776 + VIDEO59648 + ACTION2944 =74368; `gen_tokens=62592` includes ACTION |
| Decoder UND bucket | 5120; captions exceeding it use the original11776 capacity |
| Input workers | 18/rank, prefetch2; tuned with a160-core CPU quota |
| Video input | Offline VAE latents; no training-time video decoding or VAE encoding |
| Seeds | Trainer42, loader42; CLI overrides are applied last |
| Schedule | 50,000 updates, LR2e-5, warmup1,000 |
| Logging / saving | Online W&B; log every update; checkpoint every1,000 updates |
| Validation | Fixed held-out flow-loss fixture at start/end and every500 updates |

GBS must be divisible by `NNODES × NPROC_PER_NODE × ACTION_SAMPLES_PER_PACK`.
The launcher computes accumulation and audits actual sample counts. Padding is
excluded from losses and cannot be attended by real queries. Token overflow
fails rather than truncating a sample. Changing pack size or dataset requires
matching token capacities; `gen_tokens` already includes `action_tokens`.
The UND bucket changes decoder padding only; loader admission and samples stay
unchanged. A long-caption fallback can compile a second shape on first use.
VIDEO KV is stably grouped by sample for varlen attention; each tower retains its
own query and rotary encoding. Use `model.config.action_tower_attention_backend=flex`
for the all-Flex comparison or environments without Flash3.

## Data and checkpoint

Use a prepared DROID success WAM offline catalog and its matching latent
held-out fixture; see the conversion command below and
[the catalog schema](dataset_parquet.md). All referenced files must be accessible
on every node. The launcher requires an explicit dataset path, so a benchmark
subset cannot silently become the production dataset.

The launcher supplies shared-cluster defaults. Override these paths elsewhere:

```bash
export DATASET_PATH=/path/to/droid_success/offline_vae/offline.parquet
export BASE_CHECKPOINT_PATH=/path/to/converted/width_only_1b
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export HF_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export ACTION_EVAL_FIXTURE=/path/to/droid_success/offline_vae/flow_eval_offline.pt
```

`BASE_CHECKPOINT_PATH` must contain `model/.metadata`. A raw Nano checkpoint
has incompatible ACTION shapes. Prepare the scheme-D initialization once:

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. .venv/bin/python tools/prepare_small_action_checkpoint.py \
  --source /path/to/Cosmos3-Nano-DCP \
  --output /path/to/converted --presets width_only_1b
```

Conversion preserves UND/VIDEO tensors and records action coordinates and source
metadata. Initialize new runs from this converted base, not a pilot-finetuned model.

## Offline VAE data

Prepare DROID latents per independent observation window; encoding an entire episode
and slicing its latents changes the causal VAE context. Text CFG and multilingual
caption selection still run during training. Random image augmentation and random
task/history selection are rejected by this producer.

```bash
export LD_LIBRARY_PATH=''
export SOURCE_CATALOG=/path/to/droid_success/train.parquet
export SOURCE_EVAL_FIXTURE=/path/to/droid_success/flow_eval_batches.pt
export LATENT_ROOT=/path/to/droid_success/offline_vae
python -m cosmos_framework.tools.prepare_action_latents prepare \
  --catalog "$SOURCE_CATALOG" --vae "$WAN_VAE_PATH" --output "$LATENT_ROOT" --seed 42
torchrun --standalone --nproc-per-node=8 \
  -m cosmos_framework.tools.prepare_action_latents encode --output "$LATENT_ROOT"
python -m cosmos_framework.tools.prepare_action_latents finalize --output "$LATENT_ROOT"
python -m cosmos_framework.tools.prepare_action_latents evaluation \
  --output "$LATENT_ROOT" --fixture "$SOURCE_EVAL_FIXTURE"
```

`prepare --max-samples=N` makes an explicitly marked validation subset; omit it for
the full dataset. Encoding resumes completed shards with the same world size.
Finalization requires all selected windows and publishes `offline.parquet` plus an
`online.parquet` control with identical source indices and episode blocks.
Use `DATASET_PATH="$LATENT_ROOT/offline.parquet"`; the recipe defaults to offline.
Fixed validation also needs
an offline latent fixture via `ACTION_EVAL_FIXTURE`; pixel fixtures cannot be used
when the training VAE is disabled; use the generated `$LATENT_ROOT/flow_eval_offline.pt`.
Keep the manifest and latent shards immutable
when resuming training.
VAE weights are required by the producer; offline training does not load them.

## Launch and local checks

Activate the training environment and configure W&B credentials before launching:

```bash
source .venv/bin/activate
OUTPUT_ROOT=outputs/action_policy_1b_run01 \
bash examples/launch_sft_action_policy_1b.sh
```

A short check uses the same configuration and only overrides its duration/name:

```bash
OUTPUT_ROOT=outputs/action_policy_1b_smoke \
bash examples/launch_sft_action_policy_1b.sh \
  trainer.max_iter=2 job.name=smoke_droid_1b
```

For a smaller local check, adjust batch size, fixed capacities and worker count
together. This example uses two GPUs, pack4 and GBS8:

```bash
NPROC_PER_NODE=2 GLOBAL_BATCH_SIZE=8 ACTION_SAMPLES_PER_PACK=4 \
OUTPUT_ROOT=outputs/action_policy_1b_local \
bash examples/launch_sft_action_policy_1b.sh \
  trainer.max_iter=2 job.name=local_droid_1b \
  model.config.fixed_sequence_length.und_tokens=512 \
  model.config.action_tower_und_short_tokens=0 \
  model.config.fixed_sequence_length.gen_tokens=2816 \
  model.config.fixed_sequence_length.action_tokens=128 \
  model.config.max_num_tokens_after_packing=3328 \
  dataloader_train.max_sequence_length=3328 \
  dataloader_train.num_workers=2
```

CLI overrides are applied last. Prefer the batch environment variables over
manually overriding accumulation. Two updates test startup, not throughput or
convergence. Cold compilation can dominate them. Compile caches are per local
rank; set `ACTION_COMPILE_CACHE_ROOT` to isolate concurrent jobs or different shapes.

## Multiple nodes and resume

Run on every node with the same paths and job identity, changing `NODE_RANK`:

```bash
NNODES=2 NPROC_PER_NODE=8 GLOBAL_BATCH_SIZE=1472 \
MASTER_ADDR=<rank-0-host-ip> MASTER_PORT=29631 NODE_RANK=0 \
OUTPUT_ROOT=/shared/outputs/action_policy_1b_run01 \
bash examples/launch_sft_action_policy_1b.sh
```

Set `NODE_RANK=1` on the second host. Alternatively, supply the same
`RDZV_ENDPOINT` and unique `RDZV_ID` on all nodes for elastic rendezvous.
Choose GBS explicitly for the topology; it is not silently scaled by node count.
Actual inter-host networking must be validated on the target cluster.

Reusing the output root and job identity resumes the latest local DCP, including
optimizer, scheduler and loader cursors. Preserve topology, worker count, packing
and immutable data. The current multi-worker loader restores per-worker sample
cursors, but not the delivery rotation or caption/CFG RNG; restart is not bitwise
equivalent to uninterrupted training. Use a new output root/job name for
an independent run. Flow-loss validation does not measure robot task success.

## Performance and remaining limits

Measured on8×H100 80GB, pack92/GBS736/full AC, using the same18,400-window
DROID subset and matching per-rank input fingerprints. Each configuration ran once;
steps7–42 include loading, H2D, training and optimizer work, excluding initialization,
checkpoint saving and fixed validation.

| Configuration | Samples/s | Seconds/update |
| --- | ---: | ---: |
| Matched online VAE control | 46.429 | 15.852 |
| Offline VAE only | 109.603 | 6.715 |
| Offline VAE + short UND bucket | 112.405 | 6.548 |
| Offline VAE + short bucket + VIDEO Flash3 | **118.122** | **6.231** |

Offline VAE provides2.361× throughput; the short bucket adds2.56% and VIDEO Flash3
adds another5.09%, for2.544× combined. Maximum-rank loading wait averages6.68ms/update,
versus4.869s online; no offline measured update waits over100ms. Whole-cgroup CPU
usage averages9.69 cores versus132.01, and final whole-device NVML peaks at78.73%.

The [selected W&B run](https://wandb.ai/hechenrui123-beijing-academy-of-artificial-intelligence/cosmos3-wam-d-v023-opt/runs/es37iruc)
contains the complete-update measurements. These results cover a validation subset,
not a full-dataset long training run. BF16 encoding and changed GEMM shapes produce
small numerical differences; fixed flow-loss checks do not establish convergence
or robot task success. W&B MFU remains an uncalibrated estimator for this tower.

## Policy serving

The released Nano/Edge DROID policies use a separate serving path:
`python -m cosmos_framework.scripts.action_policy_server_robolab --port 8000`.
Install the `policy-server` dependency group and connect a
[RoboLab](https://github.com/NVlabs/RoboLab) client with
`python policies/cosmos3/run.py --task BananaInBowlTask`.
This released-model server is not a validated deployment path for the scheme-D
training checkpoint. See [inference.md](inference.md) for general inference setup.
