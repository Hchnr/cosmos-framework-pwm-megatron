# Megatron migration: implementation and acceptance ledger

The independent repository is based on the working-tree snapshot recorded in
`MIGRATION_SOURCE.json`. The production entry is
`examples/launch_pretrain_state_h1_58src_192gpu.sh`. Its new default output tree
is `outputs/megatron_pretrain58src_state_h1_nano_192gpu_ga3_run01`; continuation
reads the original repository's 52-source run unless explicitly overridden.

**Status: implementation in progress. Production end-to-end correctness and
192-GPU throughput have not yet been accepted.** A synthetic component speedup
does not establish a production speedup. Other training recipes and model
families have not yet been migrated or qualified.

## Backend integration

`model.parallelism.backend = "megatron"` in TOML selects the new backend; the
equivalent Hydra override is `model.config.parallelism.backend=megatron`.
The 58-source launcher supplies that override. `pytorch` remains available for
controlled comparisons and existing recipes.

Megatron-Core **0.20.0** manages parameter/gradient/optimizer-state sharding,
communication overlap and HSDP. Existing Cosmos data processing, packed MoT
math, parameter selection, LR groups, weight decay, FP32 Adam masters and
schedulers remain in use. This is a Megatron-FSDP backend, not a replacement of
the heterogeneous model by a generic GPT. TP, PP and CP are not enabled on this
qualified path; the production recipe already uses CP=1.

The backend lives in `cosmos_framework/training_backends/megatron.py`. Keeping
the communication runtime outside the module tree preserves original parameter
and checkpoint names. Construction freezes selected parameters before Megatron
allocates buffers. Communication completes before gradient clipping and NaN
handling. Gradient clipping computes local FP32 squared sums and reduces once
across the shard group: standard DTensor foreach norms can deadlock with
Megatron's unequal per-rank counts of nonempty parameter shards. Optimizer groups are retained rather than collapsed by an adapter.

Megatron's shallow backward-hook output inspection requires adaptation for
Cosmos `(SequencePack, metadata, cache)` outputs. The adapter preserves public
outputs and the original hook ordering while exposing tensor leaves to the
communication hooks. Without it, activation recomputation reads released weights.

NormMonitor aligns the union of statistic names before reducing packed sums and
maxima. Empty local optimizer shards can have no gradient at all; reducing only
locally present gradients caused collective ordering to diverge across ranks.
Parameter statistics reduce over the shard group, avoiding duplicate HSDP replicas.
The four-GPU regression explicitly covers missing statistics on alternating ranks
and direct parameter materialization for geometry/export callbacks.

Checkpoint metadata uses a batch exchange per state dictionary, handling missing
optimizer keys on ranks with empty local shards. The async writer receives plain
chunk metadata separately from staged CPU tensors and reconstructs DCP planning
hooks after the process queue transfer.

## Optimization parity checklist

| Production behavior | Implementation / verification |
| --- | --- |
| 58 sources, ratios sum to 100000 | Original catalog recipe retained |
| Three towers; 36-layer Nano + width-only 1B action | Original model math retained; full pretrained 36-layer real-batch replay and checkpoint restart pass |
| Action128 state, routed projections, spatial seams | Full 36-layer real-batch replay + original VAE/processor update and async save pass; live-data coverage pending |
| FlashAttention 3 video + sparse attention for other streams | Original dispatcher retained; production-capacity two-layer math and full-depth reduced-token replay tested |
| Fixed UND 11776 / GEN 62592 / ACTION 2944; short UND 5120 | Original packing retained; fixed-padding numerical unit tests pass |
| BF16 computation, FusedAdam FP32 master weights | BF16 FusedAdam checkpoint continuation tested |
| Full activation checkpointing + language compilation | Actual decoder tested with compile + AC + HSDP |
| GA=3; shard16 × replica12 on 192 GPUs | GA=3 and shard2 × replica2 tested; 192-GPU run pending |
| Shared prefetch, metadata cache, local code staging | Existing implementation retained |
| NumPy/mimalloc large-page advice settings | Launch environment retained |
| Tokenizer compile warmup and resolution coverage | Existing callbacks retained; end-to-end pending |
| Full model/optimizer/scheduler/trainer/RNG/loader resume | Full stock-trainer save/restart tested; exact live production-loader continuation pending |
| Async checkpoint staging and background writer | Full 36-layer ~128 GB checkpoint save/load tested; uneven metadata coverage audited |

EMA and CP>1 are explicitly rejected by the initial Megatron VFM backend.
The baseline entry has EMA disabled and CP=1. Initialization currently
materializes the unsharded model before Megatron wrapping; peak initialization
memory must be measured for each model size.

## Tests and reproducible commands

The production dependency environment was found at
`/share-mixed/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train`
(Python 3.13, Torch 2.10.0+cu130, FlashAttention3 1.0.3). An independent `.venv`
can inherit its dependency directory without changing it:

```bash
bash tools/setup_megatron_env.sh
source .venv/bin/activate
```

The production launcher activates this independent environment when present.
The common launcher invokes `python -m torch.distributed.run` and checks/runs
W&B through that same interpreter: dependency-sharing environments do not
necessarily contain console scripts, and inheriting a system `torchrun` would
silently select a different PyTorch. A launch-command probe covers this case.

Use the installed CUDA PyTorch/Transformer Engine/FlashAttention environment.
The original `pyproject.toml` pinned Megatron to an older Git revision; this
fork pins both the training dependency and optional `megatron` extra to 0.20.0.
Do not overwrite the installed CUDA PyTorch with an unrelated wheel.

```bash
export LD_LIBRARY_PATH=''
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
torchrun --standalone --nproc-per-node=4 tests/megatron/distributed_smoke.py
torchrun --standalone --nproc-per-node=4 tests/megatron/three_tower_smoke.py
torchrun --standalone --nproc-per-node=4 tests/megatron/three_tower_smoke.py --compile --replicas 2
torchrun --standalone --nproc-per-node=4 tests/megatron/checkpoint_smoke.py
torchrun --standalone --nproc-per-node=4 tests/megatron/checkpoint_smoke.py --optimizer FusedAdam --dtype bfloat16 --replicas 2
torchrun --standalone --nproc-per-node=4 tests/megatron/async_checkpoint_smoke.py
```

Verified so far on four H100 80GB GPUs, PyTorch
`2.12.0a0+0291f960b6.nv26.04.48445190`:

| Check | Result |
| --- | --- |
| FP32 outputs, averaged gradients, three AdamW steps with GA=3 | Pass; maximum output error 2.68e-6 |
| Actual heterogeneous packed decoder + full AC | Pass; maximum gradient error 9.31e-10 |
| Actual decoder + compile + full AC + 2×2 HSDP + GA=3 | Pass; maximum gradient error 1.70e-7 |
| FSDP2 → Megatron model + AdamW state, continue two steps | Pass |
| FSDP2 → Megatron BF16 + FusedAdam masters, 2×2 HSDP | Pass |
| Megatron → FSDP2 BF16 + FusedAdam masters, continue another step | Pass on production Torch 2.10 / ZeRO-2 / 2×2 HSDP |
| Missing-gradient NormMonitor statistics + direct uneven-parameter materialization | Pass on four GPUs |
| CPU staging → spawn queue → background DCP → reload | Pass; model weights bitwise identical |
| Actual Cosmos gradient clipping, GA=3, 2×2 HSDP, AdamW updates | Pass |
| 58-source production configuration dry-run | Pass; backend=megatron |
| Original data-loader resume tests (cursor + RNG) | 10 pass |
| Gradient clipping / finite handling / NaN-step regression tests | 13 pass |
| Projection fusion math + checkpoint key parity (experimental, disabled) | Pass |
| Production-width BF16 ZeRO-2, compiled, GA=3, clip + FusedAdam | Pass; gradient relative L2 9.66e-5; updated weights relative L2 1.11e-5 |
| Original small-action-tower tests | 11 pass |

Broader original tests expose pre-existing environment incompatibilities:
joint-path CPU FlexAttention backward is unsupported by this PyTorch build.
The first failing case was also reproduced in the unchanged source repository.
These failures are not counted as passing migration tests.

## Initial performance evidence

The JSON files in `artifacts/megatron/` contain per-step observations and memory
measurements. The controlled benchmark is an **8-layer synthetic SwiGLU stack**,
hidden size 1280, 2944 tokens/rank, BF16, full AC, GA=3 on four H100s. It excludes
attention, VAE, data loading, full three-tower routing and checkpoint overhead.

```bash
torchrun --standalone --nproc-per-node=4 tests/megatron/benchmark_fsdp.py \
  --backend pytorch --compile --steps 30 \
  --output artifacts/megatron/benchmark_pytorch_compiled.json
torchrun --standalone --nproc-per-node=4 tests/megatron/benchmark_fsdp.py \
  --backend megatron --compile --steps 30 \
  --output artifacts/megatron/benchmark_megatron_compiled.json
```

| Backend | Median optimizer-step time | Peak allocated CUDA memory |
| --- | ---: | ---: |
| PyTorch FSDP2 + compile | 59.58 ms | 590231552 bytes |
| Megatron-FSDP + compile | 53.16 ms | 637179904 bytes |

This first measurement is approximately **1.121× throughput**, with a higher
memory peak. Repeated paired runs and actual MoT/production-shape measurements
are still required. Neither the speedup nor the memory numbers should be
extrapolated to 192 GPUs.

## Production-shape component measurements (Torch 2.10.0+cu130)

These runs use **two actual MoT decoder layers**, the production 4096/12288
video/UND widths, 1280/3840 action width, 32Q/8KV/128 heads, 92 synthetic
samples, physical token capacities 5120/59648/2944, Flash3 VIDEO plus Flex
UND/ACTION, compilation, full AC, FusedAdam masters, and GA=3. They exclude
VAE/data/checkpointing and do not establish 36-layer/192-GPU performance.

| Configuration | Median step (5 measured) | Peak allocated memory |
| --- | ---: | ---: |
| Original FSDP2 | 1.04943 s | 17731441664 B |
| Megatron fully sharded | 1.04848 s | 17726485504 B |
| Megatron + fused input projections (experimental) | 1.06129 s | 16703910912 B |
| Megatron ZeRO-2 (`optim_grads`) | 1.02317 s | 17310732800 B |

The fully sharded replacement alone provides no meaningful speedup on this
compute-heavy four-GPU workload. Projection fusion regresses latency (~1.2%)
and stays disabled despite lower peak memory. ZeRO-2 is promising (~2.6%
throughput improvement); paired longer repeats, numerical checks and full-depth
memory qualification are required before selecting it for production. All
observations, including the rejected regression, are retained in artifacts.

## Longer paired repeats and real data

Two paired runs, 30 measured optimizer steps per backend per run, with the
order PyTorch → Megatron → Megatron → PyTorch, give aggregate medians
**1.05015 s (FSDP2)** and **1.02989 s (Megatron ZeRO-2)**: **1.97% throughput
improvement** on this two-layer production-shape component benchmark. Per-pair
improvements were 1.92% and 2.07%. The 60 raw observations per backend and run
order are recorded in `production_shape_repeat_summary.json` and the repeat
JSON files. VAE/data loading/checkpoint costs remain outside this measurement.

A real one-sample batch has been materialized through all 58 configured catalog
streams' initialization and the unchanged shared-prefetch/processor path. Video
shape is `[3,61,480,640]`; Action128 shape is `[61,128]`, with finite actions
and the original SequencePlan. This validates one returned sample, not decoding
every dataset. The replay tensor is in ignored `outputs/megatron_validation/`.

The stock Cosmos trainer also completed one update of the **full 36-layer,
16.182B-parameter model** from the original trained checkpoint, using the real
batch replay and reduced physical token capacities `[512,8192,128]` on four GPUs.
All four ranks completed GA=3 with finite losses; global gradient norm was
0.74295. The unmodified VAE encode path, original callbacks and full asynchronous
model/optimizer/scheduler/trainer/RNG checkpoint save completed successfully.
The DCP metadata audit verified all 1,214 model tensors and 4,866 optimizer
state tensors: 32.364 GB model storage and 95.896 GB optimizer storage have
complete, in-bounds, non-overlapping chunk coverage. The audit checks scheduler
and trainer tensor chunks as well (`tools/audit_megatron_checkpoint.py`).
CPU staging allocated about 120 GiB and the shared-filesystem save took minutes;
this is correctness evidence, not production throughput evidence. The tokenizer
compile callback was disabled for this smoke. See
`full_depth_real_batch_smoke.json` and its filtered log. Full-state restart also passed: model, optimizer, scheduler, trainer and RNG
loaded at iteration 1, all four ranks completed iteration 2, then the stock
async writer saved another complete checkpoint. See
`full_depth_real_batch_resume.json`. Replay does not validate advancing live
loader cursors.

The source checkpoint metadata reports 32.36 GB of stored model tensors.
`production_memory_estimate.json` records the analytic persistent storage
estimate for shard16 plus 36-layer checkpoint inputs. It excludes VAE and
attention/MLP workspaces and is not a measured full-model peak.

Four full-depth replay runs are recorded in `full_depth_repeat_summary.json`:
PyTorch medians were 4.6472 / 4.6612 seconds and Megatron medians were
4.3931 / 4.3881 seconds. These observations show a throughput improvement at
this reduced-token scope, with substantially higher memory for ZeRO-2. Loss
trajectories vary within the same backend (maximum relative difference about
1.9% for PyTorch and 1.1% for Megatron). Full-gradient snapshots are being
audited separately; these timings do not close the correctness or production
memory/scaling acceptance gates.

The full-depth paired replay harness is:

```bash
export BASE_CHECKPOINT_PATH=/path/to/original/checkpoints/iter_000001200
export REAL_BATCH_PATH=$PWD/outputs/megatron_validation/real_batch.pt
bash tools/benchmark_megatron_full_depth.sh
```

It runs PyTorch, Megatron, Megatron, PyTorch for 20 steps each, excludes the
first three steps, includes the stock trainer and VAE encoding, and excludes
checkpoint save I/O, tokenizer compilation and live shared-prefetch loading.
Both backends reset seed 42 after initialization; JSON files retain all timings,
mean final-microbatch losses and peak memory. This reduced-token, four-GPU replay
cannot establish production-token/192-GPU throughput or memory safety.

## Remaining acceptance gates

1. Full 58-source config construction, real-data smoke and full-state resume,
   including scheduler, RNG, loader cursors and source-extension semantics.
2. Validate actual production attention backends, tokenizer compilation, gradient
   clipping and asynchronous checkpoints at representative shapes.
3. Compare warmed steady-state step latency, sample/token throughput, memory,
   checkpoint stalls and stragglers on identical hardware and data. Include
   repeated paired runs; retain raw observations and rejected regressions.
4. Validate distributed checkpoint resharding and inference/export consumption.
5. Qualify 192-GPU scaling, then extend the backend to the remaining repository
   training paths. Do not declare the overall migration complete before these
   gates pass.
