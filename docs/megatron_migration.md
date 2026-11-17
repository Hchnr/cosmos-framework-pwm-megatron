# Megatron migration: implementation and acceptance ledger

The independent repository is based on the working-tree snapshot recorded in
`MIGRATION_SOURCE.json`. The production entry is
`examples/launch_pretrain_state_h1_58src_192gpu.sh`. Its new default output tree
is `outputs/megatron_pretrain58src_state_h1_nano_192gpu_ga3_run01`; continuation
reads the original repository's 52-source run unless explicitly overridden.

**Status: implementation in progress. Production end-to-end correctness and
192-GPU throughput have not yet been accepted.** A synthetic component speedup
does not establish a production speedup. Standalone dense Qwen3-VL and a small Qwen3-VL-MoE configuration are also
qualified below; other model families and production scales remain unqualified.

## Backend integration

`model.parallelism.backend = "megatron"` in TOML selects the new backend; the
equivalent Hydra override is `model.config.parallelism.backend=megatron`.
The 58-source launcher supplies that override and selects `optim_grads` (ZeRO-2).
Set `MEGATRON_SHARDING_STRATEGY=optim_grads_params` for the lower-memory
ZeRO-3 option. Explicit Hydra overrides remain last and take precedence. `pytorch` remains available for
controlled comparisons and existing recipes.

Megatron-Core **0.20.0** manages parameter/gradient/optimizer-state sharding,
communication overlap and HSDP. Existing Cosmos data processing, packed MoT
math, parameter selection, LR groups, weight decay, FP32 Adam masters and
schedulers remain in use. This is a Megatron-FSDP backend, not a replacement of
the heterogeneous model by a generic GPT. TP and PP are not enabled on this path. The production three-tower recipe
retains CP=1; two-tower CP=2 qualification is described below.

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
| Action128 state, routed projections, spatial seams | Full 36-layer replay and live-loader training/resume pass; original VAE/processor and async save retained |
| FlashAttention 3 video + sparse attention for other streams | Original dispatcher retained; production-capacity two-layer math and full-depth reduced-token replay tested |
| Fixed UND 11776 / GEN 62592 / ACTION 2944; short UND 5120 | Original packing retained; fixed-padding numerical unit tests pass |
| BF16 computation, FusedAdam FP32 master weights | BF16 FusedAdam checkpoint continuation tested |
| Full activation checkpointing + language compilation | Actual decoder tested with compile + AC + HSDP |
| GA=3; shard16 × replica12 on 192 GPUs | GA=3 and shard2 × replica2 tested; 192-GPU run pending |
| Shared prefetch, metadata cache, local code staging | Existing implementation retained |
| NumPy/mimalloc large-page advice settings | Launch environment retained |
| Tokenizer compile warmup and resolution coverage | All 32 VAE AOT variants validated on four ranks; subsequent full-capacity compute cycle passed |
| Full model/optimizer/scheduler/trainer/RNG/loader resume | Full stock-trainer save/restart; live batches, stream revisit, worker RNG and content hashes match uninterrupted loading |
| Async checkpoint staging and background writer | Full 36-layer ~128 GB checkpoint save/load tested; uneven metadata coverage audited |

Two-tower CP=2 is qualified below; the original three-tower CP=1 constraint
remains. The baseline entry has EMA disabled and CP=1; EMA support is described below. Default initialization materializes the unsharded model before Megatron wrapping;
opt-in bounded initialization and its model-family scope are qualified below.

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
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/distributed_smoke.py
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/three_tower_smoke.py
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/three_tower_smoke.py --compile --replicas 2
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/checkpoint_smoke.py
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/checkpoint_smoke.py --optimizer FusedAdam --dtype bfloat16 --replicas 2
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/async_checkpoint_smoke.py
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
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/benchmark_fsdp.py \
  --backend pytorch --compile --steps 30 \
  --output artifacts/megatron/benchmark_pytorch_compiled.json
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/benchmark_fsdp.py \
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
1.9% for PyTorch and 1.1% for Megatron). The first uncontrolled full-gradient comparison showed relative L2 difference
0.5713 / cosine 0.8280. Input capture then showed that all 12 comparisons had
different VAE vision tokens (maximum absolute difference 0.0390625) under the
original nondeterministic cuDNN settings; the other network input tensors were
identical. That comparison is retained but is not a same-input parity test.

The controlled full-depth BF16 audit uses deterministic noising and cuDNN:
all 252 network input tensors across four ranks × three microbatches are
**bitwise identical**. Across 811 selected parameter tensors, gradient relative
L2 error is **0.0130685** and cosine is **0.9999629**, passing the explicit
2% / 0.999 gates. `full_depth_gradient_qualification.json` records the gate;
`full_depth_deterministic_gradients.json` retains all per-parameter observations.
These numerical gates do not qualify production memory or 192-GPU scaling.


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

A full production-capacity memory probe also completed two GA=3 forward/backward
and clipping cycles on four GPUs with **optimizer updates/state allocation disabled**.
Configured capacities were UND11776 / GEN62592 / ACTION2944, with the original
short-UND5120 policy selected for this sample. Peak allocated memory was
72.540 GB. Adding the analytic shard16 FP32 Adam master/moment state (5.993 GB)
gives 78.534 GB, versus 84.943 GB device memory. This is a compute-memory
measurement plus an analytic state budget, not a measured 192-GPU peak; it
excludes optimizer workspace, replica/network workspace and tokenizer graphs.
The VAE compile/coverage probe passed all 32 variants on every rank and completed
a fourth full-capacity compute cycle after compilation. Its measured peak was
the same 72.540 GB. A separate forced full-UND11776 probe completed two cycles
with peak 72.548 GB (78.541 GB including the analytic shard16 Adam budget).
Full-UND and compiled-VAE peaks were measured separately; 192-GPU optimizer
and network workspace qualification remains pending.

NormMonitor now computes FP32 reduction norms and extrema directly, avoiding
full FP32 activation/square and absolute-value temporaries at production lengths.
The FP16/BF16/FP32/FP64 strided-input, empty and non-finite CPU regression tests
pass; the full-capacity follow-up includes the new implementation.

The real 58-catalog loader has now also trained and resumed the full model:
its first three microbatches and the three after checkpoint restart exactly
match uninterrupted loading on all four ranks (24 batch identities), including
raw video/action/text hashes, sample cursors and worker RNG hashes. This live
smoke uses one physical worker and one sample per batch with reduced model
capacities. The extended loader restart passed six resumed batches per rank against a
nine-batch uninterrupted reference, including a revisit to `egostandard:55`.
`live_loader_resume_qualification.json` records matching cursor, raw-content and
worker-RNG hashes. The loader-only extension uses GEN16384 so its ninth sample
fits; the full-model live smoke uses GEN8192. Both use max_samples_per_batch=1. The test calls the same loader-binding helper as
the production training entry; omitting that helper does not save shared-loader
execution state and is not a valid cursor/RNG qualification.

Fresh-node preparation also handles the published EgoStandard `projset` media
prefix when original MCAPs are stored under `project`, preserving existing data
directories. The five path-layout tests pass. Public HF export omits Megatron
optimizer-sharding fields so inference chooses its own backend/CP preset; both
serialization regression tests pass.

## Context parallelism and legacy varlen padding

Two-tower MoT now retains the original CP dispatch with Megatron sharding.
On four GPUs (CP2 / shard2 / replica2), two GA=3 AdamW updates pass against
native FSDP2. Sequence-sharded and replicated attention-I/O layouts both pass
with full AC and compilation. FP32 forward/gradients/updated parameters are
bitwise equal in the final paired runs. The compiled BF16 sequence-sharded run
also has bitwise-equal outputs, gradients and updated parameters. This does
not change the source three-tower model's explicit CP=1 requirement or the
standalone VLM's source CP=1 constraint.

This exposed an original legacy-padding bug: causal varlen attention used its
padding-segment offsets, but non-fixed full attention left padded query rows
outside its cumulative ranges. The fused kernel could leave their backward
rows uninitialized. With PyTorch's deterministic uninitialized-memory poisoning,
the unchanged source has finite real-token forward outputs but **65,536
non-finite parameter-gradient elements**. The corrected implementation has zero,
with bitwise-equal real-token forward outputs. Non-fixed padding queries attend
to one isolated zero KV token, avoiding a quadratic padding-only attention block.
Fixed-bucket behavior and the single-sample inference dense path are retained.
`legacy_padding_gradient_qualification.json` records the original source path
and both observations; the original repository was not edited.

An actual two-layer OmniMoT/Action128 run also passes CP2 + BF16 + compile +
full AC + GA3 + ZeRO-2 + FP32 EMA, including CPU EMA-scope cache, exact weight
restoration and asynchronous full-state checkpoint reload after zeroing online
and EMA parameters. Original padding/static-compile regressions: 21 pass.
Larger CP degrees, other models and production scaling remain unqualified.

## Nemotron dense backbone and compiled parameter casts

A real two-layer Nemotron dense MoT backbone, using the production-style
`qk_norm_for_text=False` / generation QK norm split and Action128 heads, passes
compiled BF16 HSDP tests with CP=1 and CP=2. Both compare two GA=3 AdamW updates
against native FSDP2: outputs, full gradients and updated weights are bitwise
equal. A separate actual OmniMoT construction test passes EMA scope, update,
CPU cache and asynchronous staged checkpoint restoration with bitwise state.
These qualify the small backbone configuration, not a full pretrained Edge VLM
or Nemotron throughput.

This exposed a ZeRO-3 compilation failure: MCore overrides raw Parameter `.to()`
with an empty-storage guard, and its `_typed_storage()` result cannot enter a
Dynamo full graph. Nemotron RMSNorm explicitly casts its weights to FP32. The
integration now uses the normal Tensor cast during compilation, after FSDP
forward hooks gather the weights; eager calls keep the upstream safety check.
Normalization math is unchanged. The Qwen CP=2 regression also passes with
bitwise outputs/gradients/weights, and explicitly confirms that eager casts of
released raw storage still produce the original guard warning.

## EMA integration

VFM EMA now keeps FP32 shadows with exactly the online Megatron parameter's
local shard shapes and placements, including frozen parameters. It does not
allocate an EMA gradient buffer. Construction keeps the full EMA on CPU until
optimizer selection/sharding; the trainer's device move leaves this pending
copy on CPU. Existing preload and generation-copy semantics still run before
sharding. EMA updates retain the original beta schedule and foreach arithmetic.

The adapter completes pending parameter gathers before cache/copy/restore and
invalidates compute caches after changing weights. Otherwise a later evaluation
forward could reuse stale gathered parameters. `ema_scope` still evaluates the
online model using EMA values and restores the exact original training weights.

Four-GPU 2×2 HSDP tests pass for no_shard, ZeRO-1, ZeRO-2 and ZeRO-3: three BF16/GA=3 updates,
FP32 EMA bitwise reference equality, bitwise evaluation forward, exact training
weight restoration and DCP restart. An actual two-layer `OmniMoTModel` with
Qwen MoT and Action128 heads also completed forward/backward and EMA switching,
then CPU staging → spawn queue → background DCP write → reload. The compiled
ZeRO-2 run also verifies CPU EMA-scope caching and exact restoration after
explicitly zeroing both online and EMA parameters. Tests use a
small packed batch, not the full 36-layer EMA production memory footprint.
The original owner/gradient/NaN-step regressions also pass (67 tests).

## LoRA and CP output projections

The original in-place LoRA implementation is retained, including unchanged
base `.weight` keys and sibling `lora_A.weight` / `lora_B.weight` keys. A real
two-layer OmniMoT test with rank 4 / alpha 8, eight Q/K/V/O projection adapters,
compile, full AC, HSDP/ZeRO-3 and GA=3 passes two optimizer updates, EMA scopes,
CPU cache and asynchronous checkpoint save/zero/reload. The first update has
nonzero B gradients; the second has nonzero A and B gradients. Frozen base
weights stay bitwise unchanged and all restart weights restore bitwise.

The FusedAdam warm-start comparison with native FSDP2 independently verifies
64 online and 64 EMA language tensors and has bitwise equal initialization,
outputs, gradients and updated model/EMA. These are small real model fixtures,
not a pretrained Super LoRA performance qualification. Existing three-tower
LoRA restrictions remain in force.

CP's head-sharded output projection previously read only `.weight`, bypassing
LoRA. An independent full-projection reference with nonzero A/B weights records
maximum legacy output error **1.08854** using the unchanged original source
function. The fix adds each local adapter contribution before the existing CP
sum, retaining the low-rank computation rather than forming a dense B@A matrix.
With TF32 disabled for this FP32 reference, maximum output error is
**4.77e-7** and parameter-gradient error **1.86e-8**. The original collective's
backward convention is retained; all projection/input gradients are checked
against the independent full projection with the appropriate CP partial sums.

Actual CP=2 + LoRA + compile + EMA training/checkpoint tests also pass. Output-
projection LoRA B gradients are explicitly nonzero on both updates, A learns
on the second update, and frozen bases remain unchanged. CP=2 native FSDP2 vs
Megatron outputs, full gradients and updated parameters are bitwise equal.
The plain non-LoRA CP=2 regression remains bitwise equal. See `lora_cp_projection`,
`lora_mot_cp2_compiled_parity`, `omni_ema_integration_lora_*` and
`omni_hf_warm_start_lora_fusedadam` artifacts.

## Fresh HF initialization

Language-model HF loading now slices by the destination tensor's actual layout,
independently of the mesh used to distribute file reads. The old converter
unconditionally sliced whenever DP was configured, but Megatron wrapping happens
after loading; fresh unsharded targets therefore received the wrong shape.
Replicated targets receive full values, native DTensors use their placement,
and Megatron targets use uneven chunk metadata. Tied embedding head recovery
handles different local ownership of embeddings and lm_head. Four-GPU 2×2 HSDP
checks pass for all three layouts, an odd-sized vocabulary/norm, and both tied
and untied checkpoints, with bitwise equal weights. Original loader regressions:
49 passed / 2 skipped.

The actual Omni startup hook also requires parent-runtime metadata: calling
`language_model.state_dict()` bypasses the root post-hook. Before the fix,
`lm_head` loading tried a uniform 16-row slice where actual local storage had
0 or 32 rows. The parent runtime is now supplied explicitly for child loading,
and online compute buffers refresh after HF values are installed. EMA loading
uses the same chunk layout without installing EMA values into the online model.

UND→GEN copying now uses logical tensor values and the destination's actual
slice; plain/native parameters retain their original copy behavior. Existing
family-specific rules for generation-only norms/routers are preserved. Actual
Qwen and Nemotron startup-hook tests with compile, GA=3 and EMA have bitwise
initial states, outputs, gradients, optimizer updates and updated EMA versus
native FSDP2. FusedAdam/ZeRO-2 also passes. A partial-generation-freeze
FusedAdam/ZeRO-3 case exercises **four different UND/GEN ownership cases per
rank**; online and EMA each independently verify 48 language tensors against
the HF fixture, then update bitwise equally. A separate FusedAdam warm-start
case verifies all 48 online and EMA language tensors against HF/checkpoint
values, preserves the checkpoint's GEN weights, and has bitwise equal outputs,
gradients and updated model/EMA. Original loader/owner/gradient/
NaN regressions: **116 passed, 2 skipped**. See `omni_hf_initialization_smoke.py`
and `omni_fresh_hf_*` artifacts. These tests use tiny real model classes and
synthetic HF weights, not the full production dataset/checkpoint.

## Standalone VLM and HF consumption

`parallelize_vlm` now supports `backend="megatron"`, retaining the existing HF
model, freeze rules, AC/compile order and optimizer groups. VLM mixed precision
uses BF16 compute with native FP32 parameter masters and FP32 gradient buffers;
`fsdp_reduce_dtype` controls communication precision independently. The runtime
casts root/block floating inputs like FSDP2. This includes FP32 RoPE cos/sin kwargs:
omitting their BF16 cast produced different outputs despite identical weights.
The fixed full Qwen3-VL-2B audit has zero initial-weight differences and **all 174
captured forward outputs bitwise equal**. The rejected uncast audit is retained.

HF loading first materializes CPU masters, then casts compute parameters on CUDA.
Optimizer construction shards and restores exact master values. Frozen FP32
weights retain separate local master copies for checkpoint serialization, so
freezing does not lose source precision. Shared embedding identities survive
parameter swapping. This initialization still requires unsharded BF16 compute
weights to fit on one device and full CPU masters per rank; larger VLM families
need a different initialization path before they can be accepted.

`COSMOS_MEGATRON_CPU_INITIALIZATION=1` is an opt-in alternative that leaves
BF16 compute parameters on CPU until Megatron allocates CUDA buckets, including
through the trainer's pre-optimizer `.to("cuda")`. On Qwen3-VL-2B, every rank's
master SHA256 matches the default path, and MoE updates/full-state restart pass.
The measured peaks are 10,235,808,768 versus 10,235,198,464 bytes: **no meaningful
2B memory gain**, so it remains disabled by default. It still needs full CPU
masters per rank and does not qualify large-model initialization.

`COSMOS_MEGATRON_META_INITIALIZATION=1` is a separate opt-in for Qwen3-VL
(dense/MoE, without audio extensions). Parameters stay on meta until Megatron
materializes one direct owner at a time. HF files are assigned across shard
ranks and loaded after wrapping, directly into FP32 optimizer/frozen-master
views. This avoids full CPU masters per rank and a full CUDA model before
sharding. Required HF keys are validated before training. Tied owners retain
shared Parameter identity, and the embedding tensor already in flight also
populates the head's potentially different uneven slice. Buffers keep their
original precision. DCP resume defers weight restoration to the normal trainer.

The 2B initialization check independently compared all **626 state tensors per
rank** against HF file slices; every value is bitwise equal, and all four master
SHA256 hashes match the established initialization path. Peak allocation is
10,449,541,632 bytes versus 10,235,808,768 for the default: no GPU memory saving
at this size, so the option remains disabled by default. Initialization times
were about 11.5–12.0 seconds versus earlier 19.5–22.9-second observations;
these are unpaired startup observations, not a training-throughput claim.
Compiled dense and MoE 2-layer HSDP tests pass three GA=3 updates plus native
checkpoint continuation and self-restart. Frozen vision and restart weights
are bitwise equal; the MoE losses and updated weights are also bitwise equal
across backends. The complete pretrained 2B model also passed the same compiled
GA=3 update and checkpoint suite via meta initialization, with maximum loss
relative error 0.003223 and final weight relative L2 0.0001976, matching the
established initialization path. See `vlm_meta_*` artifacts and
`vlm_initialization_probe.py`.

The real Qwen3-VL-30B-A3B checkpoint also initialized successfully on four H100
80GB devices with ZeRO-3 and FP32 masters/gradients. Every rank independently
verified all **882 state tensors** bitwise against HF slices; peak allocation
was **78,384,509,952 bytes per GPU**. Initialization took 93.7–94.5 seconds in
this observation. The full model's forward produced finite DP-average loss
11.31694, but backward failed when a 2.32GiB FP32 communication bucket was
requested with only 1.64GiB device memory free. The failure is retained in
`vlm_meta_forward_backward_30b_capacity.json`. This qualifies 30B initialization,
not backward or full optimizer updates; Adam moments require additional memory.
The 235B family remains untested. A compiled dense test with frozen tied
embedding/head also passed direct DCP resume without HF preloading: all losses,
updated weights and restart weights are bitwise equal. The corresponding MoE
ZeRO-3/BF16-communication direct-resume test passes with maximum loss relative
error 0.000110 and final parameter relative L2 0.000229.

The meta path also passes single-GPU and replication-only (three replicas,
shard degree one) visual-input/compile/GA=3 tests with direct DCP resume. Maximum
loss relative errors are 0.000117 and 0.000052 respectively; checkpoint restart
weights and frozen vision remain bitwise equal. The single-GPU native baseline
uses the source's unsharded precision behavior, so this is tolerance-based
training parity rather than a bitwise forward claim.

On four H100s, the actual pretrained Qwen3-VL-2B completed three GA=3 AdamW
updates with real visual-token input, full AC, compile and 2×2 HSDP/ZeRO-2.
Maximum loss relative error over the run was 0.003223; final parameter relative
L2 was 0.0001976. Frozen vision weights are bitwise equal. Native FSDP2 full-state
checkpoint continuation and Megatron restart passed; restarted weights are
bitwise equal. This test uses a deliberately large lr=1e-3 and checks behavior,
not model quality or throughput (`vlm_qwen3_2b_compiled_hsdp_vision_checkpoint.json`).

A two-layer Qwen3-VL-MoE test also passed with visual input, compile/full AC,
2×2 HSDP, GA=3, EP=1 and global load-balancing loss coefficient 0.01: losses and
final weights were bitwise equal across backends, including both checkpoint
continuations. Expert parameters receive an explicit ordinary-DP expert mesh.
The original MoE patch suppressed differentiable auxiliary-statistic computation
during eager AC recomputation, causing 68 saved tensors versus 66; it now runs
the same math and suppresses only the repeated stash. Recomputed graphs are not
retained by the module. ZeRO-3 with BF16 gradient communication also passed the same continuation
checks (maximum loss relative error 0.000110, final weight relative L2 0.000229).
This does not qualify full 30B/235B MoE optimizer updates.

The full live-trained 16.182B checkpoint was exported through the original
`cosmos_framework.scripts.export_model` command into 32,364,243,344 bytes of HF
safetensors. Missing backend fields in public/older configs now default to
PyTorch during model construction, as required for inference. The exported model
loaded through the original HF reader and produced finite [4,128] Action128 WAM
outputs with unused slots zero. The original VAE decoded a finite [3,5,256,256]
video, the written MP4 was decoded back to five frames, and the physical action
inverse transform matched the contract. The training seam contract requires WAM camera
metadata; a generic text-to-image request is not a valid test of that checkpoint.
See `tests/megatron/export_consumption.py` and its JSON result. This is a tiny
2-step compatibility smoke, not a generation-quality evaluation.

Reproduce the scoped VLM checks with the independent environment activated:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/vlm_smoke.py \
  --vision --compile --replicas 2 --sharding-strategy optim_grads
python -m torch.distributed.run --standalone --nproc-per-node=4 tests/megatron/vlm_smoke.py \
  --moe --vision --compile --replicas 2 --sharding-strategy optim_grads
```

## Bounded Omni meta initialization and full 60B LoRA execution

`COSMOS_MEGATRON_OMNI_META_INITIALIZATION=1` enables an experimental bounded
initialization path for unquantized two-tower Omni models without audio
encoders. Native Qwen visual support is qualified below. The default remains unchanged. Small non-language heads
are constructed on CPU so their constant buffers retain real values, then
initialized on CUDA with the original policies. Language parameters stay on
meta until MCore materializes one direct owner at a time with the original
family initializer. RoPE/Fourier and routing buffers are rebuilt correctly.
Temporary reset methods are restored after wrapping. Quantization, three-tower
and generation-only noisy/cosine/shared-expert variants remain outside this
option's qualified scope.

The stock trainer's `.to("cuda")` moves other components while pending model
parameters remain on meta. EMA language parameters also stay on meta and become
FP32 local shadows after online sharding, avoiding a complete CPU/GPU EMA.
LoRA A matrices use full small Kaiming samples sliced by actual local offsets;
B starts at zero. This avoids applying ordinary DTensor random offsets to
MCore's uneven ownership. Tests explicitly confirm full-rank adapters and
identical logical initialization across all ranks/replicas.

Dense and small MoE models pass compiled GA=3 updates, active adapter gradients,
frozen-base invariance, EMA scope and asynchronous DCP restoration. CP=2 also
passes the combined dense LoRA/EMA/checkpoint suite. A fresh HF/FusedAdam/LoRA
comparison has bitwise equal initialized weights, outputs, gradients and updated
model/EMA versus native FSDP2, with 64 online and 64 EMA tensors independently
verified. UND→GEN seeding now keeps adapter A/B initialization: generation-only
adapters have no UND counterpart, so attempting to copy them was invalid.

The actual local Qwen3-VL-30B-A3B text model plus its complete generation tower
also ran on four H100 80GB devices:

- **60,493,967,424** total parameters; **13,369,344** trainable LoRA parameters
  (rank 16, alpha 32), ZeRO-3, EMA disabled.
- Pretrained HF UND loading and full UND→GEN seeding completed in about 54 seconds
  in this observation, excluding process import time.
- Every rank independently verified **1,060 UND/GEN base tensors** against the
  corresponding HF file slices, bitwise equal.
- A real **GA=3 forward/backward + FusedAdam update** completed with finite loss
  and global gradient norm 11.49608. Logical LoRA SHA256 matches across all ranks.
- Peak allocation was **36,667,569,152 bytes per GPU**. A complete unsharded BF16
  parameter copy alone would require about 121GB, exceeding one device.

This full-size probe uses two text and three action tokens, no live loader/VAE,
no block compilation and no full-size EMA. It establishes initialization and
real updates, not production throughput or quality. Small tests cover compilation
and EMA. Full-size checkpoint qualification is detailed below.
See `omni_meta_60b_lora.json`, `omni_ema_integration_*meta*` and
`omni_meta_fresh_hf_*` artifacts. Original CPU regressions still report
116 passed / 2 skipped.

## Qwen visual tower with bounded Omni initialization

The opt-in Omni meta path now supports Qwen native ViT in addition to text-only
conditioning. Parameters stay empty while constant position/rotary buffers are
created on CPU, then moved to CUDA. External Edge visual construction, audio,
quantized/FP8 and three-tower meta initialization remain guarded.

Real image patches pass through the native ViT, packed DeepStack conditioning
and Action128 prediction. Fresh HF loading independently verifies 109 online
and 109 EMA tensors in the frozen-ViT/LoRA test, or 93 each with trainable ViT.
Both cases use compiled language blocks, full AC, GA=3, FusedAdam and 2×2 HSDP:
initialization, outputs, gradients and updated model/EMA are bitwise equal to
native FSDP2. The trainable-ViT case explicitly checks a nonzero visual gradient.
Two compiled LoRA updates also pass frozen-base invariance, active A/B learning,
EMA scope, CPU shared/pinned asynchronous checkpoint staging, zeroing and
bitwise restoration. Cached versus uncached packing metadata outputs match.

The uncached input exposed an existing DeepStack failure: the packer generated
indices internally, but the consumer accessed the absent caller metadata.
DeepStack now uses the actual pack's causal indices. The original related CPU
regressions report **71 passed**, with **44 passed / 1 skipped** in supplemental
configuration/statistic/projection tests. See `omni_meta_visual_*` and
`omni_ema_integration_visual_lora_meta_compiled.json` artifacts.

A full local Qwen3-VL-30B-A3B understanding tower plus its complete generation
tower and native ViT (**61,032,598,832 parameters**) also passes pretrained
loading and two compiled GA=3 LoRA/FusedAdam updates on four H100s. Each rank
independently checks **1,411 base tensors** against HF file slices, bitwise equal.
Gradient norms are finite (14.66709 and 12.85167); logical adapter hashes match
all ranks. Peak allocation is **38,047,115,776 bytes per GPU**. EMA is disabled
for this full-size test. Its synthetic input contains three text/visual tokens,
one 2×2 patch grid and three action tokens. This establishes bounded
initialization and compiled training, without a live loader, VAE, quality or
production-throughput claim. See `omni_meta_60b_visual_compiled_lora.json`.

## Full-size checkpoint and reasoner entry points

The actual 60.494B two-tower MoE LoRA probe now completes two GA=3 updates,
CPU shared/pinned staging and a spawned Gloo writer for its entire model and
optimizer DCP. After explicitly zeroing model tensors, FP32 optimizer masters,
moments and scalar states, restoration reproduces every rank's complete local
state SHA256 exactly. An independent metadata audit finds no missing, overlapping
or out-of-range tensor chunks: model **120,987,936,016 bytes**, optimizer
**160,435,200 bytes**. A new process rebuilds the model from meta, skips HF
initialization, reloads the DCP, independently matches all four saved state hashes
and completes the next GA=3 update with finite gradient norm 15.17785. Peak
allocation is **36,707,680,256 bytes per GPU**. Artifacts are
`omni_meta_60b_lora_checkpoint.json`, `omni_meta_60b_lora_resume.json` and
`omni_meta_60b_checkpoint_coverage.json`. This direct probe covers model/optimizer;
the separate full-depth trainer test covers scheduler, RNG and live loader state.
The flat-layout auditor is opt-in; standard trainer checkpoint completion-marker
and four-component checks still pass on the original 16.182B qualification.

Cosmos's alternate `generate_reasoner_text` and decoder `reasoner_forward`
methods now enter through existing module FSDP pre/post hooks. Previously these
methods bypassed optimizer-to-compute view installation and could fail with a
Megatron parameter-map KeyError. The adapter restores ordinary and compiled
forward methods even on exceptions. Qwen ZeRO-2 and Nemotron ZeRO-3 comparisons
use compiled training blocks, greedy generation, a deliberately modified EMA
head, EMA restoration and a subsequent GA=3 training update. Generated tokens,
training outputs, gradients and updated parameters match native FSDP2 bitwise.
See `reasoner_forward_*` artifacts. This validates Cosmos AR entry points;
it does not qualify arbitrary third-party generation APIs.

## Standalone VLM compute performance

The pretrained Qwen3-VL-2B visual/text path was measured in four independent
process launches ordered **PyTorch, Megatron, Megatron, PyTorch**. Both use
BF16 compute, FP32 masters/gradients/communication, full AC, static block compile,
frozen vision/projector, fused AdamW, GA=3 and 2×2 HSDP on four H100s. Native
FSDP2 uses its existing reshard behavior; Megatron explicitly uses ZeRO-2.
Inputs are fixed synthetic 1024-token sequences with one 16×16 patch image;
every rank's input hash matches across all runs. Five warmup updates are
excluded and all remaining 60 observations per backend are retained.

| Metric | PyTorch FSDP2 | Megatron ZeRO-2 |
| --- | ---: | ---: |
| Mean update seconds | 0.626689 | 0.467669 |
| Median update seconds | 0.600267 | 0.432529 |
| P95 update seconds | 0.647559 | 0.894541 |
| Peak allocated bytes | 19,652,319,232 | 26,095,465,984 |

Mean-based throughput improves **34.00%**; the median ratio implies **38.78%**.
Memory is higher and P95 latency is worse. Maximum relative loss-trace difference
is 0.001658. This measures forward/backward and optimizer compute, including
warmup in memory peaks. It excludes live data loading, callback execution,
gradient clipping, checkpoint I/O and VAE; it is not production VLM throughput.
Python automatic GC remains enabled in this direct-loop driver. The source
trainer's configurable manual-GC callback is unchanged and is not exercised by
this benchmark. Both raw per-rank observations and aggregation are retained in
`vlm_2b_1024_*` artifacts. Reproduce with:

```bash
VLM_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct bash tools/benchmark_megatron_vlm.sh
```

The full-depth three-tower replay was also reaggregated including every
steady-state observation: mean updates are 4.655577 versus 4.390820 seconds,
**6.03% mean-based throughput gain**, consistent with its 5.98% median ratio.

## Remaining acceptance gates

The source entry, 58-source configuration, live training, asynchronous full-state
restart, exact loader continuation, production attention, GA/clipping, VAE AOT
coverage and controlled full-depth gradient comparison have passed the scoped
checks above. The four-GPU reduced-token full-depth paired replay measured
**5.98% higher throughput** (4.65231 s → 4.38990 s median) with higher ZeRO-2
memory. Production-capacity compute probes and an analytic shard16 state budget
are available, but do not replace a distributed production run.

1. Measure actual 192-GPU production-token throughput, peak memory, checkpoint
   stalls and stragglers with the original 16-shard × 12-replica layout.
2. Extend qualification to remaining model families, large-model initialization,
   optimizers, higher CP degrees and other parallelism paths.
3. Measure live, representative standalone VLM recipes including data loading,
   callbacks, clipping and checkpoint I/O. The 2B synthetic compute benchmark
   above establishes a scoped gain, not end-to-end production throughput.

Do not declare the entire repository migration complete before these gates pass.
