# Attention-only activation checkpointing

Implementation and measurements: [record](../../plans/v0.2.8_ac_for_attn/record.md).

The online recipe uses `mode="full"`, where `save_ops_regex` is ignored. On the
NV FA3 provider tested here, the selective policy sees
`_flash_attn_forward.default`, **not** the CUDA kernel name `fmha`.

Use the existing training options (no attention backend or model change):

The opt-in `examples/launch_sft_vision_nano_usr_attn_ac.sh` supplies the three
attention settings below and forwards all online launcher controls. Its default
topology is inherited from the online launcher; set `NNODES=1` and shard=8 for
this eight-GPU experiment. Changing AC policy requires a new process/cache, not
mutating an already compiled model.

```bash
NNODES=1 NPROC_PER_NODE=8 bash examples/launch_sft_vision_nano_usr_128gpu.sh \
  model.config.parallelism.data_parallel_shard_degree=8 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  model.config.activation_checkpointing.mode=selective \
  'model.config.activation_checkpointing.save_ops_regex=["^_flash_attn_forward[.]default$"]'
```

Set independent output/job/cache paths for experiments. The command above keeps
the online sequence budget; capacity and final tested budgets are in the record.
The equivalent TOML section is:

```toml
[model.activation_checkpointing]
mode = "selective"
save_ops_regex = ['^_flash_attn_forward[.]default$']
preserve_rng_state = true
determinism_check = "default"
```

This retains the FA3 forward operator's output/LSE/auxiliary returns, while all
other operations retain the existing `MUST_RECOMPUTE` policy. It does not cache
the whole attention module or skip backward. The regex is deliberately specific
to the observed backend: other providers/backends require a separate probe.
Revert to `mode=full` to restore the original policy. Production defaults remain
unchanged.

## Reproduce

Use the checkout's existing `.venv` (Torch 2.10 / CUDA 13.0). The runner forwards
that interpreter's environment to the original launch shell, derives the TOML
before validation, isolates output/cache, validates restored iteration 25000,
and disables benchmark checkpoint writing and W&B. It fixes `split_default`
and the NV provider. Run names must be new.

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. .venv/bin/python tools/ac_attn/probe.py \
  --output /tmp/cosmos-ac-attn/probe.json
LD_LIBRARY_PATH='' .venv/bin/python tools/ac_attn/run.py \
  --ac full --name full_0 --gen-tokens 65536 --filter-oversized --warmup 10 --steps 30
LD_LIBRARY_PATH='' .venv/bin/python tools/ac_attn/run.py \
  --ac attention_only --name saved_0 --gen-tokens 65536 --filter-oversized --warmup 10 --steps 30
```

`--gen-tokens` jointly changes packing and fixed GEN budgets; UND remains 3072.
`--dryrun` resolves configuration without training. `--audit --warmup 0 --steps 1`
captures **all** trainable parameter shards, clipped gradients and updated
weights plus noising/input fingerprints; its I/O is excluded from performance
runs. `--nsys` captures rank 0 for two complete optimizer steps after warmup,
while all eight ranks train normally. The existing
`tools/unified_attn/summarize.py` and `review_nsys.py` accept these run directories.

Dispatcher nesting under selective AC can inflate profiler API-event counts.
The GPU probe checks actual `flash_attn_3::fwd` extension executions and compares
GEMM dispatch counts between the two selective policies; real nsys kernels are
the final evidence for the full training workload.

The original GEN98304 budget OOMs with attention-only caching on eight H100s;
GEN65536 passes the initial capacity check. Use the same budget to isolate the AC effect. For a final configuration
comparison, measure effective tokens/s across the actual budgets and document
changes in sample admission. The raw online entry retains its original length.

The shortened budget also needs `--filter-oversized`: the online dataset contains
individual videos exceeding 64K GEN tokens. This experiment adapter admits only
whole samples fitting the exact UND/GEN budgets (including the required padding
token), preserving stream order and latent contents. Production admission still
raises on oversized samples. `capacity/*.jsonl` records skipped IDs, URLs and
sizes; counts include loader prefetch and are not counts of consumed optimizer
steps. Formal A/B use this flag identically and verify consumed input/RNG hashes.

`--deterministic-attention` enables diagnostic Torch/FA3 determinism while
retaining compile; `--eager-audit` additionally disables language compilation.
Both require `--audit` and are excluded from performance measurements. The
default BF16 training path has substantial update-repeat variability; even
compiled deterministic full vs SAC has measurable numerical differences that
have not been established to be harmless rounding. See
the record for exact numerical evidence before choosing a production policy.

## Measured outcome (8 H100s)

The final configuration comparison is **96K full AC → 64K attention-only SAC**.
One added 96K run (10 warmup + 30 measured steps), compared with the existing
three 64K SAC runs, gives effective GEN throughput **30,869.2 → 34,356.4 tokens/s
(+11.30%)**. Peak allocated memory is **64.22 → 64.50 GiB**; reserved is
**70.70 → 70.04 GiB**. Mean step time is 43.111 → 25.979 seconds, but the
smaller token count per step means that ratio is not the throughput gain.
Production source hashes, dependency versions, dataset source and starting
checkpoint match. The 64K candidate filters oversized whole samples, so this
comparison measures the complete configuration change, not AC alone.

Reproduce the added baseline with a new run name (no `--filter-oversized`):

```bash
LD_LIBRARY_PATH='' .venv/bin/python tools/ac_attn/run.py \
  --ac full --name final96k_full_new --gen-tokens 98304 --warmup 10 --steps 30
LD_LIBRARY_PATH='' .venv/bin/python tools/unified_attn/summarize.py \
  /tmp/cosmos-ac-attn/runs/final96k_full_new
```

The same-budget experiment below isolates the AC effect:

Three independent 10-warmup/30-measurement runs per policy at UND3072 +
GEN65536, GA2, shard=8: median-of-run-medians 28.457 → 25.091 seconds
(1.134×), effective GEN throughput +14.79%, peak allocated 47.30 → 64.50 GiB.
This uses the documented whole-sample capacity filter; it is not a 96K result.

Real rank-zero Nsight captures verify FA replay 288 → 0, original FA forward
and backward 288 each, and identical GEMM kernel names/counts by phase.
After exporting reports with `tools/unified_attn/review_nsys.py` and
`trace_breakdown.py`, run `kernel_audit.py FULL_RUN ATTENTION_RUN --output audit.json`.

Deterministic eager full/SAC gradients and updates are bitwise equal across all
405 trainable tensors on eight ranks. Deterministic compiled comparison has
gradient relative L2 0.0165075 and update relative L2 0.1066044; it has **not**
passed strict numerical equivalence or long-term convergence validation. The
production default remains full. Full raw evidence and decisions are in the record.
