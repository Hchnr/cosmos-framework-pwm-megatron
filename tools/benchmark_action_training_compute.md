# Local Action128 compute diagnostics

`benchmark_action_training_compute.py` benchmarks the current small Action block
with independent UND/VIDEO/ACTION parameters. It uses random weights and synthetic
equal sample lengths, not a production checkpoint or real packed training inputs.
The default physical lengths are 5120 / 59648 / 2944, hidden widths 4096 / 4096 /
1280, and Q32/KV8/head_dim128. All branches and their inputs require gradients.
Seed is fixed to 42. The production long UND bucket is not covered by this default.

This single-GPU forward/backward diagnostic excludes the optimizer, FSDP,
communication, VAE, packing and dataloader. It cannot establish an end-to-end
training speedup or whole-model capacity. It never reads a production checkpoint,
uploads metrics, writes training checkpoints or signals another process.

## Timing and numerical comparison

Use an idle GPU and the repository training environment. Output and reference
paths must be new. Run performance cases sequentially on the same GPU, with
baseline rechecks. Compilation/metadata setup, warmup and reference I/O are
excluded from the reported timed iterations.

`--reduction-split-size default|8|16|32|64|128` explicitly controls
`triton.mix_order_reduction_split_size` in the compiled block. The tool default
leaves the Inductor heuristic unchanged; the production Action128 recipe sets
32. Pass `--reduction-split-size 32` when reproducing that production setting.
Earlier November 4 compute measurements used the Inductor default.

```bash
export LD_LIBRARY_PATH=''
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
source /share-mixed/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/activate

python tools/benchmark_action_training_compute.py \
  --kind block --variant split --ac full --compile \
  --samples 16 --warmup 4 --repeats 12 \
  --reference /tmp/action-compute-reference.pt --write-reference \
  --output /tmp/action-compute-full.json

python tools/benchmark_action_training_compute.py \
  --kind block --variant merge_va --ac full --compile \
  --samples 16 --warmup 4 --repeats 12 \
  --reference /tmp/action-compute-reference.pt \
  --output /tmp/action-compute-merged.json
```

Variants:

- `split`: calls the actual production `small_layer_forward`; VIDEO uses FA3,
  UND/ACTION use Flex.
- `merge_va`: benchmark-only implementation combining VIDEO/ACTION queries into
  one noncausal FA3 call, retaining separate causal UND. Parameters stay separate.
- `unified_flex`: benchmark-only joint call preserving sample isolation and
  causal-UND/full-generation visibility.
- `fa3_split`: benchmark-only common block implementation, with three separate
  FA3 calls. UND is causal and VIDEO/ACTION use the same full KV.
- `fa3_merge_va`: exactly the same block implementation and FA3 backend, but
  combines VIDEO/ACTION into one noncausal call; UND remains causal and separate.
  Use these last two variants for an FA3-only controlled comparison; neither
  is the current online mixed-backend implementation.
- `fa3_merge_perm`: benchmark-only merged variant using inverse-permutation
  gathers in backward, valid only because its two indices are bijections.
  `--skewed-samples` tests unequal synthetic sample lengths.

Policies: `full`, `save_attention` (save the observed FA3 forward op),
`save_downproj` (diagnostic shape-selective MLP downprojection policy), and `none`.
The attention-only `--kind attention` mode requires `--ac none`.

`--tiny --samples 4` performs a smaller full-block numerical check. Reference
comparison records every real output, input gradient and parameter gradient,
asserts matching tensor sets and finite values, and reports relative L2/max error.
It does not certify a universal error tolerance or training convergence. BF16
and compilation differences must be reviewed rather than called exact equality.

## Memory and Nsight

`--memory-probe --layers 4` measures forward retention with synchronization.
Multiple blocks share parameters so only activation-memory scaling is assessed.
These timings are excluded from performance claims. Measure retained activations
as well as peak allocation: a policy can reduce temporary peak allocation while
increasing what every preceding layer must retain until backward.

```bash
nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/action-compute-profile \
  python tools/benchmark_action_training_compute.py \
    --kind block --variant split --ac full --compile \
    --samples 16 --warmup 4 --repeats 3 --profile \
    --output /tmp/action-compute-profile.json
```

Profile runs provide kernel attribution only. Do not use CPU NVTX duration as GPU
execution time, or kernel-time percentages as full-training walltime shares.
Classify FA3 kernels before generic `cutlass`/GEMM names; FA3 also uses CUTLASS.

November 4 results and raw artifacts are in
`outputs/action_compute_20261104/`. The workspace report is
`plans/v0.3.1_dataloader/compute_report_1104.md`.
