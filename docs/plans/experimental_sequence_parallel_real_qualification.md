# Experimental sequence-parallel qualification

This is an isolated experiment. It does not enable sequence parallelism in the
training backend, measure throughput, or qualify a production training run.

The worker composes the actual `FirstStageInputs` / `LastStageOutputs`, two
Qwen MoT layers, production attention, masks, RoPE and action blocks with
Megatron-sharded projections. Its synthetic two-sample inputs and squared-output
loss cover two microsteps and four FP32 AdamW updates, absent action, vision-only
targets and a no-target dummy. Both layers use full activation checkpointing.
The reference is the corresponding unsharded eager model, with frozen UND.

The sequence-sharded implementation gathers before column projections and uses
sum reduce-scatter after row projections. A replicated output head needs final
gather backward to take only its local slice; replicated dummy context needs a
1/TP backward factor before summing input-adapter gradients. Both mistakes are
tested as negative controls in the combined graph. Head-local Q/K norm and
sequence-local norm gradients each receive exactly one explicit TP sum. The
worker checks every parameter's gradient presence and value, Adam state, and
Python distributed API collective order and tensor shapes on every rank.

Run CPU tests with `pytest tests/megatron/tp_pp_sequence_parallel_real_test.py`.
Gloo and NCCL are distinct backend qualifications; the CPU test establishes no
NCCL result. The NCCL qualifier must run behind the cooperative shared GPU
lock after the synthetic NCCL prerequisite passes. Its tiny dimensions,
synthetic loss and eager SDPA are not the real 36-layer RF training workload;
BF16, compile, full model memory, correctness and fair TP/FSDP throughput
remain separate required checks.
