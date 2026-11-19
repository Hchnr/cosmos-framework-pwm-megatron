# Experimental multirow TP reduction

This opt-in diagnostic groups the three independent row-parallel attention
output reductions and, separately, the three MLP down-projection reductions.
It leaves MCore local GEMMs and parameter objects separate. The training
default is unchanged. All TP ranks must enter with identical stream shapes,
dtype, branch order and group; inference caches and layers without the action
tower are explicitly unsupported. Installation must occur before constructing
the TP stage. Install shared-input reduction first when combining candidates.

In a one-layer production-I/O fixture, CPU/Gloo TP2 and TP4 completed four
FP32 AdamW updates, including absent action and unused output branches. Eager
forward SUM counts were 6 before and 2 after per layer; TP2 was bitwise equal
and TP4 maximum tensor relative L2 was 9.18e-7 in the original run. TP2/TP4
full-AC fullgraph AOT-eager also passed. These are not CUDA kernel counts,
performance measurements, FusedAdam results or 36-layer acceptance.

**BF16 CPU/Gloo failed the fixed numerical gate** on the first TP4 update,
including when compute parameters used FP32 masters and FP32 AdamW moments.
The four ranks had 67, 74, 70 and 58 failing fields; maximum tensor relative
L2 was 0.02131. The CPU optimizer is not production FusedAdam, but this result
is sufficient to keep the candidate disabled. A controlled Gloo replay found
that packing/unpacking was bitwise exact before communication; separate and
packed `all_reduce` on the same local inputs produced different BF16 results.
The changed message layout/collective arithmetic is therefore a candidate
cause on Gloo; the NCCL result is unknown. Do not relax the fixed numerical
gate or infer a speedup from fewer collective calls.

Reports and failure logs are retained under
`outputs/megatron_tp4_fsdp4_microbatch_20261119/multirow_cpu_20261119/`.
`tools/qualify_tp_pp_multirow_reduction.py` prepares the independent TP2/TP4
FP32 production-I/O CUDA qualification; profiler counts are diagnostic only.
BF16 NCCL, actual 36-layer multi-step numerical acceptance, and fair TP4 vs
FSDP4 measurements remain outstanding.
