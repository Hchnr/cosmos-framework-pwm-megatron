# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare the real three-stream MCore layer with grouped row sums on CPU."""

import argparse
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from tp_pp_multirow_mcore_worker import tensor_diff

from tools.run_tp_pp_multirow_reduction import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--full-ac", action="store_true")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--compile-backend", choices=("aot_eager", "inductor"), default="aot_eager")
    args = parser.parse_args()
    sys.path.insert(0, str(args.worktree / "tests/megatron"))
    from tp_pp_network_io_test import make_native, make_sequence

    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs
    from cosmos_framework.model.generator.megatron_tp_pp.layers import parallelize_layer
    from cosmos_framework.model.generator.mot import small_action_tower as tower

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 4):
        raise ValueError("Only TP2 and TP4 are covered")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
    torch.manual_seed(521)
    native, shape = make_native()
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=world,
        pipeline_model_parallel_size=1,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
    )
    first = FirstStageInputs(
        native.config,
        native.language_model.config,
        None,
        shape=shape,
        vocab_parallel=False,
    )
    first.to_empty(device="cpu")
    source = native.state_dict()
    first.load_state_dict({key: source[key] for key in first.state_dict()}, strict=True)
    source_layer = copy.deepcopy(native.language_model.model.layers[0])
    if world == 4:
        # The stock tiny fixture has only two KV heads. Increase to four to
        # exercise a valid TP4 shape without changing its three-stream logic.
        source_layer.self_attn.num_key_value_heads = 4
        source_layer.self_attn.num_key_value_groups = 1
        for suffix in ("", "_moe_gen"):
            for name in ("k_proj", "v_proj"):
                old = getattr(source_layer.self_attn, name + suffix)
                replacement = torch.nn.Linear(old.in_features, 64, bias=old.bias is not None)
                replacement.weight.requires_grad_(old.weight.requires_grad)
                setattr(source_layer.self_attn, name + suffix, replacement)
        block = source_layer.action_block_moe_gen_action
        block.shape = dataclasses.replace(block.shape, kv_heads=4)
        for name in ("k_proj", "v_proj"):
            old = getattr(block, name)
            replacement = torch.nn.Linear(old.in_features, 64, bias=old.bias is not None)
            replacement.weight.requires_grad_(old.weight.requires_grad)
            setattr(block, name, replacement)
    baseline = parallelize_layer(copy.deepcopy(source_layer), config, rank, world)
    original = tower.small_layer_forward
    install()
    candidate = parallelize_layer(copy.deepcopy(source_layer), config, rank, world)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    baseline.to(device=device, dtype=dtype)
    candidate.to(device=device, dtype=dtype)
    assert [n for n, _ in baseline.named_parameters()] == [n for n, _ in candidate.named_parameters()]
    # Keep FP32 masters and moments even when compute weights are BF16, as in
    # the production optimizer. This remains torch AdamW, not MCore FusedAdam.
    oldparams = {name: p for name, p in baseline.named_parameters() if p.requires_grad}
    newparams = {name: p for name, p in candidate.named_parameters() if p.requires_grad}
    oldmasters = {name: torch.nn.Parameter(p.detach().float().clone()) for name, p in oldparams.items()}
    newmasters = {name: torch.nn.Parameter(p.detach().float().clone()) for name, p in newparams.items()}
    oldopt = torch.optim.AdamW(oldmasters.values(), lr=0.001, weight_decay=0.01)
    newopt = torch.optim.AdamW(newmasters.values(), lr=0.001, weight_decay=0.01)
    reports = []
    real_all_reduce = torch.ops._c10d_functional.all_reduce
    counts = {"value": 0}

    def counted_all_reduce(*positional, **keywords):
        counts["value"] += 1
        return real_all_reduce(*positional, **keywords)

    if not args.compile:
        torch.ops._c10d_functional.all_reduce = counted_all_reduce

    def source_forward(pack):
        return original(baseline, pack)[0]

    def run_candidate(pack):
        return tower.small_layer_forward(candidate, pack)[0]

    def checkpointed(function):
        def run(pack):
            from torch.utils.checkpoint import checkpoint

            def inner(u, v, a):
                return function({**pack, "causal_seq": u, "full_only_seq": v, "action_seq": a})

            return checkpoint(inner, pack["causal_seq"], pack["full_only_seq"], pack["action_seq"], use_reentrant=False)

        return run

    if args.full_ac:
        source_forward, run_candidate = checkpointed(source_forward), checkpointed(run_candidate)
    if args.compile:
        source_forward = torch.compile(source_forward, backend=args.compile_backend, fullgraph=True, dynamic=False)
        run_candidate = torch.compile(run_candidate, backend=args.compile_backend, fullgraph=True, dynamic=False)
    for step, (has_action, selected) in enumerate(
        (
            (True, ("full_only_seq", "action_seq")),
            (True, ("full_only_seq",)),
            (True, ("action_seq",)),
            (False, ("full_only_seq",)),
        )
    ):
        torch.manual_seed(9800 + step)
        packed = first({"packed_sequence": make_sequence(action=has_action)})
        from torch.utils import _pytree

        packed = _pytree.tree_map(lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, packed)
        for name in ("_small_rope", "_small_action_rope"):
            packed[name] = tuple(tuple(value.to(dtype) for value in pair) for pair in packed[name])
        old, new = dict(packed), dict(packed)
        for name in ("causal_seq", "full_only_seq", "action_seq"):
            source = packed[name].detach().to(dtype)
            old[name] = source.clone().requires_grad_(name != "causal_seq")
            new[name] = source.clone().requires_grad_(name != "causal_seq")
        oldopt.zero_grad(set_to_none=True)
        newopt.zero_grad(set_to_none=True)
        baseline.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        before = counts["value"]
        expected = source_forward(old)
        baseline_forward = None if args.compile else counts["value"] - before
        before = counts["value"]
        actual = run_candidate(new)
        candidate_forward = None if args.compile else counts["value"] - before
        if not args.compile:
            assert baseline_forward == 6 and candidate_forward == 2, (baseline_forward, candidate_forward)
        sum(expected[name].square().sum() for name in selected).backward()
        sum(actual[name].square().sum() for name in selected).backward()
        differences = [dict(name=f"output.{name}", **tensor_diff(expected[name], actual[name])) for name in selected]
        for (name, p), (other_name, q) in zip(baseline.named_parameters(), candidate.named_parameters(), strict=True):
            assert name == other_name
            if p.grad is None or q.grad is None:
                assert p.grad is None and q.grad is None, name
            else:
                differences.append(dict(name=f"gradient.{name}", **tensor_diff(p.grad, q.grad)))
        for params, masters in ((oldparams, oldmasters), (newparams, newmasters)):
            for name, parameter in params.items():
                masters[name].grad = parameter.grad.float().clone() if parameter.grad is not None else None
        oldopt.step()
        newopt.step()
        with torch.no_grad():
            for params, masters in ((oldparams, oldmasters), (newparams, newmasters)):
                for name, parameter in params.items():
                    parameter.copy_(masters[name].to(parameter.dtype))
        for (name, p), (_, q) in zip(baseline.named_parameters(), candidate.named_parameters(), strict=True):
            differences.append(dict(name=f"parameter.{name}", **tensor_diff(p, q)))
            if name in oldmasters:
                oldmaster, newmaster = oldmasters[name], newmasters[name]
                differences.append(dict(name=f"master.{name}", **tensor_diff(oldmaster, newmaster)))
                assert set(oldopt.state[oldmaster]) == set(newopt.state[newmaster])
                differences.extend(
                    dict(
                        name=f"state.{key}.{name}",
                        **tensor_diff(oldopt.state[oldmaster][key], newopt.state[newmaster][key]),
                    )
                    for key in oldopt.state[oldmaster]
                )
        maximum = max(row["relative_l2"] for row in differences)
        failures = [row for row in differences if row["relative_l2"] > 1e-4 and row["max_abs"] > 2e-6]
        failed = torch.tensor([bool(failures)], device=device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        reports.append(
            {
                "step": step,
                "selected": selected,
                "has_action": has_action,
                "max_relative_l2": maximum,
                "baseline_forward_sums": baseline_forward,
                "candidate_forward_sums": candidate_forward,
                "compile_backend": args.compile_backend if args.compile else None,
                "full_activation_checkpointing": args.full_ac,
                "dtype": args.dtype,
                "passed": not bool(failed.item()),
                "local_failed_fields": len(failures),
                "largest_relative_fields": sorted(differences, key=lambda row: row["relative_l2"], reverse=True)[:10],
            }
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "cases": reports}) + "\n")
        assert not failed.item(), f"fixed numerical gate failed; rank={rank}, local failures={len(failures)}"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "cases": reports,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            }
        )
        + "\n"
    )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
