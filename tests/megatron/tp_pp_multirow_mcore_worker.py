# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise actual MCore row-linear GEMMs and grouped SUM over Gloo."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

from tools.run_tp_pp_multirow_reduction import _row_group, install


def tensor_diff(a, b):
    delta = a.detach().to(torch.float64) - b.detach().to(torch.float64)
    denom = a.detach().to(torch.float64).norm().item()
    return {
        "relative_l2": delta.norm().item() / max(denom, 1e-30),
        "max_abs": delta.abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
    from cosmos_framework.model.generator.megatron_tp_pp.layers import (
        TensorParallelLinear,
    )

    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        tensor_model_parallel_size=world,
        pipeline_model_parallel_size=1,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
    )
    torch.manual_seed(812)
    sources = [
        torch.nn.Linear(width * world, out, bias=index == 1)
        for index, (width, out) in enumerate(((4, 6), (6, 8), (8, 4)))
    ]
    for parameter in sources[0].parameters():
        parameter.requires_grad_(False)
    baseline = torch.nn.ModuleList(
        [TensorParallelLinear(source, config, column=False, tp_rank=rank, tp_size=world) for source in sources]
    )
    install()
    candidate = torch.nn.ModuleList(
        [
            TensorParallelLinear(copy.deepcopy(source), config, column=False, tp_rank=rank, tp_size=world)
            for source in sources
        ]
    )
    for p, q in zip(baseline.parameters(), candidate.parameters(), strict=True):
        assert torch.equal(p, q) and p.requires_grad == q.requires_grad
    oldopt = torch.optim.AdamW(
        (p for p in baseline.parameters() if p.requires_grad),
        lr=0.003,
        weight_decay=0.01,
    )
    newopt = torch.optim.AdamW(
        (p for p in candidate.parameters() if p.requires_grad),
        lr=0.003,
        weight_decay=0.01,
    )
    cases = []
    for step, selected in enumerate(((1, 2), (1,), (2,), (1, 2))):
        oldopt.zero_grad(set_to_none=True)
        newopt.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(9000 + rank * 100 + step)
        local = [torch.randn(rows, width, generator=generator) for rows, width in ((3, 4), (5, 6), (2, 8))]
        old_inputs = [x.detach().clone().requires_grad_(i != 0) for i, x in enumerate(local)]
        new_inputs = [x.detach().clone().requires_grad_(i != 0) for i, x in enumerate(local)]
        expected = tuple(projection(value) for projection, value in zip(baseline, old_inputs, strict=True))
        actual = _row_group(candidate, new_inputs)
        assert not actual[0].requires_grad
        sum(expected[i].square().sum() for i in selected).backward()
        sum(actual[i].square().sum() for i in selected).backward()
        differences = []
        for p, q in zip(baseline.parameters(), candidate.parameters(), strict=True):
            if p.grad is None or q.grad is None:
                assert p.grad is None and q.grad is None
            else:
                differences.append(tensor_diff(p.grad, q.grad))
        for old, new in zip(old_inputs, new_inputs, strict=True):
            if old.grad is None or new.grad is None:
                assert old.grad is None and new.grad is None
            else:
                differences.append(tensor_diff(old.grad, new.grad))
        oldopt.step()
        newopt.step()
        for p, q in zip(baseline.parameters(), candidate.parameters(), strict=True):
            differences.append(tensor_diff(p, q))
            if p in oldopt.state or q in newopt.state:
                assert set(oldopt.state[p]) == set(newopt.state[q])
                differences.extend(tensor_diff(oldopt.state[p][k], newopt.state[q][k]) for k in oldopt.state[p])
        assert max(row["relative_l2"] for row in differences) < 1e-4
        cases.append(
            {
                "step": step,
                "selected": list(selected),
                "max_relative_l2": max(row["relative_l2"] for row in differences),
                "max_abs": max(row["max_abs"] for row in differences),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "world": world, "cases": cases}) + "\n")
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
