# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate diagnostic gathered-weight gradients with no extra TP factor."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as functional

from tools.run_tp_pp_native_block_diagnostic import GatherNativeWeight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(818)
    full = nn.ParameterList(
        [nn.Parameter(torch.randn(32, 16, device="cuda") * 0.1), nn.Parameter(torch.randn(12, 32, device="cuda") * 0.1)]
    )
    axes = (0, 1)
    shards = nn.ParameterList(
        [
            nn.Parameter(value.detach().chunk(size, dim=axis)[rank].clone())
            for value, axis in zip(full, axes, strict=True)
        ]
    )
    reference = torch.optim.AdamW(full, lr=1e-3, foreach=False)
    candidate = torch.optim.AdamW(shards, lr=1e-3, foreach=False)
    for owner, local, axis in zip(full, shards, axes, strict=True):
        mean, variance = torch.randn_like(owner) * 1e-4, torch.rand_like(owner) * 1e-4
        reference.state[owner] = dict(step=torch.tensor(1200.0), exp_avg=mean, exp_avg_sq=variance)
        candidate.state[local] = dict(
            step=torch.tensor(1200.0),
            exp_avg=mean.chunk(size, dim=axis)[rank].clone(),
            exp_avg_sq=variance.chunk(size, dim=axis)[rank].clone(),
        )
    for update in range(3):
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        inputs = torch.randn(9, 16, device="cuda", requires_grad=True)
        replay = inputs.detach().clone().requires_grad_()
        weights = [
            GatherNativeWeight.apply(parameter, axis, dist.group.WORLD)
            for parameter, axis in zip(shards, axes, strict=True)
        ]
        expected = functional.linear(functional.silu(functional.linear(inputs, full[0])), full[1])
        actual = functional.linear(functional.silu(functional.linear(replay, weights[0])), weights[1])
        assert torch.equal(expected, actual)
        expected.square().mean().backward()
        actual.square().mean().backward()
        assert torch.equal(inputs.grad, replay.grad)
        for owner, local, axis in zip(full, shards, axes, strict=True):
            assert torch.equal(owner.grad.chunk(size, dim=axis)[rank], local.grad)
        reference.step()
        candidate.step()
        for owner, local, axis in zip(full, shards, axes, strict=True):
            assert torch.equal(owner.chunk(size, dim=axis)[rank], local)
            for key in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(reference.state[owner][key].chunk(size, dim=axis)[rank], candidate.state[local][key])
    args.output.with_suffix(f".rank{rank}.json").write_text(
        json.dumps(dict(passed=True, updates=3, full_tp_acceptance=False)) + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
