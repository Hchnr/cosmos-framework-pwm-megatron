# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU/Gloo two-rank autograd qualification for the reconstructed K/V diagnostic."""

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from tools.tp_pp_fp32_reconstructed_kv import reconstructed_linear


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(24)
    original_x = torch.randn(5, 6)
    original_weight = torch.randn(8, 6)
    original_bias = torch.randn(8)
    upstream = torch.randn(5, 8)
    results = []
    for train_input, train_weight, use_bias in ((True, True, True), (True, False, False), (False, True, True)):
        x = original_x.clone().requires_grad_(train_input)
        weight = original_weight.chunk(2)[rank].clone().requires_grad_(train_weight)
        bias = original_bias.chunk(2)[rank].clone().requires_grad_(train_weight) if use_bias else None
        native_x = original_x.clone().requires_grad_(train_input)
        native_weight = original_weight.clone().requires_grad_(train_weight)
        native_bias = original_bias.clone().requires_grad_(train_weight) if use_bias else None
        value = reconstructed_linear(x, weight, bias, dist.group.WORLD)
        reference = F.linear(native_x, native_weight, native_bias)
        torch.testing.assert_close(value, reference.chunk(2, dim=-1)[rank], rtol=0, atol=0)
        (value * upstream.chunk(2, dim=-1)[rank]).sum().backward()
        (reference * upstream).sum().backward()
        if train_input:
            torch.testing.assert_close(x.grad, native_x.grad, rtol=1e-6, atol=1e-6)
        if train_weight:
            torch.testing.assert_close(weight.grad, native_weight.grad.chunk(2)[rank], rtol=1e-6, atol=1e-6)
            if bias is not None:
                torch.testing.assert_close(bias.grad, native_bias.grad.chunk(2)[rank], rtol=1e-6, atol=1e-6)
        else:
            assert weight.grad is None
        results.append({"input_grad": train_input, "weight_grad": train_weight, "bias": use_bias})
    missing_weight = torch.nn.Parameter(original_weight.chunk(2)[rank].clone())
    optimizer = torch.optim.AdamW([missing_weight], lr=0.01, weight_decay=0.1)
    missing_weight.grad = torch.ones_like(missing_weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = missing_weight.detach().clone()
    before_step = optimizer.state[missing_weight]["step"].clone()
    missing_x = original_x.clone().requires_grad_(True)
    ignored = reconstructed_linear(missing_x, missing_weight, None, dist.group.WORLD)
    used_weight = original_weight.chunk(2)[rank].clone().requires_grad_()
    used = reconstructed_linear(missing_x, used_weight, None, dist.group.WORLD)
    used.sum().backward()
    assert ignored.shape[-1] == 4 and missing_weight.grad is None
    optimizer.step()
    assert torch.equal(before, missing_weight) and torch.equal(before_step, optimizer.state[missing_weight]["step"])
    output = Path(sys.argv[1]) / f"rank{rank}.json"
    output.write_text(json.dumps({"rank": rank, "cases": results, "missing_grad_none_and_adam_skip": True}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
