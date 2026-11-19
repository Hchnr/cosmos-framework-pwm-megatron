# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tools.tp_pp_full_width_row_diagnostic import full_row


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(401)
    weight_full = torch.randn(7, world * 5)
    ref = torch.nn.Parameter(weight_full.clone())
    local = torch.nn.Parameter(weight_full[:, rank * 5 : (rank + 1) * 5].clone())
    optimizer = torch.optim.AdamW([local], lr=0.001)
    ref_optimizer = torch.optim.AdamW([ref], lr=0.001)
    torch.manual_seed(717)
    max_error = 0.0
    for step in range(4):
        x = torch.randn(11, world * 5, requires_grad=True)
        x_local = x.detach()[:, rank * 5 : (rank + 1) * 5].clone().requires_grad_()
        dy = torch.randn(11, 7)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        expected, actual = F.linear(x, ref), full_row(x_local, local)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (expected * dy).sum().backward()
        (actual * dy).sum().backward()
        torch.testing.assert_close(x_local.grad, x.grad[:, rank * 5 : (rank + 1) * 5], rtol=0, atol=0)
        torch.testing.assert_close(local.grad, ref.grad[:, rank * 5 : (rank + 1) * 5], rtol=0, atol=0)
        optimizer.step()
        ref_optimizer.step()
        torch.testing.assert_close(local, ref[:, rank * 5 : (rank + 1) * 5], rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[local][key], ref_optimizer.state[ref][key][:, rank * 5 : (rank + 1) * 5], rtol=0, atol=0
            )
        assert optimizer.state[local]["step"] == ref_optimizer.state[ref]["step"]
    optimizer.zero_grad(set_to_none=True)
    # All ranks omit the same branch; the parameter and optimizer state stay untouched.
    discarded = full_row(torch.ones(3, 5), local)
    saved_parameter = local.detach().clone()
    saved_state = {key: value.clone() for key, value in optimizer.state[local].items()}
    unrelated = torch.ones((), requires_grad=True)
    unrelated.backward()
    assert local.grad is None and discarded.requires_grad
    optimizer.step()
    assert torch.equal(local, saved_parameter)
    assert all(torch.equal(optimizer.state[local][key], value) for key, value in saved_state.items())
    try:
        full_row(torch.ones(3, 5, requires_grad=True), local).mul(float(rank + 1)).sum().backward()
    except RuntimeError as error:
        assert "gradient differs" in str(error)
    else:
        raise AssertionError("Rank-divergent dY must fail")
    Path(os.environ["REPORT_DIR"], f"rank{rank}.json").write_text(
        json.dumps({"passed": True, "world": world, "updates": 4, "max_error": max_error}) + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
