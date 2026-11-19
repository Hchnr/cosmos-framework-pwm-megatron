# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare grouped vs independent row reductions with real Gloo collectives."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tools.tp_pp_multirow_reduction import grouped_row_linears


class SeparateSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.set_materialize_grads(False)
        result = value.clone().contiguous()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def make_parameters(rank, dtype, disconnected):
    generator = torch.Generator().manual_seed(701 + rank)
    result = []
    for index, (rows, in_features, out_features) in enumerate(((3, 4, 6), (5, 6, 8), (2, 8, 4))):
        # Exercise noncontiguous input and mixed projected shapes, including
        # an entirely frozen first stream.
        source = torch.randn(rows, in_features * 2, generator=generator, dtype=dtype)[:, ::2]
        input_value = source.detach().requires_grad_(index != 0)
        weight = torch.randn(out_features, in_features, generator=generator, dtype=dtype).requires_grad_(index != 0)
        bias = None if index != 1 else torch.randn(out_features, generator=generator, dtype=dtype).requires_grad_()
        result.append((input_value, weight, bias))
    return result


def make_inputs(parameters):
    def clone(value):
        if value is None:
            return None
        if value.ndim == 2 and not value.is_contiguous():
            storage = value.new_empty(value.shape[0], value.shape[1] * 2)
            result = storage[:, ::2]
            result.copy_(value.detach())
        else:
            result = value.detach().clone()
        return result.requires_grad_(value.requires_grad)

    return [tuple(clone(value) for value in row) for row in parameters]


def execute(parameters, group, selected, grouped):
    inputs = [row[0] for row in parameters]
    weights = [row[1] for row in parameters]
    biases = [row[2] for row in parameters]
    if grouped:
        outputs = grouped_row_linears(inputs, weights, biases, group)
    else:
        outputs = tuple(
            SeparateSum.apply(F.linear(x, w), group) + b if b is not None else SeparateSum.apply(F.linear(x, w), group)
            for x, w, b in parameters
        )
    # An unused result is deliberately omitted: its trainable parameters must
    # remain grad=None rather than receive zeros and incur Adam weight decay.
    loss = sum(outputs[index].square().sum() * (index + 1) for index in selected)
    loss.backward()
    return outputs


def check_values(left, right, tolerance):
    delta = left.detach().to(torch.float64) - right.detach().to(torch.float64)
    denominator = left.detach().to(torch.float64).norm().item()
    relative = delta.norm().item() / max(denominator, 1e-30)
    maximum = delta.abs().max().item() if delta.numel() else 0.0
    if relative > tolerance and maximum > 1e-6:
        raise AssertionError(f"Grouped reduction numerical discrepancy: relative={relative}, max_abs={maximum}")
    return {"relative_l2": relative, "max_abs": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world, group = dist.get_rank(), dist.get_world_size(), dist.group.WORLD
    assert world in (2, 4)
    reports = []
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        for selected in ((1,), (2,), (1, 2)):
            original = make_parameters(rank, dtype, selected)
            independent, combined = make_inputs(original), make_inputs(original)
            assert all(not row[0].is_contiguous() for row in independent + combined)
            baseline = execute(independent, group, selected, False)
            candidate = execute(combined, group, selected, True)
            tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-4 if dtype == torch.float32 else 1e-10
            checks = []
            for a, b in zip(baseline, candidate, strict=True):
                checks.append(check_values(a, b, tolerance))
            for old, new in zip(independent, combined, strict=True):
                for x, y in zip(old, new, strict=True):
                    if x is not None:
                        if x.grad is None or y.grad is None:
                            assert x.grad is None and y.grad is None
                        else:
                            checks.append(check_values(x.grad, y.grad, tolerance))
            assert not candidate[0].requires_grad
            inactive = 2 if selected == (1,) else 1 if selected == (2,) else None
            if inactive is not None:
                assert combined[inactive][0].grad is None
                assert combined[inactive][1].grad is None
                if combined[inactive][2] is not None:
                    assert combined[inactive][2].grad is None
            reports.append(
                {
                    "dtype": str(dtype),
                    "selected": list(selected),
                    "passed": True,
                    "max_relative_l2": max(check["relative_l2"] for check in checks),
                    "max_absolute_error": max(check["max_abs"] for check in checks),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "world": world, "cases": reports}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
