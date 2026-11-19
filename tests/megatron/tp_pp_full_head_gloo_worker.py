import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tools.tp_pp_full_head_attention_experiment import full_head_attention


def attend(q, k, v, mask, backend="sdpa", *, frozen_und=False):
    assert backend == "sdpa"
    if q.shape[0] == 0:
        return q.flatten(1) + (k.sum() + v.sum()) * 0
    with sdpa_kernel(SDPBackend.MATH):
        value = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            attn_mask=mask[None, None],
            enable_gqa=True,
            scale=q.shape[-1] ** -0.5,
        )
    return value[0].transpose(0, 1).flatten(1)


def run():
    dist.init_process_group("gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    torch.manual_seed(2026)
    q0, k0, v0 = (
        torch.randn(3, size * 2, 4),
        torch.randn(5, size, 4),
        torch.randn(5, size, 4),
    )
    dy = torch.randn(3, size * 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0, -1] = False
    checks = 0
    for pattern in range(8):
        needed = [bool(pattern & (1 << index)) for index in range(3)]
        full = [value.clone().requires_grad_(flag) for value, flag in zip((q0, k0, v0), needed, strict=True)]
        local = [
            value[:, rank * heads : (rank + 1) * heads].clone().requires_grad_(flag)
            for value, heads, flag in zip((q0, k0, v0), (2, 1, 1), needed, strict=True)
        ]
        native = attend(*full, mask)
        actual = full_head_attention(*local, mask, group, attend)
        torch.testing.assert_close(actual, native[:, rank * 8 : (rank + 1) * 8], rtol=0, atol=0)
        if any(needed):
            native.backward(dy)
            actual.backward(dy[:, rank * 8 : (rank + 1) * 8].contiguous())
            for f, local_value, heads, flag in zip(full, local, (2, 1, 1), needed, strict=True):
                if flag:
                    torch.testing.assert_close(
                        local_value.grad,
                        f.grad[:, rank * heads : (rank + 1) * heads],
                        rtol=0,
                        atol=0,
                    )
                else:
                    assert local_value.grad is None
        checks += 1

    # An entirely unused attention branch must leave all parameter gradients
    # absent, not manufacture zeros that decay an Adam moment.
    unused = [
        torch.nn.Parameter(value[:, rank * h : (rank + 1) * h].clone())
        for value, h in zip((q0, k0, v0), (2, 1, 1), strict=True)
    ]
    optimizer = torch.optim.AdamW(unused, lr=0.02, weight_decay=0.1)
    sum(value.square().sum() for value in unused).backward()
    optimizer.step()
    previous = [
        (
            value.detach().clone(),
            {key: tensor.clone() for key, tensor in optimizer.state[value].items()},
        )
        for value in unused
    ]
    optimizer.zero_grad(set_to_none=True)
    full_head_attention(*unused, mask, group, attend)
    used = q0[:, rank * 2 : (rank + 1) * 2].clone().requires_grad_()
    key, val = (k0[:, rank : rank + 1].clone(), v0[:, rank : rank + 1].clone())
    full_head_attention(used, key, val, mask, group, attend).sum().backward()
    assert all(value.grad is None for value in unused)
    optimizer.step()
    for value, (expected, old_state) in zip(unused, previous, strict=True):
        assert torch.equal(value, expected)
        assert all(torch.equal(optimizer.state[value][key], old) for key, old in old_state.items())

    # Empty query is a valid missing-modality case: preserve zero gradients
    # and avoid a zero-sized NCCL all-gather.
    empty = q0[:0, rank * 2 : (rank + 1) * 2].clone().requires_grad_()
    ek = k0[:, rank : rank + 1].clone().requires_grad_()
    ev = v0[:, rank : rank + 1].clone().requires_grad_()
    full_head_attention(empty, ek, ev, mask[:0], group, attend).sum().backward()
    assert torch.count_nonzero(ek.grad) == 0 and torch.count_nonzero(ev.grad) == 0

    wrong_mask = mask.clone()
    if rank == 1:
        wrong_mask[0, 0] = False
    try:
        full_head_attention(used.detach(), key, val, wrong_mask, group, attend)
    except ValueError as error:
        assert "masks differ" in str(error)
    else:
        raise AssertionError("Rank-divergent mask was accepted")
    different_grad_flag = q0[:, rank * 2 : (rank + 1) * 2].clone().requires_grad_(rank == 0)
    try:
        full_head_attention(different_grad_flag, key, val, mask, group, attend)
    except ValueError as error:
        assert "branch flags" in str(error)
    else:
        raise AssertionError("Rank-divergent gradient flags were accepted")
    path = Path(os.environ["ATTENTION_TEST_OUTPUT"]) / f"rank{rank}.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "rank": rank,
                "world": size,
                "gradient_patterns": checks,
                "unused_grad_none": True,
                "empty_query": True,
                "mask_mismatch_rejected": True,
            }
        )
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
