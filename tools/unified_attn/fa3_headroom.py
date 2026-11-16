# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Paired FA3 cost audit. Prepacked leaves are a diagnostic boundary, not a training replacement."""

import argparse
import json
import os
import random
import statistics
from pathlib import Path

import torch
from flash_attn_interface import flash_attn_varlen_func as fa3
from flash_attn_interface import flash_attn_varlen_grouped_func as grouped

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_all_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.joint_flash_attention import joint_flash_attention, prepare_flash_metadata

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--geometry", type=Path, required=True)
parser.add_argument("--case", choices=["two", "three"], required=True)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--samples", type=int, default=30)
args = parser.parse_args()
rank = int(os.environ["LOCAL_RANK"])
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/cosmos-unified-attn/headroom-{args.output.name}/rank{rank}"
torch.cuda.set_device(rank)
torch.manual_seed(719 + rank)
for fn in (fa3, grouped):
    torch._dynamo.allow_in_graph(fn)


def cum(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)


def error(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


if args.case == "two":
    geo = json.loads((args.geometry / f"geometry_rank{rank}.json").read_text())
    ui, gi, offset = [], [], 0
    for n, mode in zip(geo["split_lens"], geo["attn_modes"], strict=True):
        (ui if mode == "causal" else gi).extend(range(offset, offset + n))
        offset += n
    pack = sequence_pack_from_packed_sequence(
        torch.zeros(offset, 1, device="cuda"),
        geo["attn_modes"],
        geo["split_lens"],
        geo["sample_lens"],
        torch.tensor(ui, device="cuda"),
        torch.tensor(gi, device="cuda"),
        fixed_causal_len=3072,
        fixed_full_len=98304,
    )
    metadata = prepare_flash_metadata(pack, "unified_fa3_grouped")
    lengths = [3072, 98304]
    source = tuple(
        torch.randn(n, h, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h in (32, 8, 8)
        for n in lengths
    )
    dos = tuple(torch.randn(n, 32, 128, device="cuda", dtype=torch.bfloat16) for n in lengths)
    dos[0][len(ui) :] = 0
    dos[1][len(gi) :] = 0

    def packed(x):
        q0, q1, k0, k1, v0, v1 = x
        ka = get_all_seq(from_und_gen_splits(k0, k1, pack), include_padding=True)
        va = get_all_seq(from_und_gen_splits(v0, v1, pack), include_padding=True)
        return q0, q1, k0, v0, ka, va

    def split(*x):
        q0, q1, k0, v0, ka, va = packed(x)
        return tuple(joint_flash_attention((q0, q1), k0, v0, ka, va, {**metadata, "backend": "split_fa3_fork"}, None))

    def joint(*x):
        q0, q1, k0, v0, ka, va = packed(x)
        return tuple(
            joint_flash_attention((q0, q1), k0, v0, ka, va, {**metadata, "backend": "unified_fa3_grouped"}, None)
        )

    q0, q1, k0, v0, ka, va = packed(source)
    leaves = tuple(t.detach().requires_grad_() for t in (torch.cat((q0, q1)), torch.cat((k0, ka)), torch.cat((v0, va))))

    def prepacked(q, k, v):
        out = grouped(
            q, k, v, metadata["cu_q"], metadata["cu_k"], metadata["max_g"], metadata["max_all"], metadata["flags_bool"]
        )
        return out.split(lengths)

    functions = {"split": (split, source), "grouped": (joint, source), "prepacked_grouped": (prepacked, leaves)}
else:
    # Capacity-matched synthetic 92-sample case, NOT recorded action training data.
    u = [128] * 92
    v = [649] * 32 + [648] * 60
    a = [32] * 92
    lengths = [sum(u), sum(v), sum(a)]
    geo = {
        "kind": "synthetic_capacity_matched",
        "samples": 92,
        "und": u,
        "video": v,
        "action": a,
        "shared_kv_and_identical_rope_assumed": True,
    }
    source = tuple(
        torch.randn(n, h, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h in (32, 8, 8)
        for n in lengths
    )
    dos = tuple(torch.randn(n, 32, 128, device="cuda", dtype=torch.bfloat16) for n in lengths)
    cu = [cum(n) for n in (u, v, a)]
    sample_lens = [x + y + z for x, y, z in zip(u, v, a, strict=True)]
    ck = cum(sample_lens)
    order, vaorder, undo_v, undo_a = [], [], [], []
    pu = pv = pa = pva = 0
    for nu, nv, na in zip(u, v, a, strict=True):
        order.extend(range(pu, pu + nu))
        order.extend(range(lengths[0] + pv, lengths[0] + pv + nv))
        order.extend(range(lengths[0] + lengths[1] + pa, lengths[0] + lengths[1] + pa + na))
        vaorder.extend(range(pv, pv + nv))
        vaorder.extend(range(lengths[1] + pa, lengths[1] + pa + na))
        undo_v.extend(range(pva, pva + nv))
        undo_a.extend(range(pva + nv, pva + nv + na))
        pu += nu
        pv += nv
        pa += na
        pva += nv + na
    order, vaorder, undo_v, undo_a = [torch.tensor(t, device="cuda") for t in (order, vaorder, undo_v, undo_a)]
    cq3 = cum(u + v + a)
    ck3 = cum(u + sample_lens + sample_lens)
    flags3 = torch.tensor([True] * 92 + [False] * 184, device="cuda")
    cq2 = cum(u + [x + y for x, y in zip(v, a, strict=True)])
    ck2 = cum(u + sample_lens)
    flags2 = torch.tensor([True] * 92 + [False] * 92, device="cuda")

    def split(*x):
        q = x[:3]
        k = x[3:6]
        val = x[6:9]
        ka, va = torch.cat(k)[order], torch.cat(val)[order]
        return (
            fa3(q[0], k[0], val[0], cu[0], cu[0], max(u), max(u), causal=True),
            fa3(q[1], ka, va, cu[1], ck, max(v), max(sample_lens)),
            fa3(q[2], ka, va, cu[2], ck, max(a), max(sample_lens)),
        )

    def joint(*x):
        q = torch.cat(x[:3])
        k = torch.cat(x[3:6])[order]
        val = torch.cat(x[6:9])[order]
        out = grouped(
            q, torch.cat((x[3], k, k)), torch.cat((x[6], val, val)), cq3, ck3, max(v), max(sample_lens), flags3
        )
        return out.split(lengths)

    def shared(*x):
        qva = torch.cat(x[1:3])[vaorder]
        k, val = torch.cat(x[3:6])[order], torch.cat(x[6:9])[order]
        out = grouped(
            torch.cat((x[0], qva)),
            torch.cat((x[3], k)),
            torch.cat((x[6], val)),
            cq2,
            ck2,
            max(v) + max(a),
            max(sample_lens),
            flags2,
        )
        ou, ova = out.split([lengths[0], lengths[1] + lengths[2]])
        return ou, ova[undo_v], ova[undo_a]

    def shared_split(*x):
        qva = torch.cat(x[1:3])[vaorder]
        k, val = torch.cat(x[3:6])[order], torch.cat(x[6:9])[order]
        ou = fa3(x[0], x[3], x[6], cu[0], cu[0], max(u), max(u), causal=True)
        ova = fa3(qva, k, val, cq2[92:] - lengths[0], ck, max(v) + max(a), max(sample_lens))
        return ou, ova[undo_v], ova[undo_a]

    pk, pv = torch.cat(source[3:6])[order], torch.cat(source[6:9])[order]
    leaves = tuple(
        t.detach().requires_grad_()
        for t in (
            torch.cat((source[0], torch.cat(source[1:3])[vaorder])),
            torch.cat((source[3], pk)),
            torch.cat((source[6], pv)),
        )
    )

    def prepacked_shared(q, k, val):
        out = grouped(q, k, val, cq2, ck2, max(v) + max(a), max(sample_lens), flags2)
        ou, ova = out.split([lengths[0], lengths[1] + lengths[2]])
        return ou, ova[undo_v], ova[undo_a]

    functions = {
        "split3": (split, source),
        "grouped3_duplicate_kv": (joint, source),
        "grouped2_shared_full": (shared, source),
        "split2_shared_full": (shared_split, source),
        "prepacked_grouped2": (prepacked_shared, leaves),
    }

compiled = {name: (torch.compile(fn, fullgraph=True, dynamic=False), x) for name, (fn, x) in functions.items()}
validation = {}
reference = None
for name, (fn, x) in compiled.items():
    y = fn(*x)
    grads = torch.autograd.grad(y, x, dos)
    if reference is None:
        reference = tuple(t.detach().clone() for t in y), tuple(t.detach().clone() for t in grads)
    oe = [error(t, r) for t, r in zip(y, reference[0], strict=True)]
    ge = None if name.startswith("prepacked") else [error(t, r) for t, r in zip(grads, reference[1], strict=True)]
    assert max(oe) < 0.03 and (ge is None or max(ge) < 0.03), (name, oe, ge)
    assert all(torch.isfinite(g).all() for g in grads)
    validation[name] = {
        "output_relative_l2": oe,
        "source_gradient_relative_l2": ge,
        "boundary": "independent packed leaves; excludes source gradient restoration"
        if ge is None
        else "same original source leaves",
    }
    del y, grads
    for _ in range(10):
        y = fn(*x)
        torch.autograd.grad(y, x, dos)
    torch.cuda.synchronize()
    print(rank, name, "warmed", flush=True)

rounds = []
rng = random.Random(913 + rank)
for round_id in range(args.rounds):
    samples = {name: [] for name in compiled}
    for _ in range(args.samples):
        execution_order = list(compiled)
        rng.shuffle(execution_order)
        for name in execution_order:
            fn, x = compiled[name]
            start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            torch.cuda.synchronize()
            start.record()
            y = fn(*x)
            mid.record()
            grads = torch.autograd.grad(y, x, dos)
            end.record()
            end.synchronize()
            samples[name].append(
                {
                    "forward_ms": start.elapsed_time(mid),
                    "backward_ms": mid.elapsed_time(end),
                    "fb_ms": start.elapsed_time(end),
                }
            )
            del y, grads
    rounds.append(
        {
            "round": round_id,
            "samples": samples,
            "medians": {n: {k: statistics.median(s[k] for s in ss) for k in ss[0]} for n, ss in samples.items()},
        }
    )
    print(rank, round_id, rounds[-1]["medians"], flush=True)
args.output.mkdir(exist_ok=True, parents=True)
(args.output / f"rank{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "case": args.case,
            "geometry": geo,
            "lengths": lengths,
            "validation": validation,
            "rounds": rounds,
            "torch": torch.__version__,
            "caveat": "microbenchmark on each of 8 GPUs; no optimizer/FSDP; synthetic three-tower is not real training; prepacked is not end-to-end-equivalent implementation",
        },
        indent=2,
    )
)
