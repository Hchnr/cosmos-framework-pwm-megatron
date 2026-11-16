# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Replay captured, post-RoPE QKV and actual dO; compare all source gradients."""

import argparse
import json
import os
from pathlib import Path

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.attention import two_way_attention
from cosmos_framework.model.generator.mot.joint_attention import joint_attention, prepare_joint_attention

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--backend", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--write-reference", action="store_true")
args = parser.parse_args()
rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
root = args.run / "activations" / f"rank{rank}"
geometry = json.loads((args.run / f"geometry_rank{rank}.json").read_text())
report = {}
for layer in (0, 17, 35):
    uv = [
        torch.load(root / f"layer{layer}_{stream}_qkv.pt", map_location="cpu", weights_only=True) for stream in (0, 1)
    ]
    sources = [uv[stream][kind].cuda().requires_grad_(True) for kind in range(3) for stream in (0, 1)]
    nu, ng = sources[0].shape[0], sources[1].shape[0]
    uids, gids, start = [], [], 0
    for size, mode in zip(geometry["split_lens"], geometry["attn_modes"]):
        (uids if mode == "causal" else gids).extend(range(start, start + size))
        start += size
    pack = sequence_pack_from_packed_sequence(
        torch.zeros(start, 1, device="cuda"),
        geometry["attn_modes"],
        geometry["split_lens"],
        geometry["sample_lens"],
        torch.tensor(uids, device="cuda"),
        torch.tensor(gids, device="cuda"),
        fixed_causal_len=nu,
        fixed_full_len=ng,
    )
    qp, kp, vp = [from_und_gen_splits(*sources[i : i + 2], pack) for i in (0, 2, 4)]
    output = (
        two_way_attention(qp, kp, vp)
        if args.backend == "split_default"
        else joint_attention(qp, kp, vp, prepare_joint_attention(pack, args.backend))
    )
    ys = [get_und_seq(output), get_gen_seq(output)]
    dos = []
    for stream, y in enumerate(ys):
        path = root / f"layer{layer}_{stream}_dout.pt"
        dos.append(
            torch.load(path, map_location="cpu", weights_only=True)[0].cuda() if path.exists() else torch.zeros_like(y)
        )
    sum((y.float() * do.float()).sum() for y, do in zip(ys, dos)).backward()
    values = {
        "out_u": ys[0][: len(uids)],
        "out_g": ys[1][: len(gids)],
        **{f"grad_{i}": source.grad for i, source in enumerate(sources)},
    }
    reference_path = root / f"layer{layer}_replay_reference.pt"
    if args.write_reference:
        assert not reference_path.exists(), reference_path
        torch.save({k: v.detach().cpu() for k, v in values.items()}, reference_path)
        refs = {
            f"out_{label}": torch.load(root / f"layer{layer}_{stream}_out.pt", map_location="cpu", weights_only=True)[
                0
            ][:length]
            for stream, label, length in ((0, "u", len(uids)), (1, "g", len(gids)))
        }
        for stream in (0, 1):
            path = root / f"layer{layer}_{stream}_dq.pt"
            if path.exists():
                refs[f"grad_{stream}"] = torch.load(path, map_location="cpu", weights_only=True)[0]
    else:
        refs = torch.load(reference_path, map_location="cpu", weights_only=True)
    errors = {}
    for key, reference in refs.items():
        actual = values[key].float()
        reference = reference.to(actual.device).float()
        diff = actual - reference
        denom = reference.norm().clamp_min(1e-12)
        rel = (diff.norm() / denom).item()
        cosine = ((actual * reference).sum() / (actual.norm() * denom).clamp_min(1e-30)).item()
        errors[key] = {"relative_l2": rel, "max_abs": diff.abs().max().item(), "cosine": cosine}
        assert torch.isfinite(actual).all() and rel < 0.03, (layer, key, errors[key])
    report[layer] = errors
    del uv, sources, qp, kp, vp, output, ys, dos, values, refs, actual, reference, diff, pack
    torch.cuda.empty_cache()
args.output.mkdir(parents=True, exist_ok=True)
(args.output / f"rank{rank}.json").write_text(json.dumps({"backend": args.backend, "errors": report}, indent=2))
