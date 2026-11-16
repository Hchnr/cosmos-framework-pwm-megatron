# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compare sampled real FSDP shards after the first identical GA2 optimizer step."""

import argparse
import json
import math
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--require-replay", action="store_true")
args = parser.parse_args()
if args.require_replay:
    for rank in range(8):
        identities = []
        for run in (args.reference, args.candidate):
            records = [json.loads(line) for line in (run / f"rank{rank}.jsonl").read_text().splitlines()]
            noise = [r["hashes"] for r in records if r["event"] == "noising_audit"]
            assert len(noise) == 2, (run, rank, "missing complete GA2 noising audit")
            first = next(r for r in records if r["event"] == "step")
            identities.append((noise, first["input_fingerprints"], first["rng_fingerprints"]))
            finite = [r for r in records if r["event"] == "pre_clip_finite_audit"]
            assert len(finite) == 2 and all(not r["nonfinite_parameters"] for r in finite), (run, rank)
        assert identities[0] == identities[1], (rank, "first-update replay mismatch")
statistics = {}
for rank in range(8):
    ref = torch.load(args.reference / f"update_rank{rank}.pt", map_location="cpu", weights_only=True)
    got = torch.load(args.candidate / f"update_rank{rank}.pt", map_location="cpu", weights_only=True)
    assert ref.keys() == got.keys(), (rank, ref.keys() - got.keys(), got.keys() - ref.keys())
    for name in ref:
        r, g = ref[name], got[name]
        assert torch.equal(r["before"], g["before"]), (rank, name, "initial parameter differs")
        for kind in ("grad_after_clip", "update"):
            a = r[kind].float() if kind != "update" else r["after"].float() - r["before"].float()
            b = g[kind].float() if kind != "update" else g["after"].float() - g["before"].float()
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), (rank, name, kind)
            key = kind + ":" + name
            row = statistics.setdefault(
                key, {"reference_sq": 0.0, "candidate_sq": 0.0, "diff_sq": 0.0, "dot": 0.0, "max_abs": 0.0}
            )
            row["reference_sq"] += a.double().square().sum().item()
            row["candidate_sq"] += b.double().square().sum().item()
            row["diff_sq"] += (b.double() - a).square().sum().item()
            row["dot"] += (a.double() * b).sum().item()
            row["max_abs"] = max(row["max_abs"], (a - b).abs().max().item() if a.numel() else 0.0)
    del ref, got


def finalize(row):
    return {
        **row,
        "relative_l2": math.sqrt(row["diff_sq"] / max(row["reference_sq"], 1e-30)),
        "cosine": row["dot"] / max(math.sqrt(row["reference_sq"] * row["candidate_sq"]), 1e-30),
    }


summary = {}
for kind in ("grad_after_clip", "update"):
    rows = [r for key, r in statistics.items() if key.startswith(kind + ":")]
    total = {key: sum(r[key] for r in rows) for key in ("reference_sq", "candidate_sq", "diff_sq", "dot")}
    total["max_abs"] = max(r["max_abs"] for r in rows)
    summary[kind] = finalize(total)
result = {
    "reference": str(args.reference),
    "candidate": str(args.candidate),
    "initial_weights_bitwise_equal": True,
    "full_first_step_replay_verified": args.require_replay,
    "scope": "sampled layers 0/17/35 and input/output heads, all eight local FSDP shards",
    "summary": summary,
    "parameters": {key: finalize(row) for key, row in statistics.items()},
}
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(summary, indent=2))
