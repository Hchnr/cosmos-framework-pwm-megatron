# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Locate FP32 layer divergence without replacing full mathematical gates."""

import argparse
import json
from pathlib import Path

import torch


def difference(expected, actual):
    if expected["shape"] != actual["shape"] or expected["sample"].dtype != actual["sample"].dtype:
        raise ValueError("Trace shapes or compute dtypes differ")
    if expected["sample"].dtype != torch.float32:
        raise ValueError("This analysis requires actual FP32 compute tensors")
    left, right = expected["sample"].double(), actual["sample"].double()
    delta = left - right
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("Nonfinite sampled trace")
    return {
        "sample_relative_l2": (delta.norm() / left.norm().clamp_min(1e-30)).item(),
        "sample_maximum_absolute_error": delta.abs().max().item(),
        "full_norm_relative_difference": abs(expected["norm"] - actual["norm"]) / max(expected["norm"], 1e-30),
    }


def analyze(reference, candidate):
    expected = {path.name for path in reference.glob("*.pt")}
    actual = {path.name for path in candidate.glob("*.pt")}
    if expected != actual:
        raise ValueError(f"Trace coverage differs: missing={len(expected - actual)}, extra={len(actual - expected)}")
    outputs = [name for name in expected if name.endswith(".outputs.pt")]
    if len(outputs) != 36 * 6:
        raise ValueError("Require all 36 layers, two batches and three updates")
    rows = []
    for name in sorted(expected):
        left = torch.load(reference / name, map_location="cpu", weights_only=False)
        right = torch.load(candidate / name, map_location="cpu", weights_only=False)
        if name.endswith(".outputs.pt"):
            if left.keys() != right.keys():
                raise ValueError("Trace streams differ")
            for field in left:
                rows.append({"file": name, "stream": field, **difference(left[field], right[field])})
        else:
            rows.append({"file": name, "stream": name.split(".")[2], **difference(left, right)})
    return {
        "scope": "FP32 fixed head/tail row samples plus full stream norms; diagnostic only, not full tensor/state acceptance",
        "coverage_complete": True,
        "mathematical_acceptance": False,
        "reference": str(reference),
        "candidate": str(candidate),
        "first_different_output": next(
            (row for row in rows if row["file"].endswith(".outputs.pt") and row["sample_maximum_absolute_error"] > 0),
            None,
        ),
        "maximum_sample_relative_l2": max(row["sample_relative_l2"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.reference, args.candidate)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
