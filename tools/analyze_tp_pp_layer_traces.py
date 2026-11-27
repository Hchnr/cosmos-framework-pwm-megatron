# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Locate FP32 layer divergence without replacing full mathematical gates."""

import argparse
import json
import math
from pathlib import Path

import torch


def difference(expected, actual):
    if (
        expected["shape"] != actual["shape"]
        or expected["sample"].shape != actual["sample"].shape
        or expected["sample"].dtype != actual["sample"].dtype
    ):
        raise ValueError("Trace shapes or compute dtypes differ")
    if expected["sample"].dtype != torch.float32:
        raise ValueError("This analysis requires actual FP32 compute tensors")
    left, right = expected["sample"].double(), actual["sample"].double()
    delta = left - right
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("Nonfinite sampled trace")
    left_norm, right_norm = left.norm().item(), right.norm().item()
    reference_norm, candidate_norm = expected["norm"], actual["norm"]
    if not all(math.isfinite(norm) and norm >= 0 for norm in (reference_norm, candidate_norm)):
        raise ValueError("Invalid full stream norm")
    return {
        "sample_relative_l2": delta.norm().item() / left_norm if left_norm else None,
        "sample_maximum_absolute_error": delta.abs().max().item() if delta.numel() else 0.0,
        "sample_cosine_similarity": (
            max(-1.0, min(1.0, (left * right).sum().item() / (left_norm * right_norm)))
            if left_norm and right_norm
            else None
        ),
        "full_norm_absolute_difference": abs(reference_norm - candidate_norm),
        "full_norm_relative_difference": abs(reference_norm - candidate_norm) / reference_norm
        if reference_norm
        else None,
    }


def candidate_trace_paths(directory, dp_size):
    if dp_size == 1:
        return {path.name: path for path in directory.glob("*.pt")}
    result = {}
    for rank in range(dp_size):
        for path in (directory / f"dp{rank:02d}").glob("*.pt"):
            capture = int(path.name.split(".", 1)[0].removeprefix("forward"))
            if capture % dp_size != rank or path.name in result:
                raise ValueError(f"Unexpected or duplicate logical trace capture: {path}")
            result[path.name] = path
    return result


def analyze(reference, candidate, *, candidate_dp_size=1):
    if candidate_dp_size not in (1, 2):
        raise ValueError("The two-slot diagnostic supports DP1 or DP2 traces")
    expected = {path.name for path in reference.glob("*.pt")}
    actual = candidate_trace_paths(candidate, candidate_dp_size)
    if expected != actual.keys():
        raise ValueError(
            f"Trace coverage differs: missing={len(expected - actual.keys())}, extra={len(actual.keys() - expected)}"
        )
    outputs = {name for name in expected if name.endswith(".outputs.pt")}
    required_outputs = {
        f"forward{capture:03d}.layer{layer:02d}.outputs.pt" for capture in range(6) for layer in range(36)
    }
    if outputs != required_outputs:
        raise ValueError("Require all 36 layers, two batches and three updates")
    rows = []
    for name in sorted(expected):
        left = torch.load(reference / name, map_location="cpu", weights_only=False)
        right = torch.load(actual[name], map_location="cpu", weights_only=False)
        if name.endswith(".outputs.pt"):
            if left.keys() != right.keys():
                raise ValueError("Trace streams differ")
            for field in left:
                rows.append({"file": name, "stream": field, **difference(left[field], right[field])})
        else:
            # PP schedules divide by local microbatches. The DP average runs
            # after these cotangents are captured, so apply it for comparison.
            effective = {
                **right,
                "sample": right["sample"] / candidate_dp_size,
                "norm": right["norm"] / candidate_dp_size,
            }
            rows.append({"file": name, "stream": name.split(".")[2], **difference(left, effective)})
    first_by_boundary = {"forward": {}, "backward": {}}
    relative_errors = []
    for row in rows:
        capture, layer, _ = row["file"].split(".", 2)
        direction = "forward" if row["file"] in outputs else "backward"
        row.update(
            capture=int(capture.removeprefix("forward")), layer=int(layer.removeprefix("layer")), direction=direction
        )
        if row["sample_relative_l2"] is not None:
            relative_errors.append(row["sample_relative_l2"])
        if row["sample_maximum_absolute_error"] == 0 and row["full_norm_absolute_difference"] == 0:
            continue
        key = capture + "." + row["stream"]
        # Captures are independent. Within a stream the backward traversal starts at the last layer.
        first = first_by_boundary[direction]
        if direction == "forward":
            first.setdefault(key, row)
        else:
            first[key] = row
    return {
        "scope": "FP32 sampled layer outputs and available output cotangents, plus full stream norms",
        # Matching the files present on both sides does not establish that
        # every expected cotangent was captured on both sides.
        "coverage_complete": False,
        "forward_output_coverage_complete": True,
        "backward_coverage_complete": None,
        "backward_trace_file_count": len(expected - outputs),
        "mathematical_acceptance": False,
        "reference": str(reference),
        "candidate": str(candidate),
        "candidate_data_parallel_size": candidate_dp_size,
        "candidate_pre_dp_cotangent_scale": 1 / candidate_dp_size,
        "first_different_output": next(
            (row for row in rows if row["file"].endswith(".outputs.pt") and row["sample_maximum_absolute_error"] > 0),
            None,
        ),
        "first_observed_difference_by_capture_and_stream": first_by_boundary,
        "maximum_sample_relative_l2": max(relative_errors, default=None),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-dp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.reference, args.candidate, candidate_dp_size=args.candidate_dp_size)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
