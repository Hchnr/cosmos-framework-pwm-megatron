# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare every native-reference output, raw gradient and Adam tensor bitwise."""

import argparse
import json
import math
import struct
from pathlib import Path

import torch


def tensor_exact(left, right):
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if (
        left.dtype != right.dtype
        or left.shape != right.shape
        or left.layout != torch.strided
    ):
        return False
    if right.layout != torch.strided or left.stride() != right.stride():
        return False
    if (left.is_floating_point() or left.is_complex()) and not bool(
        torch.isfinite(left).all()
    ):
        return False
    a = left.detach().contiguous().reshape(-1).view(torch.uint8)
    b = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(a, b)


def nested_exact(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return tensor_exact(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            nested_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        return math.isfinite(left) and struct.pack("d", left) == struct.pack("d", right)
    return left == right


def compare(reference, streamed, expected_parameters, expected_updates):
    baseline = json.loads((reference / "manifest.json").read_text())
    candidate = json.loads((streamed / "manifest.json").read_text())
    keys = (
        "names",
        "input_hashes",
        "runtime_constants_sha256",
        "objective",
        "retain_full_text",
        "clipping_norm_device",
        "native_optimizer",
        "gpu_leaf_reference",
        "preserve_saved_tensor_dense_layout",
        "deterministic_algorithms",
        "original_capacity",
        "attention_backend",
    )
    if not baseline.get("passed") or not candidate.get("passed"):
        raise ValueError("Both reference manifests must report completed native runs")
    if (
        baseline.get("gradient_streaming_candidate", False)
        or candidate.get("gradient_streaming_candidate") is not True
    ):
        raise ValueError("Expected an unstreamed baseline and a streamed candidate")
    if (
        baseline.get("clipping_implementation") != "native_grad_clip"
        or candidate.get("clipping_implementation")
        != "streamed_cuda_norm_and_clip_candidate"
    ):
        raise ValueError("Unexpected native or streamed clipping implementation")
    mismatched_keys = [
        key
        for key in keys
        if key not in baseline
        or key not in candidate
        or baseline[key] != candidate[key]
    ]
    if mismatched_keys:
        raise ValueError(f"Reference provenance differs: {mismatched_keys}")
    names = baseline["names"]
    if (
        len(names) != expected_parameters
        or len(set(names)) != len(names)
        or len(baseline["input_hashes"]) != 2
    ):
        raise ValueError("Unexpected parameter or input coverage")
    original_outputs = torch.load(
        reference / "outputs.pt", map_location="cpu", weights_only=True
    )
    streamed_outputs = torch.load(
        streamed / "outputs.pt", map_location="cpu", weights_only=True
    )
    if (
        len(original_outputs) != expected_updates
        or len(streamed_outputs) != expected_updates
    ):
        raise ValueError("Both references must contain every requested update")
    output_checks = []
    for update, (left, right) in enumerate(
        zip(original_outputs, streamed_outputs, strict=True)
    ):
        if len(left["outputs"]) != 2 or len(right["outputs"]) != 2:
            raise ValueError("Each update must contain both microbatch outputs")
        output_checks.append({"update": update, "passed": nested_exact(left, right)})
    fields = {"gradient", "master", "exp_avg", "exp_avg_sq", "step"}
    mismatches, checked = [], 0
    for update in range(expected_updates):
        for index, name in enumerate(names):
            filename = f"update{update}.parameter{index:04d}.pt"
            left = torch.load(
                reference / filename, map_location="cpu", weights_only=True, mmap=True
            )
            right = torch.load(
                streamed / filename, map_location="cpu", weights_only=True, mmap=True
            )
            if set(left) != fields or set(right) != fields:
                raise ValueError(f"Unexpected parameter state fields in {filename}")
            for field in sorted(fields):
                checked += 1
                if not tensor_exact(left[field], right[field]):
                    mismatches.append({"update": update, "name": name, "field": field})
            del left, right
    return {
        "passed": all(row["passed"] for row in output_checks) and not mismatches,
        "scope": "bitwise native unstreamed versus streamed CUDA reference for all recorded outputs and parameter states",
        "reference": str(reference),
        "streamed": str(streamed),
        "expected_parameters": expected_parameters,
        "expected_updates": expected_updates,
        "microbatches_per_update": 2,
        "compared_parameter_files": expected_parameters * expected_updates,
        "compared_tensor_fields": checked,
        "output_checks": output_checks,
        "mismatches": mismatches,
        "provenance_fields_equal": list(keys),
        "comparison": "exact tensor bytes, dtype, shape and stride; finite tensors; exact norm and output values",
        "full_capacity_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--streamed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parameters", type=int, default=811)
    parser.add_argument("--expected-updates", type=int, default=3)
    args = parser.parse_args()
    if args.reference.resolve() == args.streamed.resolve() or args.output.exists():
        raise ValueError("Reference, candidate and new report must be distinct")
    result = compare(
        args.reference, args.streamed, args.expected_parameters, args.expected_updates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    if not result["passed"]:
        raise SystemExit(
            "Native streamed reference differs from the unstreamed reference"
        )


if __name__ == "__main__":
    main()
