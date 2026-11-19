# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare CUDA foreach arithmetic on synthetic gradients with native model shapes.

Reads parameter names and shapes only; no parameter values, gradients, optimizer
state or reference outputs are loaded. Holds one synthetic FP32 tensor per
trainable parameter, so this test requires at least 30 GiB of free GPU memory.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def synthetic(shape, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        tuple(shape), generator=generator, dtype=torch.float32, device=device
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())["parameters"]
    raw_names = json.loads(args.native_order.read_text())["names"]
    names = [name if name.startswith("net.") else "net." + name for name in raw_names]
    expected = {name for name, entry in manifest.items() if entry["optimizer_fields"]}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError(
            "Native parameter order does not cover exactly the trainable manifest shapes"
        )
    shapes = [manifest[name]["global_shape"] for name in names]
    total_bytes = sum(math.prod(shape) * 4 for shape in shapes)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    available, _ = torch.cuda.mem_get_info(device)
    largest = max(math.prod(shape) for shape in shapes) * 4
    if available < total_bytes + 3 * largest + 4 * 1024**3:
        raise RuntimeError(
            "Insufficient free GPU memory for all synthetic native parameter gradients"
        )
    gradients = [
        synthetic(shape, 404000 + index, device) for index, shape in enumerate(shapes)
    ]
    grouped = torch._foreach_norm(gradients, 2)
    single = [torch._foreach_norm([gradient], 2)[0] for gradient in gradients]
    mismatched = [
        name
        for name, a, b in zip(names, grouped, single, strict=True)
        if not torch.equal(a, b)
    ]
    if mismatched:
        raise AssertionError(
            f"Grouped/single norm mismatch for {len(mismatched)} parameters: {mismatched[:8]}"
        )
    total_grouped = torch.linalg.vector_norm(torch.stack(grouped), 2)
    total_single = torch.linalg.vector_norm(torch.stack(single), 2)
    if not torch.equal(total_grouped, total_single):
        raise AssertionError("Native-order total norm differs")
    subset = list(range(0, len(names), 3))
    subset_norms = torch._foreach_norm([gradients[index] for index in subset], 2)
    if any(
        not torch.equal(value, single[index])
        for index, value in zip(subset, subset_norms, strict=True)
    ):
        raise AssertionError(
            "Missing-gradient subset changes a per-tensor foreach norm"
        )
    subset_total_grouped = torch.linalg.vector_norm(torch.stack(subset_norms), 2)
    subset_total_single = torch.linalg.vector_norm(
        torch.stack([single[index] for index in subset]), 2
    )
    if not torch.equal(subset_total_grouped, subset_total_single):
        raise AssertionError("Missing-gradient subset total norm differs")
    grouped_coefficient = torch.clamp(0.3 / (total_grouped + 1e-6), max=1.0)
    single_coefficient = torch.clamp(0.3 / (total_single + 1e-6), max=1.0)
    if not torch.equal(grouped_coefficient, single_coefficient):
        raise AssertionError("Native-order clipping coefficients differ")
    torch._foreach_mul_(gradients, grouped_coefficient)
    for index, (name, actual) in enumerate(zip(names, gradients, strict=True)):
        expected_gradient = synthetic(shapes[index], 404000 + index, device)
        torch._foreach_mul_([expected_gradient], single_coefficient)
        if not torch.equal(actual, expected_gradient):
            raise AssertionError(f"Grouped/single clipping differs for {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(
            {
                "passed": True,
                "scope": "synthetic FP32 CUDA gradients; real native trainable tensor shapes and registration order",
                "parameter_count": len(names),
                "synthetic_gradient_bytes": total_bytes,
                "largest_gradient_bytes": largest,
                "subset_count": len(subset),
                "subset_norm": subset_total_single.item(),
                "norm": total_single.item(),
                "clip_coefficient": single_coefficient.item(),
                "checks": "bitwise grouped/single per-tensor and total norms, subset norms, clipping coefficients and all clipped gradients",
                "uses_reference_values": False,
                "full_model_acceptance": False,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
