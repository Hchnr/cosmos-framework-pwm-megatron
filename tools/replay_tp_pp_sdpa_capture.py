# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replay captured full-head vs TP-head SDPA at identical inputs and cotangent.

Use two complete forward captures from the 36-layer K/V intervention. This is
an operator diagnostic only, not numerical acceptance or a production fix.
Captured stride is reconstructed; original storage offset was not recorded.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def load_record(directory, layer, call, kind, rank, tp_size):
    path = directory / f"layer{layer:02d}.attention_call{call}.{kind}.rank{rank}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not (
        record["layer"] == layer
        and record["tag"] == f"attention_call{call}"
        and record["kind"] == kind
        and record["rank"] == rank
        and record["tp_rank"] == rank
        and record["tp_size"] == tp_size
        and record["scope"] == {"kind": "full"}
        and record["dtype"] == str(record["value"].dtype)
        and list(record["value"].shape) == record["shape"]
    ):
        raise ValueError(f"Invalid or partial capture {path}")
    digest = hashlib.sha256(record["value"].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    if digest != record["stored_sha256"]:
        raise ValueError(f"Capture hash mismatch: {path}")
    return record


def restore(record, device, *, requires_grad=False):
    shape, stride = record["shape"], record["stride"]
    value = record["value"]
    if any(not isinstance(n, int) or n < 0 for n in (*shape, *stride)) or len(shape) != len(stride):
        raise ValueError("Invalid captured shape or stride")
    result = torch.empty_strided(shape, stride, dtype=value.dtype, device=device)
    result.copy_(value.to(device))
    return result.detach().requires_grad_(requires_grad)


def compare(expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError("Shape or dtype differs in diagnostic comparison")
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("Nonfinite diagnostic tensor")
    delta = expected.double() - actual.double()
    return {
        "exact": bool(torch.equal(expected, actual)),
        "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-30)).item(),
        "maximum_absolute_error": delta.abs().max().item() if delta.numel() else 0.0,
    }


def attend(q, k, v, mask):
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


def run_case(native, tp, *, layer, call, device):
    records = {
        key: load_record(native, layer, call, key, 0, 1)
        for key in ("query", "key", "value", "mask", "output", "output_dgrad")
    }
    ranks = [
        {
            key: load_record(tp, layer, call, key, rank, 4)
            for key in ("query", "key", "value", "mask", "output", "output_dgrad")
        }
        for rank in range(4)
    ]
    if any(record["value"].dtype != torch.float32 for key, record in records.items() if key != "mask"):
        raise ValueError("Captured activations and cotangents must be FP32")
    if records["mask"]["value"].dtype != torch.bool:
        raise ValueError("Expected a bool attention mask")
    q, k, v = [restore(records[key], device, requires_grad=True) for key in ("query", "key", "value")]
    mask, dy = [restore(records[key], device) for key in ("mask", "output_dgrad")]
    output = attend(q, k, v, mask)
    full_grads = torch.autograd.grad(output, (q, k, v), grad_outputs=dy)
    native_expected = restore(records["output"], device)
    native_replay = compare(native_expected, output)
    result = {
        "forward": records["query"]["forward"],
        "layer": layer,
        "attention_call": call,
        "native_record_strides": {key: record["stride"] for key, record in records.items()},
        "native_forward_matches_capture": native_replay,
        "rank_results": [],
    }
    for rank, local in enumerate(ranks):
        native_views = [
            tensor.narrow(1, rank * local[key]["shape"][1], local[key]["shape"][1])
            for tensor, key in zip((q, k, v), ("query", "key", "value"), strict=True)
        ]
        local_inputs = [restore(local[key], device, requires_grad=True) for key in ("query", "key", "value")]
        qkv_equal = [
            bool(torch.equal(reference, actual)) for reference, actual in zip(native_views, local_inputs, strict=True)
        ]
        local_mask = restore(local["mask"], device)
        mask_equal = bool(torch.equal(mask, local_mask))
        if not all(qkv_equal) or not mask_equal:
            raise ValueError(f"Rank {rank} input or mask does not match native shard")
        expected_width = local["output"]["shape"][1]
        local_dy = torch.empty_strided(
            local["output_dgrad"]["shape"],
            local["output_dgrad"]["stride"],
            dtype=dy.dtype,
            device=device,
        )
        local_dy.copy_(dy.narrow(1, rank * expected_width, expected_width))
        local_output = attend(*local_inputs, local_mask)
        local_grads = torch.autograd.grad(local_output, local_inputs, grad_outputs=local_dy)
        expected_output = output.narrow(1, rank * expected_width, expected_width)
        recorded_output = restore(local["output"], device)
        actual_dy = restore(local["output_dgrad"], device)
        result["rank_results"].append(
            {
                "rank": rank,
                "qkv_input_exact": qkv_equal,
                "mask_exact": mask_equal,
                "local_record_strides": {key: record["stride"] for key, record in local.items()},
                "local_forward_matches_capture": compare(recorded_output, local_output),
                "full_vs_shard_forward": compare(expected_output, local_output),
                "full_vs_shard_vjp": {
                    key: compare(
                        gradient.narrow(1, rank * actual.shape[1], actual.shape[1]),
                        actual,
                    )
                    for key, gradient, actual in zip(("dq", "dk", "dv"), full_grads, local_grads, strict=True)
                },
                "captured_tp_dy_vs_native_dy": compare(local_dy, actual_dy),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-base", type=Path, required=True)
    parser.add_argument("--tp-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cases = [
        run_case(
            args.native_base / f"forward{forward:03d}" / "native",
            args.tp_base / f"forward{forward:03d}" / "tp",
            layer=0,
            call=call,
            device=args.device,
        )
        for forward in (0, 1)
        for call in (1, 2)
    ]
    reproduced = all(
        case["native_forward_matches_capture"]["exact"]
        and all(rank["local_forward_matches_capture"]["exact"] for rank in case["rank_results"])
        for case in cases
    )
    report = {
        "scope": "same captured q/k/v/mask and common native cotangent, math SDPA; not full-model acceptance",
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32 if args.device == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "captured_forward_reproduced": reproduced,
        "storage_offset_independently_captured": False,
        "full_model_acceptance": False,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not reproduced:
        raise SystemExit("Captured native/TP output was not exactly reproduced; no causal conclusion")


if __name__ == "__main__":
    main()
