# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare captured frozen UND Q path and optionally replay its FP32 GEMM.

The weight operands are exact runtime tensors captured while the modules ran,
not reconstructed checkpoint masters. This is a single-operator diagnostic,
not a complete training or model-accuracy test.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

from tools.analyze_tp_pp_attention_isolation import analyze, load_records
from tools.run_tp_pp_und_first_attention_capture import UND_MODULES, tensor_digest


def load_weight(path, rank, size):
    data = torch.load(path, map_location="cpu", weights_only=True)
    weight, bias = data["weight"], data["bias"]
    if (
        data["rank"] != rank
        or data["tp_rank"] != rank
        or data["tp_size"] != size
        or not str(data["weight_device"]).startswith("cuda")
        or weight.dtype != torch.float32
        or tensor_digest(weight) != data["weight_sha256"]
        or (bias is None) != (data["bias_sha256"] is None)
        or (bias is not None and tensor_digest(bias) != data["bias_sha256"])
    ):
        raise ValueError(f"Invalid captured runtime parameter: {path}")
    return data


def verify_runtime_parameters(capture):
    results, weights = {}, {}
    for name in UND_MODULES:
        tag = name.replace(".", "_")
        native = load_weight(capture / "weights/native" / f"{tag}.rank0.pt", 0, 1)
        shards = [load_weight(capture / "weights/tp" / f"{tag}.rank{rank}.pt", rank, 4) for rank in range(4)]
        sharded = name == "self_attn.q_proj"
        weight_equal = []
        bias_equal = []
        for rank, local in enumerate(shards):
            expected = native["weight"].chunk(4, dim=0)[rank] if sharded else native["weight"]
            weight_equal.append(torch.equal(expected, local["weight"]))
            expected_bias = native["bias"]
            if sharded and expected_bias is not None:
                expected_bias = expected_bias.chunk(4, dim=0)[rank]
            actual_bias = local["bias"]
            bias_equal.append(
                (expected_bias is None and actual_bias is None)
                or (expected_bias is not None and actual_bias is not None and torch.equal(expected_bias, actual_bias))
            )
        results[name] = {
            "weight_equal_by_rank": weight_equal,
            "bias_equal_by_rank": bias_equal,
            "native_weight_sha256": native["weight_sha256"],
            "rank_weight_sha256": [part["weight_sha256"] for part in shards],
            "runtime_weight_devices": [native["weight_device"], *[part["weight_device"] for part in shards]],
            "eps_equal_by_rank": [native["epsilon"] == part["epsilon"] for part in shards],
        }
        weights[name] = native, shards
    return results, weights


def difference(expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError("Replay tensors have incompatible shape or dtype")
    expected, actual = expected.detach().cpu(), actual.detach().cpu()
    delta = expected.double() - actual.double()
    return {
        "exact": torch.equal(expected, actual),
        "maximum_absolute_error": delta.abs().max().item() if delta.numel() else 0.0,
        "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-30)).item(),
    }


def restore(record, device, source=None):
    if record["scope"] != {"kind": "full"} or list(record["value"].shape) != record["shape"]:
        raise ValueError("GPU replay requires the complete captured operand")
    value = torch.empty_strided(
        tuple(record["shape"]), tuple(record["stride"]), dtype=record["value"].dtype, device=device
    )
    value.copy_(record["value"].to(device) if source is None else source)
    return value


def replay_q(capture, weights, device):
    from cosmos_framework.model.generator.megatron_tp_pp.compile import NonePreservingLinear

    native = load_records(capture / "native")
    local = load_records(capture / "tp")
    key_in, key_out = (0, "self_attn_q_proj", "input"), (0, "self_attn_q_proj", "output")
    xref, yref = native[key_in][0], native[key_out][0]
    xparts, yparts = sorted(local[key_in], key=lambda x: x["rank"]), sorted(local[key_out], key=lambda x: x["rank"])
    if (
        len(xparts) != 4
        or len(yparts) != 4
        or any(not torch.equal(xref["value"], part["value"]) or xref["stride"] != part["stride"] for part in xparts)
    ):
        raise ValueError("Q replay requires four TP inputs bitwise identical to native")
    native_weight, shard_weights = weights["self_attn.q_proj"]
    if any(w["bias"] is not None for w in [native_weight, *shard_weights]):
        raise ValueError("Q replay currently requires bias-free projection")
    if any(not torch.equal(native_weight["weight"].chunk(4, 0)[i], shard_weights[i]["weight"]) for i in range(4)):
        raise ValueError("Captured runtime Q weights differ between native and TP")
    if any(tuple(item["weight"].stride()) != tuple(item["weight_stride"]) for item in [native_weight, *shard_weights]):
        raise ValueError("Captured Q weight layout cannot be reproduced")
    x = restore(xref, device)
    w = native_weight["weight"].to(device)
    local_out = []
    for part in shard_weights:
        local_out.append(
            NonePreservingLinear.apply(
                x, part["weight"].to(device), None, False, False, False, None, None, None, 1, None
            )
        )
    full_out = F.linear(x, w)
    return {
        "same_input": True,
        "captured_native_reproduced": difference(yref["value"], full_out),
        "captured_tp_reproduced": [difference(yparts[i]["value"], local_out[i]) for i in range(4)],
        "native_full_vs_tp_sharded": difference(full_out, torch.cat(local_out, dim=-1)),
        "runtime_weight_scope": "captured directly during original forward on each rank",
        "full_model_acceptance": False,
    }


def replay_headwise(capture, weights, device):
    from cosmos_framework.model.generator.mot.small_action_tower import rotate
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRMSNorm

    native, local = load_records(capture / "native"), load_records(capture / "tp")

    def records(tag, kind):
        key = (0, tag, kind)
        expected, candidates = native[key], sorted(local[key], key=lambda row: row["rank"])
        if len(expected) != 1 or len(candidates) != 4 or [row["rank"] for row in candidates] != list(range(4)):
            raise ValueError(f"Incomplete captured records for {key}")
        return expected[0], candidates

    ref_norm_input, tp_norm_input = records("self_attn_q_norm", "input")
    ref_norm_output, tp_norm_output = records("self_attn_q_norm", "output")
    native_weight, shard_weights = weights["self_attn.q_norm"]
    eps = native_weight["epsilon"]
    if eps is None or any(item["epsilon"] != eps for item in shard_weights):
        raise ValueError("Captured Q norm epsilon differs or is unavailable")
    if any(not torch.equal(item["weight"], native_weight["weight"]) for item in shard_weights):
        raise ValueError("Captured Q norm weights differ")
    norm = Qwen3VLTextRMSNorm(native_weight["weight"].numel(), eps=float(eps)).to(device)
    norm.weight.data.copy_(native_weight["weight"].to(device))
    full_input = restore(ref_norm_input, device)
    if full_input.ndim != 3 or full_input.shape[1] % 4:
        raise ValueError("Expected equally head-sharded 3-D Q norm inputs")
    width = full_input.shape[1] // 4
    full_output = norm(full_input)
    captured_outputs = [norm(restore(row, device)) for row in tp_norm_input]
    common_outputs = [
        norm(restore(tp_norm_input[rank], device, full_input.narrow(1, rank * width, width))) for rank in range(4)
    ]
    norm_report = {
        "captured_native_reproduced": difference(ref_norm_output["value"], full_output),
        "captured_tp_reproduced": [difference(tp_norm_output[i]["value"], captured_outputs[i]) for i in range(4)],
        "native_full_vs_tp_sharded_common_input": difference(full_output, torch.cat(common_outputs, dim=1)),
    }

    ref_input, tp_input = records("und_rotate0", "input")
    ref_cos, tp_cos = records("und_rotate0", "cos")
    ref_sin, tp_sin = records("und_rotate0", "sin")
    ref_output, tp_output = records("und_rotate0", "output")
    if any(not torch.equal(ref_cos["value"], row["value"]) for row in tp_cos) or any(
        not torch.equal(ref_sin["value"], row["value"]) for row in tp_sin
    ):
        raise ValueError("Captured UND rotary position factors differ")
    x, cos, sin = restore(ref_input, device), restore(ref_cos, device), restore(ref_sin, device)
    if x.ndim != 3 or x.shape[1] % 4:
        raise ValueError("Expected equally head-sharded 3-D rotary inputs")
    width = x.shape[1] // 4
    full_output = rotate(x, (cos, sin))
    captured_outputs = [
        rotate(restore(tp_input[i], device), (restore(tp_cos[i], device), restore(tp_sin[i], device))) for i in range(4)
    ]
    common_outputs = [rotate(restore(tp_input[i], device, x.narrow(1, i * width, width)), (cos, sin)) for i in range(4)]
    rotate_report = {
        "position_factors_equal": True,
        "captured_native_reproduced": difference(ref_output["value"], full_output),
        "captured_tp_reproduced": [difference(tp_output[i]["value"], captured_outputs[i]) for i in range(4)],
        "native_full_vs_tp_sharded_common_input": difference(full_output, torch.cat(common_outputs, dim=1)),
    }
    return {"q_norm": norm_report, "rope": rotate_report, "full_model_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-replay", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results, weights = verify_runtime_parameters(args.capture_dir)
    trace = analyze(args.capture_dir / "native", args.capture_dir / "tp")
    output = {
        "scope": "captured frozen UND Q path diagnostic; not full-model acceptance",
        "runtime_parameters": results,
        "trace": trace,
        "gpu_replay": None,
    }
    if args.gpu_replay:
        if not torch.cuda.is_available() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("GPU replay requires CUDA and deterministic CUBLAS_WORKSPACE_CONFIG")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        with torch.no_grad():
            output["gpu_replay"] = {
                "projection": replay_q(args.capture_dir, weights, torch.device("cuda:0")),
                "headwise": replay_headwise(args.capture_dir, weights, torch.device("cuda:0")),
            }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
