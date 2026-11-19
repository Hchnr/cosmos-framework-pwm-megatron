# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Controlled TP4 FP32 native-width vs sharded row GEMM replay.

Uses actual captured native and TP inputs independently for capture checks,
then native input and native dY on both methods for controlled comparisons.
Canonical masters are verified, but runtime parameter bytes were not captured.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.distributed as dist
from torch.nn import functional as F

from tools.tp_pp_full_width_row_diagnostic import full_row

TAGS = {
    "video_o": ("self_attn_o_proj_moe_gen", "self_attn.o_proj_moe_gen.weight"),
    "action_o": ("action_block_moe_gen_action_o_proj", "action_block_moe_gen_action.o_proj.weight"),
    "video_down": ("mlp_moe_gen_down_proj", "mlp_moe_gen.down_proj.weight"),
}


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def trace(root, tag, kind, rank, tp):
    path = root / f"layer00.{tag}.{kind}.rank{rank}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False)
    tensor = record["value"]
    if (
        record["forward"] != 0
        or record["layer"] != 0
        or record["tag"] != tag
        or record["kind"] != kind
        or record["rank"] != rank
        or record["tp_size"] != tp
        or record["scope"] != {"kind": "full"}
        or tensor.dtype != torch.float32
        or list(tensor.shape) != record["shape"]
        or sha_tensor(tensor) != record["stored_sha256"]
    ):
        raise ValueError(f"Invalid capture {path}")
    return tensor


def master(bundle, name, tp):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["format"] != "cosmos_tp_pp_canonical" or not manifest["complete"] or manifest["tp"] != tp:
        raise ValueError("Bad bundle manifest")
    entry = manifest["parameters"][name]
    if entry["axis"] != 1 or not entry["optimizer_fields"] or not (bundle / ".save_complete").is_file():
        raise ValueError("Expected complete, trainable, row-sharded FP32 master")
    pieces, offset = [], 0
    for rank, shard in enumerate(sorted(entry["shards"], key=lambda value: value["tensor_rank"])):
        target = (bundle / shard["path"]).resolve()
        if not target.is_relative_to(bundle.resolve()) or shard["tensor_rank"] != rank or shard["offset"] != offset:
            raise ValueError("Invalid shard identity")
        item = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
        weight = item["state"]["master_param"]
        if (
            item["canonical_name"] != name
            or weight.dtype != torch.float32
            or sha_tensor(weight) != shard["sha256"]["master_param"]
        ):
            raise ValueError("Master does not match its manifest")
        pieces.append(weight)
        offset += weight.shape[1]
    if len(pieces) != tp:
        raise ValueError("Missing weight shards")
    value = torch.cat(pieces, dim=1)
    if list(value.shape) != entry["global_shape"]:
        raise ValueError("Unexpected full weight shape")
    return value


def diff(left, right):
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Nonfinite or mismatched result")
    a, b = left.double(), right.double()
    return {
        "exact": torch.equal(left, right),
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / a.norm().clamp_min(1e-30)).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    for arg in ("native-trace", "tp-trace", "native-bundle", "tp-bundle", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--projection", choices=tuple(TAGS), required=True)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4 or args.output.exists() or os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("TP4, fresh output and deterministic CUBLAS workspace required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tag, suffix = TAGS[args.projection]
    name = "net.language_model.model.layers.0." + suffix
    native_weight = master(args.native_bundle, name, 2)
    tp_weight = master(args.tp_bundle, name, 4)
    if not torch.equal(native_weight, tp_weight):
        raise ValueError("Canonical TP2/TP4 master mismatch")
    x = trace(args.native_trace, tag, "input", 0, 1).cuda()
    native_out = trace(args.native_trace, tag, "output", 0, 1).cuda()
    dy = trace(args.native_trace, tag, "output_dgrad", 0, 1).cuda()
    local_x = trace(args.tp_trace, tag, "input", rank, 4).cuda()
    captured_tp = trace(args.tp_trace, tag, "output", rank, 4).cuda()
    weight = native_weight.cuda()
    width = weight.shape[1] // world
    w = weight[:, rank * width : (rank + 1) * width].contiguous()
    with torch.no_grad():
        native = F.linear(x, weight)
        captured_local = F.linear(local_x, w)
        dist.all_reduce(captured_local)
        common_partial = F.linear(x[:, rank * width : (rank + 1) * width].contiguous(), w)
        dist.all_reduce(common_partial)
    local = x[:, rank * width : (rank + 1) * width].detach().clone().requires_grad_()
    candidate_w = w.detach().clone().requires_grad_()
    candidate = full_row(local, candidate_w)
    candidate.backward(dy)
    expected_dx = torch.mm(dy, weight)[:, rank * width : (rank + 1) * width].contiguous()
    expected_dw = torch.mm(dy.t(), x)[:, rank * width : (rank + 1) * width].contiguous()
    report = {
        "rank": rank,
        "projection": args.projection,
        "scope": "controlled layer0 FP32 row GEMM, not training acceptance",
        "canonical_only_runtime_weight_not_captured": True,
        "full_weight_sha256": sha_tensor(native_weight),
        "native_input_vs_tp_local": diff(x[:, rank * width : (rank + 1) * width], local_x),
        "native_capture_reproduced": diff(native_out, native),
        "tp_capture_reproduced_with_own_input": diff(captured_tp, captured_local),
        "native_full_vs_shard_sum_common_input": diff(native, common_partial),
        "candidate_full_forward": diff(native, candidate),
        "candidate_dx": diff(expected_dx, local.grad),
        "candidate_dw": diff(expected_dw, candidate_w.grad),
        "full_model_acceptance": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.projection}.rank{rank}.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
