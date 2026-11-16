#!/usr/bin/env python3
"""Estimate useful model FLOPs from observed sample splits for both study towers."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from cosmos_framework.model.generator.mot.small_action_tower import action_shape
from cosmos_framework.tools.flops import compute_omni_mot_flops_per_batch, get_omni_mot_model_descriptor

BASE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
)
PEAK_TFLOPS = 989.0
WORLD_SIZE = 4


def descriptor():
    config = json.loads(BASE_CONFIG.read_text())["text_config"]
    return get_omni_mot_model_descriptor(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        use_moe="num_experts" in config,
        latent_patch_size=2,
        latent_channel_size=48,
        action_dim=64,
        predict_text_tokens=False,
    )


def flops_for_microbatch(
    scene: str, microbatch: dict, cfg, *, recompute: bool = True, physical_padding: bool = False
) -> dict[str, float]:
    splits = microbatch["split_lens"]
    if len(splits) != 2 * microbatch["samples"]:
        raise ValueError("Malformed packed sample splits")
    pairs = list(zip(splits[::2], splits[1::2], strict=True))
    actions = [32 if scene == "action" else 0] * len(pairs)
    if any(g < a for (_, g), a in zip(pairs, actions, strict=True)):
        raise ValueError("Action split shorter than 32 tokens")
    base_splits = [value for (u, g), a in zip(pairs, actions, strict=True) for value in (u, g - a)]
    u_total = sum(splits[::2])
    v_total = sum(g - a for (_, g), a in zip(pairs, actions, strict=True))
    tokens = microbatch["tokens"]
    if physical_padding and scene != "t2v":
        raise ValueError("Physical padding estimate is implemented only for T2V")
    # The fixed decoder pads UND and GEN independently and exposes the pad as
    # one extra (causal, full) attention sample. The callback's real split_lens
    # omit this pair, so reconstruct it from the physical lengths here.
    priced_splits = base_splits
    if physical_padding:
        pad_und = tokens["physical_und_token_length"] - u_total
        pad_gen = tokens["physical_gen_token_length"] - v_total
        if pad_und <= 0 or pad_gen <= 0:
            raise ValueError(f"Expected a nonempty paired fixed pad, got UND={pad_und}, GEN={pad_gen}")
        priced_splits = [*base_splits, pad_und, pad_gen]
    base = compute_omni_mot_flops_per_batch(
        cfg=cfg,
        B=1,
        text_tokens=tokens["physical_und_token_length"] if physical_padding else u_total,
        vision_tokens=tokens["physical_gen_token_length"] if physical_padding else v_total,
        freeze_und=False,
        vision_gen=True,
        action_gen=False,
        backwardpass_ratio=2.0,
        split_lens=priced_splits,
        attn_modes=["causal", "full"] * (len(pairs) + int(physical_padding)),
        include_padding=False,
        use_activation_checkpointing=recompute,
    )
    if scene == "t2v":
        return {"base_und_video_flops": float(base), "small_action_flops": 0.0, "total_flops": float(base)}

    shape = action_shape("width_only_1b")
    d, m, q, k, dh = shape.hidden, shape.intermediate, shape.q_heads, shape.kv_heads, shape.head_dim
    qdim, kdim = q * dh, k * dh
    action_tokens = sum(actions)
    # Each dense linear MAC costs two FLOPs. The small tower has Q/K/V/O,
    # gate/up/down, and two routed action input/output projections. Pairwise
    # attention terms follow the actual UND/VIDEO/ACTION visibility in
    # small_layer_forward. Multiply by three for forward + two backward.
    projection_fwd = 2 * action_tokens * (d * qdim + 2 * d * kdim + qdim * d + 3 * d * m)
    attention_fwd = 0
    for (u, g), a in zip(pairs, actions, strict=True):
        v = g - a
        attention_fwd += 4 * q * dh * a * (u + v + a) + 5 * q * a * (u + v + a)
        attention_fwd += 4 * cfg.num_attention_heads * cfg.head_dim * v * a
        attention_fwd += 5 * cfg.num_attention_heads * v * a
    action_fwd = len(shape.layers) * (projection_fwd + attention_fwd) + 4 * 64 * d * action_tokens
    action_train = 3 * action_fwd
    total = Decimal(base) + action_train
    return {
        "base_und_video_flops": float(base),
        "small_action_flops": float(action_train),
        "total_flops": float(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("t2v", "action"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    cfg = descriptor()
    cache: dict[tuple, tuple[dict[str, float], dict[str, float], dict[str, float]]] = {}
    for path in sorted(args.run.glob("**/seqlen_efficiency/rank_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["window"] != "steady":
                continue
            components = []
            no_recompute = []
            physical = []
            for mb in record["microbatches"]:
                key = (
                    tuple(mb["split_lens"]),
                    mb["tokens"]["physical_und_token_length"],
                    mb["tokens"]["physical_gen_token_length"],
                )
                if key not in cache:
                    useful = flops_for_microbatch(args.scene, mb, cfg)
                    legacy = flops_for_microbatch(args.scene, mb, cfg, recompute=False)
                    executed = (
                        flops_for_microbatch(args.scene, mb, cfg, physical_padding=True)
                        if args.scene == "t2v"
                        else useful
                    )
                    cache[key] = (useful, legacy, executed)
                useful, legacy, executed = cache[key]
                components.append(useful)
                no_recompute.append(legacy)
                physical.append(executed)
            rows.append(
                {
                    "rank": record["rank"],
                    "iteration": record["iteration"],
                    "base_und_video_flops": sum(c["base_und_video_flops"] for c in components),
                    "small_action_flops": sum(c["small_action_flops"] for c in components),
                    "total_flops": sum(c["total_flops"] for c in components),
                    "no_recompute_flops": sum(c["total_flops"] for c in no_recompute),
                    "physical_proxy_flops": sum(c["total_flops"] for c in physical),
                }
            )
    if not rows:
        raise ValueError(f"No steady-state benchmark records under {args.run}")
    by_step = {}
    for row in rows:
        by_step.setdefault(row["iteration"], []).append(row)
    benchmark = {}
    for path in sorted(args.run.glob("**/seqlen_efficiency/rank_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["window"] == "steady":
                benchmark[(record["rank"], record["iteration"])] = record
    total_flops = 0.0
    total_no_recompute_flops = 0.0
    total_physical_proxy_flops = 0.0
    total_wall = 0.0
    for step, step_rows in sorted(by_step.items()):
        if len(step_rows) != WORLD_SIZE:
            raise ValueError(f"Step {step} has {len(step_rows)} rank records, expected {WORLD_SIZE}")
        total_flops += sum(row["total_flops"] for row in step_rows)
        total_no_recompute_flops += sum(row["no_recompute_flops"] for row in step_rows)
        total_physical_proxy_flops += sum(row["physical_proxy_flops"] for row in step_rows)
        total_wall += max(benchmark[(row["rank"], step)]["seconds"] for row in step_rows)
    result = {
        "scene": args.scene,
        "method": "MFU: real-token forward+backward, no AC or padding; HFU estimate: forward+backward+full-block AC with physical UND/GEN and paired padding attention; no hardware counters",
        "h100_bf16_dense_peak_tflops": PEAK_TFLOPS,
        "world_size": WORLD_SIZE,
        "steady_steps": sorted(by_step),
        "total_useful_flops_all_ranks": total_flops,
        "total_no_recompute_flops_all_ranks": total_no_recompute_flops,
        "total_physical_proxy_flops_all_ranks": total_physical_proxy_flops,
        "sum_slowest_rank_step_seconds": total_wall,
        "mfu_pct": total_no_recompute_flops / (WORLD_SIZE * PEAK_TFLOPS * 1e12 * total_wall) * 100,
        "hfu_proxy_pct": total_physical_proxy_flops / (WORLD_SIZE * PEAK_TFLOPS * 1e12 * total_wall) * 100,
        "ac_real_pct": total_flops / (WORLD_SIZE * PEAK_TFLOPS * 1e12 * total_wall) * 100,
        "legacy_no_recompute_mfu_pct": total_no_recompute_flops / (WORLD_SIZE * PEAK_TFLOPS * 1e12 * total_wall) * 100,
        "rank_steps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"MFU={result['mfu_pct']:.2f}%, HFU proxy={result['hfu_proxy_pct']:.2f}% on {len(by_step)} steady updates")


if __name__ == "__main__":
    main()
