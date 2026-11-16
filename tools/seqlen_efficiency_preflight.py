#!/usr/bin/env python3
"""Audit all synthetic study inputs through the real catalog/packing pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from cosmos_framework.data.generator.catalog_processors import build_catalog_training_dataloader
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution
from tools.seqlen_efficiency_data import SAMPLE_COUNTS

CAPACITY = {
    "t2v": dict(fixed_und_tokens=5_958, fixed_gen_tokens=59_578, fixed_action_tokens=0, max_sequence_length=65_536),
    "action": dict(
        fixed_und_tokens=11_776,
        fixed_gen_tokens=62_592,
        fixed_action_tokens=8_256,
        max_sequence_length=74_368,
    ),
}


def inspect_group(root: Path, scene: str, n: int, processor: Qwen3VLProcessor) -> dict:
    group = root / scene / f"n{n:03d}"
    source = json.loads((group / "manifest.study.json").read_text())
    digest = hashlib.sha256((group / "latent.safetensors").read_bytes()).hexdigest()
    if digest != source["latent_sha256"]:
        raise ValueError(f"{scene} N={n}: latent SHA256 does not match manifest")
    latent_shape = source["latent_shape"]
    pixel_shape = (latent_shape[2] * 16, latent_shape[3] * 16)
    tier = get_vision_data_resolution(pixel_shape)
    if tier not in ("256", "480", "720"):
        raise ValueError(f"{scene} N={n}: resolution tier {tier} is absent from model shift config")
    loader = build_catalog_training_dataloader(
        [{"path": source["catalog"], "ratio": 1}],
        processor=processor,
        vae_mode="offline",
        conditioning_config={0: 1.0, 1: 0.0, 2: 0.0},
        seed=42,
        max_samples_per_batch=n,
        num_workers=0,
        static_packing_config={
            "spatial_compression_factor": 16,
            "temporal_compression_factor": 4,
            "latent_patch_size": 2,
        },
        fixed_sequence_length_enabled=True,
        **CAPACITY[scene],
    )
    batch = next(iter(loader))
    packed = batch["packed_sequence"]
    split_lens = list(packed.split_lens)
    actual_n = len(batch["sequence_plan"])
    if actual_n != n or len(split_lens) != 2 * n:
        raise ValueError(f"{scene} N={n}: actual_n={actual_n}, split_lens={len(split_lens)}")
    und = sum(split_lens[0::2])
    gen = sum(split_lens[1::2])
    video = source["video_tokens_per_pack"]
    action = 32 * n if scene == "action" else 0
    if gen != video + action:
        raise ValueError(f"{scene} N={n}: GEN={gen} != VIDEO={video} + ACTION={action}")
    capacity = CAPACITY[scene]
    if und >= capacity["fixed_und_tokens"] or gen >= capacity["fixed_gen_tokens"]:
        raise ValueError(f"{scene} N={n}: packed sample exceeds fixed capacity")
    return {
        "scene": scene,
        "samples_per_pack": n,
        "physical_total": capacity["max_sequence_length"],
        "physical_und": capacity["fixed_und_tokens"],
        "physical_gen": capacity["fixed_gen_tokens"],
        "physical_action": capacity["fixed_action_tokens"],
        "real_und": und,
        "real_video": video,
        "real_action": action,
        "real_gen": gen,
        "real_total": und + gen,
        "mean_real_sample_length": (und + gen) / n,
        "mean_real_video_length": video / n,
        "sample_lengths": list(packed.sample_lens),
        "latent_shape": source["latent_shape"],
        "synthetic": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", default="/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--scenes", nargs="+", choices=("t2v", "action"), default=["t2v"])
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    processor = Qwen3VLProcessor(name=args.processor)
    rows = [inspect_group(args.inputs, scene, n, processor) for scene in args.scenes for n in SAMPLE_COUNTS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['scene']} N={row['samples_per_pack']}: physical={row['physical_total']} "
            f"real={row['real_total']} mean_sample={row['mean_real_sample_length']:.1f}"
        )


if __name__ == "__main__":
    main()
