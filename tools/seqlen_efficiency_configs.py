#!/usr/bin/env python3
"""Derive all fixed-pack experiment TOMLs from the checked-in recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tomli_w
import tomllib

from tools.seqlen_efficiency_data import SAMPLE_COUNTS

BASE = {
    "t2v": Path("examples/toml/sft_config/vision_sft_nano_t2v_4gpu_fixed_static.toml"),
    "action": Path("examples/toml/sft_config/action_policy_1b.toml"),
}


def make_config(scene: str, n: int, source: dict) -> dict:
    with BASE[scene].open("rb") as stream:
        cfg = tomllib.load(stream)
    cfg["job"].update(
        group="v0.2.6_seqlen_efficiency_synthetic_4gpu",
        name=f"{scene}_fixedpack_n{n:03d}",
        wandb_mode="disabled",
    )
    cfg["trainer"].update(max_iter=10, logging_iter=1)
    cfg["checkpoint"].update(save_iter=1_000, load_training_state=False)
    loader = cfg["dataloader_train"]
    loader.update(
        max_samples_per_batch=n,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=False,
        datasets=[{"path": source["catalog"], "ratio": 1}],
    )
    loader["visual_generation"]["vae_mode"] = "offline"
    if scene == "t2v":
        cfg["model"]["fixed_sequence_length"].update(und_tokens=5_958, gen_tokens=59_578)
        # Keep the four-way shard and GA=2. The tokenizer is not
        # loaded in offline mode; the real Qwen processor still reads captions.
        cfg["trainer"]["compile_cache"]["path"] = ".compile_cache/v026_t2v_fixedpack_4gpu"
    else:
        cfg["model"]["action_tower_und_short_tokens"] = 0
        cfg["model"]["fixed_sequence_length"].update(und_tokens=11_776, gen_tokens=62_592, action_tokens=8_256)
        cfg["trainer"]["compile_cache"]["path"] = ".compile_cache/v026_action_fixedpack_4gpu"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=("t2v", "action"), default=["t2v"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        for n in SAMPLE_COUNTS:
            source = json.loads((args.inputs / scene / f"n{n:03d}" / "manifest.study.json").read_text())
            cfg = make_config(scene, n, source)
            target = args.output / f"{scene}_n{n:03d}.toml"
            target.write_text(tomli_w.dumps(cfg))
            print(target)


if __name__ == "__main__":
    main()
