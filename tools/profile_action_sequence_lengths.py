#!/usr/bin/env python3
"""Decode real DROID windows and profile Nano token geometry across recipes.

Text lengths use the training tokenizer/transform/packer. Vision lengths follow
Wan's 4x16x16 compression, latent cropping and 2x2 spatial patches; --vae also
verifies the first sample of each recipe with the actual encoder.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog
from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--vae")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    processor = create_qwen2_tokenizer_with_download(args.tokenizer, "hf")
    _, special = add_special_tokens(processor.tokenizer)
    special["eos_token_id"] = processor.tokenizer.eos_token_id
    dataset = build_action_dataset_from_catalog(args.catalog, tokenizer_config=processor)
    raw_dataset = dataset._dataset
    assert raw_dataset._chunk_length == 32 and not raw_dataset._use_state
    candidates, offset = [], 0
    for _, _, length, episode in raw_dataset._episode_records:
        candidates.append((episode, offset))
        offset += length
    # Spread across distinct early episodes, keeping this bounded probe local to
    # a small number of media files. This is not a global p99 estimate.
    selected = sorted(candidates)[: args.samples]
    transform = ActionTransformPipeline(tokenizer_config=processor, max_action_dim=64, cfg_dropout_rate=0.0)
    vae = None
    if args.vae:
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

        vae = Wan2pt2VAEInterface(vae_path=args.vae, encode_exact_durations=[5, 17, 33])
    results = {f"c{chunk}_r{res}": [] for chunk, res in [(4, "256"), (16, "256"), (32, "256"), (32, "480")]}
    for sample_number, (episode, index) in enumerate(selected):
        raw = raw_dataset[index]
        for key, rows in results.items():
            chunk, resolution = key.split("_")
            chunk, resolution = int(chunk[1:]), resolution[1:]
            value = dict(raw)
            value["video"] = raw["video"][:, : chunk + 1]
            value["action"] = raw["action"][:chunk]
            value = transform(copy.deepcopy(value), resolution)
            frame_size = value["image_size"].tolist()
            latent_shape = [1, 48, 1 + chunk // 4, int(frame_size[2]) // 16, int(frame_size[3]) // 16]
            if sample_number == 0 and vae is not None:
                pixels = value["video"].unsqueeze(0).cuda().float() / 127.5 - 1
                with torch.no_grad():
                    latent = vae.encode(pixels)
                actual = list(latent[..., : latent_shape[3], : latent_shape[4]].shape)
                assert actual == latent_shape, (key, actual, latent_shape)
            builder = PackedSequenceBuilder()
            und = builder.pack_text_tokens(value["text_token_ids"].tolist(), special, True, supervise_text=False)
            # patchify_and_pack_latents pads odd spatial dimensions to 2x2.
            vision = latent_shape[2] * ((latent_shape[3] + 1) // 2) * ((latent_shape[4] + 1) // 2)
            rows.append(
                {
                    "episode": episode,
                    "flat_index": index,
                    "und": und,
                    "video": vision,
                    "action": chunk,
                    "total": und + vision + chunk,
                    "pixel_shape": list(value["video"].shape),
                    "latent_shape": latent_shape,
                    "raw_action_dim": int(value["raw_action_dim"]),
                    "first_sample_vae_verified": bool(sample_number == 0 and vae is not None),
                }
            )
        print(f"profiled sample {sample_number + 1}/{len(selected)}, episode={episode}", flush=True)
    summary = {}
    for key, rows in results.items():
        summary[key] = {
            metric: {str(p): float(np.percentile([r[metric] for r in rows], p)) for p in [0, 50, 95, 100]}
            for metric in ["und", "video", "action", "total"]
        }
    payload = {
        "catalog": args.catalog,
        "dataset_windows": len(dataset),
        "sampling": "first valid window of early distinct train episodes; CFG dropout disabled",
        "results": results,
        "percentiles": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
