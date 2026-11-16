#!/usr/bin/env python3
"""Create deterministic offline-latent inputs for the fixed-pack seqlen study.

The tensors are synthetic. They exercise the production catalog, action transform,
packing, and model paths; they are not a dataset for model-quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LATENT_T = 9  # 33 source frames with Wan 4x temporal compression.
LATENT_CHANNELS = 48
VIDEO_TARGET = 50_000
CAPTION = "A robot moves an object across a table."


def choose_latent_shape(samples_per_pack: int, scene: str = "t2v") -> tuple[int, int, int, int]:
    """Keep real pack work near 50K while changing per-sample spatial tokens."""
    if samples_per_pack not in SAMPLE_COUNTS:
        raise ValueError(f"unsupported samples_per_pack={samples_per_pack}")
    if scene not in ("t2v", "action"):
        raise ValueError(f"unsupported scene={scene}")
    # The model's configured resolution shift only accepts 256/480/720 tiers.
    # At nine latent frames, low-N packs need a wide aspect ratio so one pixel
    # dimension stays <=960 while the real pack remains near 50K tokens.
    if samples_per_pack in (1, 2, 4):
        grid_w = {1: 184, 2: 93, 4: 46}[samples_per_pack]
        return (LATENT_CHANNELS, LATENT_T, 60, 2 * grid_w)
    # The real tokenizer preflight gives 19 UND tokens for this fixed caption.
    # Action adds exactly 32 more tokens per sample. Subtract those tokens so
    # the total *real* workload stays near 50K across Action pack counts.
    target_video = VIDEO_TARGET - samples_per_pack * (51 if scene == "action" else 19)
    target_area = target_video / (samples_per_pack * LATENT_T)
    center_h = max(1, round(math.sqrt(target_area * 8 / 9)))
    candidates = []
    for grid_h in range(max(1, center_h - 3), center_h + 4):
        for grid_w in range(max(1, round(target_area / grid_h) - 3), round(target_area / grid_h) + 4):
            tokens = LATENT_T * grid_h * grid_w * samples_per_pack
            token_error = abs(tokens - target_video) / target_video
            aspect_error = abs(grid_h / grid_w - 8 / 9)
            candidates.append((token_error + 0.25 * aspect_error, token_error, grid_h, grid_w))
    _, _, grid_h, grid_w = min(candidates)
    return (LATENT_CHANNELS, LATENT_T, 2 * grid_h, 2 * grid_w)


def _write_catalog(path: Path, rows: list[dict[str, str]], row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


def _digest(path: Path) -> str:
    hash_state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_state.update(chunk)
    return hash_state.hexdigest()


def create_t2v(root: Path, n: int, packs_per_rank: int, seed: int, world_size: int) -> dict:
    shape = choose_latent_shape(n)
    path = root / "latent.safetensors"
    generator = torch.Generator().manual_seed(seed + n)
    latent = torch.randn(shape, generator=generator).to(torch.bfloat16).contiguous()
    save_file({"synthetic/video": latent}, str(path))
    rows = []
    count = world_size * packs_per_rank * n
    for index in range(count):
        data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": CAPTION}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "gen_video",
                            "vae_latent": {
                                "path": str(path.resolve()),
                                "tensor_key": "synthetic/video",
                                "latent_shape": list(shape),
                                "target_fps": 24.0,
                            },
                            "conditioning_latent_frames": 0,
                        }
                    ],
                },
            ]
        }
        rows.append(
            {
                "id": f"t2v-{n}-{index:06d}",
                "type": "visual_generation",
                "dataset": "synthetic_t2v",
                "data": json.dumps(data, separators=(",", ":")),
                "provenance": json.dumps({"synthetic": True, "index": index}),
            }
        )
    catalog = root / "catalog.parquet"
    _write_catalog(catalog, rows, min(2_048, packs_per_rank * n))
    return {"catalog": str(catalog.resolve()), "samples": count, "latent_sha256": _digest(path)}


def create_action(root: Path, n: int, packs_per_rank: int, seed: int, world_size: int) -> dict:
    shape = choose_latent_shape(n, "action")
    path = root / "latent.safetensors"
    generator = torch.Generator().manual_seed(seed + n + 10_000)
    latent = torch.randn(shape, generator=generator).to(torch.bfloat16).contiguous()
    height, width = shape[2] * 16, shape[3] * 16
    tensors = {
        "0.latent": latent,
        "0.action": torch.randn((32, 8), generator=generator).contiguous(),
        "0.image_size": torch.tensor([height, width, height, width]),
        "0.conditioning_fps": torch.tensor(15),
        "0.domain_id": torch.tensor(0),
    }
    save_file(tensors, str(path))
    count = world_size * packs_per_rank * n
    record = {
        "shard": path.name,
        "key": "0",
        "metadata": {"mode": "wam", "viewpoint": "concat_view", "ai_caption": CAPTION},
        "tensor_keys": ["action", "image_size", "conditioning_fps", "domain_id"],
        "video_shape": [3, 33, height, width],
        "latent_shape": list(shape),
    }
    records = [{**record, "source_index": index} for index in range(count)]
    manifest = {
        "format": "cosmos_action_latents_v1",
        "complete": True,
        "blocks": [[block * packs_per_rank * n, packs_per_rank * n] for block in range(world_size)],
        "records": records,
        "resolution": "256",
        "transform": {
            "max_action_dim": 64,
            "append_viewpoint_info": False,
            "append_duration_fps_timestamps": False,
            "append_resolution_info": False,
        },
        "vae_weights_sha256": "0" * 64,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    row = {
        "id": f"action-{n}",
        "type": "action",
        "dataset": "synthetic_action",
        "data": json.dumps({"adapter": "droid_offline", "manifest_path": str(manifest_path.resolve())}),
        "provenance": json.dumps({"synthetic": True}),
    }
    catalog = root / "catalog.parquet"
    _write_catalog(catalog, [row], 1)
    return {"catalog": str(catalog.resolve()), "samples": count, "latent_sha256": _digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packs-per-rank", type=int, default=24)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", nargs="+", choices=("t2v", "action"), default=["t2v"])
    args = parser.parse_args()
    if args.packs_per_rank < 1 or args.world_size < 1:
        parser.error("--packs-per-rank and --world-size must be positive")
    for scene in args.scenes:
        for n in SAMPLE_COUNTS:
            root = (args.output / scene / f"n{n:03d}").resolve()
            root.mkdir(parents=True, exist_ok=True)
            if (root / "manifest.study.json").exists():
                raise FileExistsError(f"refusing to overwrite existing study input: {root}")
            result = (create_t2v if scene == "t2v" else create_action)(
                root, n, args.packs_per_rank, args.seed, args.world_size
            )
            shape = choose_latent_shape(n, scene)
            report = {
                "scene": scene,
                "samples_per_pack": n,
                "seed": args.seed,
                "packs_per_rank": args.packs_per_rank,
                "world_size": args.world_size,
                "latent_shape": shape,
                "video_tokens_per_sample": shape[1] * (shape[2] // 2) * (shape[3] // 2),
                "video_tokens_per_pack": n * shape[1] * (shape[2] // 2) * (shape[3] // 2),
                "synthetic": True,
                **result,
            }
            (root / "manifest.study.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{scene} n={n}: VIDEO={report['video_tokens_per_pack']}, shape={shape}")


if __name__ == "__main__":
    main()
