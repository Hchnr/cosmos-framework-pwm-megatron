"""Estimate ZeRO-2 persistent storage from actual checkpoint tensor metadata."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-degree", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = FileSystemReader(args.checkpoint / "model").read_metadata()
    patterns = ("moe_gen", "time_embedder", "vae2llm", "llm2vae", "action2llm", "llm2action", "action_modality_embed")
    total = train = storage = 0
    for name, item in metadata.state_dict_metadata.items():
        if not hasattr(item, "size"):
            continue
        count = math.prod(item.size)
        storage += count * torch.empty((), dtype=item.properties.dtype).element_size()
        total += count
        if item.properties.dtype.is_floating_point and any(pattern in name for pattern in patterns):
            train += count
    report = {
        "checkpoint": str(args.checkpoint),
        "model_tensor_elements_including_buffers": total,
        "model_storage_bytes": storage,
        "trainable_name_selected_float_elements_estimate": train,
        "dp_shard_degree": args.shard_degree,
        "fp32_master_plus_moments_bytes_estimate": train * 12 / args.shard_degree,
        "bf16_sharded_gradient_bytes_estimate": train * 2 / args.shard_degree,
        "checkpoint_input_activation_bytes_estimate_short_und": 36 * 2 * (5120 * 4096 + 59648 * 4096 + 2944 * 1280),
        "checkpoint_input_activation_bytes_estimate_full_und": 36 * 2 * (11776 * 4096 + 59648 * 4096 + 2944 * 1280),
        "exclusions": ["VAE/encoding workspaces", "attention/MLP temporaries", "allocator/NCCL buffers"],
        "status": "analytical estimate, not full-depth measured peak",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
