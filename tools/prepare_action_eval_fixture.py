"""Freeze early/middle/late windows from each held-out Action catalog episode."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prefix-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = torch.load(args.prefix_fixture, map_location="cpu", weights_only=False)
    batches = copy.deepcopy(fixture["batches"])
    references = copy.deepcopy(fixture["reference_batches"])
    samples = copy.deepcopy(fixture["samples"])
    seen = {sample["dataset_index"] for sample in samples}
    processor = create_qwen2_tokenizer_with_download(args.tokenizer, "hf")
    dataset = build_action_dataset_from_catalog(args.catalog, tokenizer_config=processor)
    for number, (start, length) in enumerate(dataset.get_shuffle_blocks()):
        for offset in sorted({0, (length - 1) // 2, length - 1}):
            if start + offset in seen:
                continue
            item = dataset[start + offset]
            batch = VFMListCollator().collate([item])
            references.append(copy.deepcopy(batch))
            for video in batch["video"][0]:
                video[:, 1:] = video[:, :1].clone()
            for action in batch["action"][0]:
                action.zero_()
            for action in batch.get("action_raw", [[]])[0]:
                action.zero_()
            assert all(
                not p.robot_state_frame_indexes and not p.condition_frame_indexes_action for p in batch["sequence_plan"]
            )
            assert all(p.condition_frame_indexes_vision == [0] for p in batch["sequence_plan"])
            batches.append(batch)
            samples.append(
                dict(
                    episode=int(dataset._dataset._ep_vals[number]),
                    window_offset=offset,
                    dataset_index=start + offset,
                    caption=item["ai_caption"],
                )
            )
            seen.add(start + offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        catalog=str(args.catalog),
        catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        prefix_fixture=str(args.prefix_fixture),
        samples=samples,
    )
    torch.save(dict(batches=batches, reference_batches=references, samples=samples, manifest=manifest), args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(samples)} windows from {len({s['episode'] for s in samples})} episodes: {args.output}")


if __name__ == "__main__":
    main()
