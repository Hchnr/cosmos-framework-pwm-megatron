#!/usr/bin/env python3
"""Create episode-disjoint LIBERO catalogs and frozen held-out visualization inputs."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--task", default="put the bowl on the plate")
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = []
    for file in sorted((args.root / "meta/episodes").rglob("*.parquet")):
        rows.extend(
            r
            for r in pq.read_table(file, columns=["episode_index", "tasks", "length"]).to_pylist()
            if r["tasks"] == [args.task]
        )
    rows.sort(key=lambda r: r["episode_index"])
    random.Random(args.seed).shuffle(rows)
    if not 0 < args.train_episodes < len(rows):
        raise ValueError("Need nonempty train and held-out episode sets")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"task": args.task, "root": str(args.root), "seed": args.seed, "splits": {}}
    for split, episodes in [("train", rows[: args.train_episodes]), ("val", rows[args.train_episodes :])]:
        ids = sorted(r["episode_index"] for r in episodes)
        data = dict(
            adapter="libero",
            format="lerobot",
            root=str(args.root),
            split="full",
            episode_ids=ids,
            task_names=[args.task],
            use_state=False,
            mode="wam",
            chunk_length=16,
            fps=20,
            resolution="256",
            camera_mode="concat_view",
            action_space="frame_wise_relative",
            rotation_space="6d",
            pose_coordinate_frame="native",
            action_normalization="quantile_rot",
            append_idle_frames=False,
            append_viewpoint_info=False,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
            cfg_dropout_rate=0.1 if split == "train" else 0.0,
            sample_stride=1,
        )
        row = dict(
            id=f"libero_bowl_plate:{split}:v1",
            type="action",
            dataset=f"libero_bowl_plate_{split}",
            data=json.dumps(data),
            provenance=json.dumps(
                {
                    "source_path": str(args.root),
                    "task": args.task,
                    "seed": args.seed,
                    "episode_ids": ids,
                    "robot_state_input": False,
                }
            ),
        )
        path = args.output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist([row]), path)
        manifest["splits"][split] = dict(
            episode_ids=ids,
            episodes=len(ids),
            frames=sum(r["length"] for r in episodes),
            windows=sum(max(0, r["length"] - 16) for r in episodes),
            catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    assert set(manifest["splits"]["train"]["episode_ids"]).isdisjoint(manifest["splits"]["val"]["episode_ids"])
    from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
    from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog
    from cosmos_framework.data.generator.dataflow.collators import VFMListCollator

    processor = create_qwen2_tokenizer_with_download(args.tokenizer, "hf")
    dataset = build_action_dataset_from_catalog(args.output / "val.parquet", tokenizer_config=processor)
    raw = dataset._dataset
    batches, reference_batches, samples = [], [], []
    # Distinct held-out episodes at early/middle/late positions; manifest fixes all choices.
    for number, (start, length) in enumerate(dataset.get_shuffle_blocks()[:3]):
        offset = [0, (length - 1) // 2, length - 1][number]
        item = dataset[start + offset]
        batch = VFMListCollator().collate([item])
        reference_batches.append(copy.deepcopy(batch))
        # Only current image and text are available to the policy. Never let the
        # future expert frames/actions reach the model, even through VAE encoding.
        for video in batch["video"][0]:
            video[:, 1:] = video[:, :1].clone()
        for action in batch["action"][0]:
            action.zero_()
        if "action_raw" in batch:
            for action in batch["action_raw"][0]:
                action.zero_()
        assert not batch["sequence_plan"][0].robot_state_frame_indexes
        assert not batch["sequence_plan"][0].condition_frame_indexes_action
        batches.append(batch)
        samples.append(
            dict(
                episode=int(raw._ep_vals[number]),
                window_offset=offset,
                dataset_index=start + offset,
                caption=item["ai_caption"],
                video_shape=list(item["video"].shape),
            )
        )
    fixture = dict(
        batches=batches,
        reference_batches=reference_batches,
        samples=samples,
        seed=args.seed,
        catalog=str(args.output / "val.parquet"),
    )
    torch.save(fixture, args.output / "eval_batches.pt")
    manifest["evaluation_samples"] = samples
    manifest["eval_batches_sha256"] = hashlib.sha256((args.output / "eval_batches.pt").read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
