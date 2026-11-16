"""Freeze a small held-out DROID flow-validation fixture via the Parquet loader."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    selected = manifest["droid_offline_evaluation"]["episodes"][: args.samples]
    if not selected or len(selected) != args.samples:
        raise ValueError("Requested samples must fit the frozen evaluation selection")
    catalog = args.data_dir / "droid_success/val.parquet"
    processor = create_qwen2_tokenizer_with_download(args.tokenizer, "hf")
    ds = build_action_dataset_from_catalog(catalog, tokenizer_config=processor)
    starts = {}
    cumulative = 0
    for _, _, length, episode in ds._dataset._episode_records:
        starts[episode] = cumulative
        cumulative += length
    references, samples = [], []
    for index, row in enumerate(selected):
        random.seed(2026 + index)
        np.random.seed(2026 + index)
        torch.manual_seed(2026 + index)
        offset = row["offset_frames"][len(row["offset_frames"]) // 2]
        dataset_index = starts[row["episode"]] + offset // 16
        _, _, episode, _ = ds._dataset._resolve_index(dataset_index)
        # _resolve_index returns dataset, absolute frame, episode, window offset.
        if int(episode) != row["episode"]:
            raise AssertionError((episode, row))
        item = ds[dataset_index]
        plan = item["sequence_plan"]
        assert not plan.robot_state_frame_indexes and not plan.condition_frame_indexes_action
        assert torch.isfinite(item["action"]).all() and int(item["raw_action_dim"]) == 8
        references.append(VFMListCollator().collate([item]))
        samples.append(
            dict(
                episode=row["episode"],
                source_episode_id=row["source_episode_id"],
                window_offset=offset,
                dataset_index=dataset_index,
                caption=item["ai_caption"],
            )
        )
        print(index, row["episode"], offset, flush=True)
    output = catalog.parent / "flow_eval_batches.pt"
    torch.save(dict(reference_batches=references, samples=samples, seed=2026, catalog=str(catalog)), output)
    report = dict(
        samples=samples,
        fixture_path=str(output),
        fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        no_robot_state=True,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
