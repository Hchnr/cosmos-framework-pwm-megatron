"""CPU preflight of prepared v0.2.1 catalogs through the existing Action loader.

This checks indexes and one decoded validation window per source, not full
media coverage, model inference, FSDP training, or online W&B.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog
from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
from cosmos_framework.data.generator.sequence_packing.static_layout import (
    StaticPackingConfig,
    build_static_packed_sequence,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        required=True,
        help="Validated draft config report containing static_packing_config",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    geometry = json.loads(args.geometry_report.read_text())["two"]["static_packing_config"]
    config = create_qwen2_tokenizer_with_download(args.tokenizer, "hf")
    report = {}
    for name, specification in manifest.items():
        if "train" not in specification:
            continue
        counts = {}
        for split in ["train", "val"]:
            dataset = build_action_dataset_from_catalog(
                args.data_dir / name / f"{split}.parquet", tokenizer_config=config
            )
            counts[split] = len(dataset)
            assert counts[split] == specification[split]["windows"], (name, split, counts)
        sample = dataset[0]
        plan = sample["sequence_plan"]
        assert not plan.robot_state_frame_indexes and not plan.condition_frame_indexes_action
        assert torch.isfinite(sample["action"]).all()
        assert sample["action"].shape[0] == specification["val"]["chunk_length"]
        processor = dataset._transform.text_tokenizer._processor
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer, special = add_special_tokens(tokenizer)
        special["eos_token_id"] = int(tokenizer.eos_token_id)
        packed = build_static_packed_sequence([sample], special_tokens=special, config=StaticPackingConfig(**geometry))
        report[name] = dict(
            windows=counts,
            video_shape=list(sample["video"].shape),
            action_shape=list(sample["action"].shape),
            domain_id=int(sample["domain_id"]),
            raw_action_dim=int(sample["raw_action_dim"]),
            text_tokens=int(sample["text_token_ids"].numel()),
            sequence_length=packed.sequence_length,
            split_lens=packed.split_lens,
            attention_modes=packed.attn_modes,
            robot_state_indexes=plan.robot_state_frame_indexes,
            action_history_indexes=plan.condition_frame_indexes_action,
        )
        print(name, report[name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
