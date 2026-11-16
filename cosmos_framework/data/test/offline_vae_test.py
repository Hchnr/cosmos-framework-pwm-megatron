# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from cosmos_framework.data.generator.catalog import CatalogSchemaError
from cosmos_framework.data.generator.catalog_processors import (
    CatalogVisualGenerationProcessor,
    OfflineVaeLineageProcessor,
    build_catalog_training_dataloader,
)
from cosmos_framework.data.generator.dataflow.batchers import SequenceTokenCounts, SequentialPackingBatcher
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.dataflow.processors import IdentityProcessor
from cosmos_framework.data.generator.offline_vae import (
    SafetensorsLatentReader,
    VaeLatentLoadError,
    VaeLatentRef,
    parse_vae_latent_ref,
    validate_offline_vae_manifests,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.static_layout import (
    PACKED_SEQUENCE_KEY,
    StaticPackingConfig,
)

_SHA = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
_SPECIAL_TOKENS = {"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}


class _Tokenizer:
    eos_token_id = 2

    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return []


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, _messages, **_kwargs):
        return {
            "input_ids": torch.tensor([10, 11, 12], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.bool),
        }

    def add_assistant_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input_ids, dtype=torch.bool)


class _DecoderMustNotRun:
    def decode(self, *_args, **_kwargs):
        raise AssertionError("offline VAE processing must not decode target pixels")


def _latent_value(path: Path, *, shape: list[int] | None = None) -> dict:
    return {
        "path": str(path),
        "tensor_key": "dataset/sample-0",
        "latent_shape": shape or [48, 2, 4, 6],
        "target_fps": 24.0,
    }


def _catalog_row(path: Path, *, include_pixels: bool = False) -> dict:
    target = {
        "type": "gen_video",
        "vae_latent": _latent_value(path),
        "conditioning_latent_frames": 0,
    }
    if include_pixels:
        target["gen_video"] = "/unused.mp4"
    return {
        "id": "row-0",
        "type": "visual_generation",
        "dataset": "dataset",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "move"}]},
                {"role": "assistant", "content": [target]},
            ]
        },
        "provenance": {},
    }


def _manifest(path: Path, *, dataset: str, vae_tag: str = "wan2pt2_tokenizer-t480-test") -> Path:
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "vae_name": "wan2pt2_tokenizer",
                "vae_tag": vae_tag,
                "vae_weights_sha256": _SHA,
            }
        )
    )
    return path


def test_parse_latent_reference_validates_metadata_without_opening_shard(tmp_path: Path) -> None:
    path = (tmp_path / "missing.safetensors").resolve()

    ref = parse_vae_latent_ref(
        _latent_value(path),
        row_id="row-0",
        dataset="dataset",
        expected_channels=48,
        max_tensor_bytes=10_000,
    )

    assert ref == VaeLatentRef(str(path), "dataset/sample-0", (48, 2, 4, 6), 24.0, "row-0", "dataset")
    assert ref.expected_nbytes == 4_608
    with pytest.raises(CatalogSchemaError, match="must be absolute"):
        parse_vae_latent_ref(
            {**_latent_value(path), "path": "relative.safetensors"},
            row_id="row-0",
            dataset="dataset",
            expected_channels=48,
            max_tensor_bytes=10_000,
        )
    with pytest.raises(CatalogSchemaError, match="state_ch=48"):
        parse_vae_latent_ref(
            _latent_value(path, shape=[16, 2, 4, 6]),
            row_id="row-0",
            dataset="dataset",
            expected_channels=48,
            max_tensor_bytes=10_000,
        )


def test_reader_loads_one_bf16_key_and_reports_handle_cache(tmp_path: Path) -> None:
    path = (tmp_path / "latents.safetensors").resolve()
    expected = torch.arange(48 * 2 * 4 * 6, dtype=torch.float32).reshape(48, 2, 4, 6).to(torch.bfloat16)
    save_file({"dataset/sample-0": expected, "unrelated": torch.zeros(1, dtype=torch.bfloat16)}, path)
    ref = VaeLatentRef(str(path), "dataset/sample-0", (48, 2, 4, 6), 24.0, "row-0", "dataset")
    reader = SafetensorsLatentReader(handle_cache_size=1, max_tensor_bytes=10_000)

    first, first_metrics = reader.load(ref)
    second, second_metrics = reader.load(ref)

    assert torch.equal(first, expected)
    assert first.dtype == torch.bfloat16 and first.is_contiguous()
    assert first_metrics["cache_hit"] is False
    assert second_metrics["cache_hit"] is True
    assert second_metrics["nbytes"] == ref.expected_nbytes
    assert first.data_ptr() != second.data_ptr()
    reader.close()


def test_reader_rejects_missing_key_shape_and_dtype(tmp_path: Path) -> None:
    bf16_path = (tmp_path / "bf16.safetensors").resolve()
    fp32_path = (tmp_path / "fp32.safetensors").resolve()
    save_file({"key": torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16)}, bf16_path)
    save_file({"key": torch.zeros(48, 2, 4, 6, dtype=torch.float32)}, fp32_path)
    reader = SafetensorsLatentReader(max_tensor_bytes=10_000)

    with pytest.raises(VaeLatentLoadError, match="missing"):
        reader.load(VaeLatentRef(str(bf16_path), "other", (48, 2, 4, 6), 24.0, "row", "dataset"))
    with pytest.raises(VaeLatentLoadError, match="shape mismatch"):
        reader.load(VaeLatentRef(str(bf16_path), "key", (48, 3, 4, 6), 24.0, "row", "dataset"))
    with pytest.raises(VaeLatentLoadError, match="dtype mismatch"):
        reader.load(VaeLatentRef(str(fp32_path), "key", (48, 2, 4, 6), 24.0, "row", "dataset"))
    reader.close()


def test_manifest_preflight_requires_compatible_dataset_lineage(tmp_path: Path) -> None:
    first = _manifest(tmp_path / "first.json", dataset="first")
    second = _manifest(tmp_path / "second.json", dataset="second")

    lineages = validate_offline_vae_manifests(
        [
            {"path": "/first.parquet", "ratio": 1, "vae_manifest_path": str(first)},
            {"path": "/second.parquet", "ratio": 1, "vae_manifest_path": str(second)},
        ],
        expected_vae_name="wan2pt2_tokenizer",
        expected_vae_weights_sha256=_SHA,
    )

    assert [lineage["dataset"] for lineage in lineages] == ["first", "second"]
    _manifest(second, dataset="second", vae_tag="incompatible")
    with pytest.raises(ValueError, match="incompatible"):
        validate_offline_vae_manifests(
            [
                {"path": "/first.parquet", "ratio": 1, "vae_manifest_path": str(first)},
                {"path": "/second.parquet", "ratio": 1, "vae_manifest_path": str(second)},
            ],
            expected_vae_name="wan2pt2_tokenizer",
            expected_vae_weights_sha256=_SHA,
        )


def test_lineage_processor_checks_every_catalog_row_dataset() -> None:
    processor = OfflineVaeLineageProcessor(IdentityProcessor(), expected_dataset="dataset")

    with pytest.raises(CatalogSchemaError, match="manifest declares dataset"):
        processor.process({"id": "row-1", "dataset": "unexpected"})


@pytest.mark.parametrize("cfg_drop_ratio", [0.0, 1.0])
def test_offline_processor_emits_reference_without_video_decode(tmp_path: Path, cfg_drop_ratio: float) -> None:
    path = (tmp_path / "not-opened.safetensors").resolve()
    processor = CatalogVisualGenerationProcessor(
        _Processor(),
        video_decoder=_DecoderMustNotRun(),
        vae_mode="offline",
        conditioning_config={0: 1.0, 1: 0.0, 2: 0.0},
        cfg_drop_ratio=cfg_drop_ratio,
    )

    sample = processor.process(_catalog_row(path))

    assert sample["cfg_dropped"] == bool(cfg_drop_ratio)
    if cfg_drop_ratio:
        assert sample["ai_caption"] == ""
    assert sample["video_latent_ref"].path == str(path)
    assert "video" not in sample and "images" not in sample
    assert sample["num_frames"] == 5
    assert sample["vision_input_mode"] == "offline"
    assert sample["sequence_plan"].condition_frame_indexes_vision == []

    with pytest.raises(CatalogSchemaError, match="exactly one"):
        processor.process(_catalog_row(path, include_pixels=True))
    with pytest.raises(CatalogSchemaError, match="online VAE mode requires"):
        CatalogVisualGenerationProcessor(_Processor(), video_decoder=_DecoderMustNotRun(), vae_mode="online").process(
            _catalog_row(path)
        )


def test_offline_packing_uses_metadata_then_collator_materializes(tmp_path: Path) -> None:
    path = (tmp_path / "latents.safetensors").resolve()
    latent = torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16)
    save_file({"dataset/sample-0": latent}, path)
    ref = VaeLatentRef(str(path), "dataset/sample-0", (48, 2, 4, 6), 24.0, "row-0", "dataset")
    sample = {
        "text_token_ids": torch.tensor([10, 11, 12, 13]),
        "video_latent_ref": ref,
        "conditioning_fps": 24.0,
        "sequence_plan": SequencePlan(has_text=True, has_vision=True),
        "vision_input_mode": "offline",
    }
    batcher = SequentialPackingBatcher(
        max_sequence_length=100,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        lookahead_limit=1,
    )

    assert batcher.sample_size(sample) == 19
    fixed_batcher = SequentialPackingBatcher(
        max_sequence_length=40,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        lookahead_limit=1,
        fixed_sequence_length_enabled=True,
        fixed_und_tokens=20,
        fixed_gen_tokens=20,
        special_tokens=_SPECIAL_TOKENS,
    )
    assert fixed_batcher.sample_token_counts(sample) == SequenceTokenCounts(und=6, gen=12)

    reader = SafetensorsLatentReader(max_tensor_bytes=10_000)
    batch = VFMListCollator(
        static_packing_config=StaticPackingConfig(latent_patch_size=2),
        special_tokens=_SPECIAL_TOKENS,
        latent_reader=reader,
    ).collate([sample])

    assert torch.equal(batch["video_latent"][0][0], latent)
    assert batch["vae_latent_shapes"] == [(48, 2, 4, 6)]
    assert batch["vae_latent_nbytes"].item() == 4_608
    assert batch[PACKED_SEQUENCE_KEY].sequence_length == 18
    reader.close()


@pytest.mark.parametrize("standard_format", [False, True])
def test_offline_catalog_loader_runs_manifest_to_materialized_batch(tmp_path: Path, standard_format: bool) -> None:
    latent_path = (tmp_path / "latents.safetensors").resolve()
    latent = torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16)
    save_file({"dataset/sample-0": latent}, latent_path)
    manifest = _manifest(tmp_path / "MANIFEST.json", dataset="dataset")
    row = _catalog_row(latent_path)
    if standard_format:
        target = row["data"]["messages"][-1]["content"][0]
        target["fps"] = target["vae_latent"].pop("target_fps")
        target.update(gen_video=None, valid_frames=None, width=None, height=None, num_frames=None)
    catalog_path = tmp_path / "catalog.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": row["id"],
                    "type": row["type"],
                    "dataset": row["dataset"],
                    "data": json.dumps(row["data"]),
                    "provenance": json.dumps(row["provenance"]),
                }
            ]
        ),
        catalog_path,
    )

    joint = build_catalog_training_dataloader(
        [{"path": str(catalog_path), "ratio": 1, **({} if standard_format else {"vae_manifest_path": str(manifest)})}],
        processor=_Processor(),
        vae_mode="offline",
        expected_vae_weights_sha256=_SHA,
        max_samples_per_batch=1,
        num_workers=0,
        media_config={"max_latent_tensor_bytes": 10_000},
    )
    batch = next(iter(joint))

    assert torch.equal(batch["video_latent"][0][0], latent)
    assert batch["vision_input_mode"] == ["offline"]
    assert batch["vae_latent_cache_misses"].item() == 1


@pytest.mark.parametrize("fps", [0, -1, True, float("nan"), float("inf"), "24"])
def test_standard_latent_rejects_invalid_fps(tmp_path: Path, fps) -> None:
    row = _catalog_row(tmp_path / "missing.safetensors")
    target = row["data"]["messages"][-1]["content"][0]
    target["fps"] = fps
    target["gen_video"] = None
    target["vae_latent"].pop("target_fps")
    processor = CatalogVisualGenerationProcessor(_Processor(), vae_mode="offline")
    with pytest.raises(CatalogSchemaError, match="finite positive"):
        processor.process(row)


def test_manifestless_offline_run_rejects_understanding_catalog(tmp_path: Path) -> None:
    path = tmp_path / "understanding.parquet"
    row = _catalog_row(tmp_path / "unused.safetensors")
    row["type"] = "understanding"
    row["data"] = json.dumps(row["data"])
    row["provenance"] = json.dumps(row["provenance"])
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match="missing dataflow component"):
        build_catalog_training_dataloader(
            [{"path": str(path), "ratio": 1}],
            processor=_Processor(),
            vae_mode="offline",
            num_workers=0,
        )


@pytest.mark.parametrize("supervise_text", [False, True])
def test_usr_recipe_packs_full_2048_token_text(tmp_path: Path, supervise_text: bool) -> None:
    import tomllib

    recipe = Path(__file__).parents[3] / "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static.toml"
    raw = tomllib.loads(recipe.read_text())
    fixed = raw["model"]["fixed_sequence_length"]
    length = raw["dataloader_train"]["max_caption_tokens"]
    sample = {
        "text_token_ids": torch.arange(length),
        "video_latent_ref": VaeLatentRef(
            str(tmp_path / "unused.safetensors"), "sample", (48, 71, 40, 40), 24.0, "sample", "dataset"
        ),
        "sequence_plan": SequencePlan(
            has_text=True, has_vision=True, supervise_text=supervise_text, text_loss_mask=[supervise_text] * length
        ),
    }
    batcher = SequentialPackingBatcher(
        max_sequence_length=raw["dataloader_train"]["max_sequence_length"],
        fixed_sequence_length_enabled=True,
        fixed_und_tokens=fixed["und_tokens"],
        fixed_gen_tokens=fixed["gen_tokens"],
        special_tokens={**_SPECIAL_TOKENS, "bos_token_id": 1},
    )
    batches = list(batcher.batches(iter([sample, sample])))
    assert [len(batch) for batch in batches] == [1, 1]
    counts = batcher.sample_token_counts(sample)
    assert counts.und == length + (2 if supervise_text else 3)
    assert counts.und <= fixed["und_tokens"] - 1
    assert counts.gen == 28400
