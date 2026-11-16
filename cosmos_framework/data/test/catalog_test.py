# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import itertools
import json
import os
from collections import UserDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.configs.toml_config.sft_config import DataloaderTrainConfig
from cosmos_framework.data.generator.catalog import (
    CATALOG_COLUMNS,
    CatalogSchemaError,
    ParquetBlockDistributor,
    ParquetCatalogDataset,
)
from cosmos_framework.data.generator.catalog_processors import (
    CatalogUnderstandingProcessor,
    CatalogVisualGenerationProcessor,
    DropFailedMediaProcessor,
    build_action_dataset_from_catalog,
    build_catalog_joint_dataloader,
    build_catalog_training_dataloader,
)
from cosmos_framework.data.generator.dataflow.batchers import SimpleBatcher
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.dataflow.loader import CosmosDataLoader, JointCosmosDataLoader
from cosmos_framework.data.generator.dataflow.processors import IdentityProcessor
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY, DROP_SAMPLE_REASON_KEY
from cosmos_framework.data.generator.media_decode import MediaDecodeError
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_understanding_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder

SAMPLES = Path("/share/project/liyue/Project/data-processor/outputs/dataloader_samples")


def test_parquet_toml_dataset_and_media_schema() -> None:
    config = DataloaderTrainConfig.model_validate(
        {
            "datasets": [{"path": "/data/a.parquet", "ratio": 3}],
            "media": {"decode_timeout_sec": 10},
        }
    )
    assert config.datasets[0].ratio == 3
    assert config.media.max_sampled_frames == 256
    assert config.visual_generation.conditioning_config == {0: 0.7, 1: 0.2, 2: 0.1}
    normalized = DataloaderTrainConfig.model_validate(
        {"visual_generation": {"conditioning_config": {"0": 7, "1": 2, "2": 1}}}
    )
    assert normalized.visual_generation.conditioning_config == {0: 0.7, 1: 0.2, 2: 0.1}
    with pytest.raises(ValueError):
        DataloaderTrainConfig.model_validate({"datasets": [{"path": "/data/a.parquet", "ratio": 0}]})
    with pytest.raises(ValueError):
        DataloaderTrainConfig.model_validate({"visual_generation": {"conditioning_config": {"0": 1, "1": 0}}})


@pytest.mark.parametrize(
    ("relative", "task_type", "rows"),
    [
        ("understanding_onevision", "understanding", 8),
        ("visual_generation_action100m", "visual_generation", 8),
        ("action_droid.parquet", "action", 1),
    ],
)
def test_real_five_column_catalogs(relative: str, task_type: str, rows: int) -> None:
    dataset = ParquetCatalogDataset(SAMPLES / relative, expected_type=task_type)
    assert len(dataset) == rows
    assert dataset[0]["type"] == task_type
    for parquet_path in dataset.files:
        assert tuple(pq.ParquetFile(parquet_path).schema_arrow.names) == CATALOG_COLUMNS
        assert "media" not in pq.ParquetFile(parquet_path).schema_arrow.names


def test_action_catalog_requires_exactly_one_row(tmp_path: Path) -> None:
    schema = pa.schema([(name, pa.string()) for name in CATALOG_COLUMNS])
    empty_path = tmp_path / "empty_action.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=schema), empty_path)
    with pytest.raises(CatalogSchemaError, match="catalog is empty"):
        build_action_dataset_from_catalog(empty_path)

    row = {
        "id": "action-dataset",
        "type": "action",
        "dataset": "droid",
        "data": json.dumps({"format": "lerobot", "adapter": "droid", "root": "/unused"}),
        "provenance": "{}",
    }
    multiple_path = tmp_path / "multiple_action.parquet"
    pq.write_table(pa.Table.from_pylist([row, {**row, "id": "action-dataset-copy"}], schema=schema), multiple_path)
    with pytest.raises(CatalogSchemaError, match="exactly one dataset row, got 2"):
        build_action_dataset_from_catalog(multiple_path)


@pytest.mark.parametrize("mode", ["forward_dynamics", "inverse_dynamics"])
def test_action_catalog_keeps_droid_merged_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    sentinel = object()
    captured: dict = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "cosmos_framework.data.generator.catalog_processors.get_action_droid_merged_lerobot_sft_dataset",
        fake_builder,
    )
    row = {
        "id": "droid-fd",
        "type": "action",
        "dataset": "droid_merged",
        "data": json.dumps(
            {
                "format": "lerobot",
                "adapter": "droid_merged",
                "root": "/unused",
                "mode": mode,
                "sample_stride": 2,
            }
        ),
        "provenance": "{}",
    }
    path = tmp_path / "droid_merged.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)

    assert build_action_dataset_from_catalog(path) is sentinel
    assert captured["mode"] == mode
    assert captured["sample_stride"] == 2


def test_footer_index_and_bounded_row_group_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "id": str(index),
            "type": "understanding",
            "dataset": "test",
            "data": json.dumps({"messages": [{"role": "user", "content": str(index)}]}),
            "provenance": "{}",
        }
        for index in range(9)
    ]
    path = tmp_path / "catalog.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    calls: list[int] = []
    original = pq.ParquetFile.read_row_group

    def wrapped(self, row_group, *args, **kwargs):
        calls.append(row_group)
        return original(self, row_group, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", wrapped)
    dataset = ParquetCatalogDataset(path)
    assert calls == []
    assert dataset[7]["id"] == "7"
    assert calls == [2]
    assert dataset[8]["id"] == "8"
    assert calls == [2]
    with pytest.raises(CatalogSchemaError, match="row group has 3 rows"):
        ParquetCatalogDataset(path, max_row_group_rows=2)


def test_block_distribution_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    rows = [
        {
            "id": str(index),
            "type": "understanding",
            "dataset": "test",
            "data": json.dumps({"messages": [{"role": "user", "content": str(index)}]}),
            "provenance": "{}",
        }
        for index in range(20)
    ]
    path = tmp_path / "catalog.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
    distributor = ParquetBlockDistributor(ParquetCatalogDataset(path), seed=17)
    first = [next_item["id"] for next_item in itertools.islice(distributor.stream(0, 2, 0, 1), 10)]
    second = [next_item["id"] for next_item in itertools.islice(distributor.stream(1, 2, 0, 1), 10)]
    repeated = [next_item["id"] for next_item in itertools.islice(distributor.stream(0, 2, 0, 1), 10)]
    assert first == repeated
    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == {str(index) for index in range(20)}


def test_media_errors_are_dropped_and_schema_errors_fail(tmp_path: Path) -> None:
    rows = [
        {
            "id": str(index),
            "type": "understanding",
            "dataset": "same-source",
            "data": json.dumps({"messages": [{"role": "user", "content": str(index)}]}),
            "provenance": "{}",
        }
        for index in range(4)
    ]
    path = tmp_path / "catalog.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    catalog = ParquetCatalogDataset(path)

    class FailsFirst(IdentityProcessor):
        def process(self, item):
            if item["id"] == "0":
                raise MediaDecodeError("bad media")
            return {"served": item["id"]}

    processor = DropFailedMediaProcessor(FailsFirst())
    dropped = processor.process(catalog[0])
    assert dropped[DROP_SAMPLE_KEY] is True
    assert dropped["catalog_id"] == "0"
    assert "bad media" in dropped[DROP_SAMPLE_REASON_KEY]

    loader = CosmosDataLoader(
        distributor=ParquetBlockDistributor(catalog, shuffle=False),
        processor=processor,
        batch_size=1,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["served"] == ["1"]
    assert batch["sample_index"].tolist() == [1]

    class SchemaFailure(IdentityProcessor):
        def process(self, _item):
            raise CatalogSchemaError("bad schema")

    with pytest.raises(CatalogSchemaError, match="bad schema"):
        DropFailedMediaProcessor(SchemaFailure()).process(catalog[0])


def test_catalog_vfm_packing_preserves_resumable_stream_order(tmp_path: Path) -> None:
    row = {
        "id": "video-0",
        "type": "visual_generation",
        "dataset": "test",
        "data": json.dumps(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "move"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "gen_video",
                                "gen_video": "/unused.mp4",
                                "trim_head_frames": 0,
                                "valid_frames": 5,
                                "fps": 5,
                            }
                        ],
                    },
                ]
            }
        ),
        "provenance": "{}",
    }
    path = tmp_path / "visual_generation.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)

    joint = build_catalog_training_dataloader(
        [{"path": str(path), "ratio": 1}],
        processor=_FakeQwenProcessor(),
        max_sequence_length=45_056,
        num_workers=0,
        static_packing_config={
            "spatial_compression_factor": 16,
            "temporal_compression_factor": 4,
            "latent_patch_size": 1,
        },
    )
    batcher = joint._loaders[0].dataset._batcher

    assert batcher.lookahead_limit == 1
    assert batcher.tokenizer_spatial_compression_factor == 16
    assert batcher.tokenizer_temporal_compression_factor == 4
    assert batcher.patch_spatial == 1
    sample = {
        "text_token_ids": torch.arange(4),
        "video": torch.zeros(3, 17, 256, 256),
    }
    assert batcher.sample_size(sample) == 1_287


def test_exact_periodic_ratio_and_resume() -> None:
    loaders = {
        "a": {"dataloader": itertools.repeat({"value": torch.tensor(1)}), "ratio": 3},
        "b": {"dataloader": itertools.repeat({"value": torch.tensor(2)}), "ratio": 2},
    }
    joint = JointCosmosDataLoader(loaders, seed=9, selection_mode="exact_periodic")
    stream = iter(joint)
    names = [next(stream)["dataset_name"] for _ in range(15)]
    for start in range(0, 15, 5):
        assert names[start : start + 5].count("a") == 3
        assert names[start : start + 5].count("b") == 2
    resumed = JointCosmosDataLoader(loaders, seed=9, selection_mode="exact_periodic")
    resumed.set_start_iteration(10)
    resumed_names = [next(iter(resumed))["dataset_name"] for _ in range(5)]
    assert resumed_names == names[10:15]


def test_joint_checkpoint_validates_schedule_and_accepts_legacy_state() -> None:
    loaders = {
        "a": {"dataloader": itertools.repeat({"value": torch.tensor(1)}), "ratio": 3},
        "b": {"dataloader": itertools.repeat({"value": torch.tensor(2)}), "ratio": 2},
    }
    original = JointCosmosDataLoader(loaders, seed=9, selection_mode="exact_periodic")
    callback = JointCosmosDataLoaderStateCallback(original)
    state = callback.state_dict()
    same = JointCosmosDataLoader(loaders, seed=9, selection_mode="exact_periodic")
    JointCosmosDataLoaderStateCallback(same).load_state_dict(state)
    changed = JointCosmosDataLoader(loaders, seed=10, selection_mode="exact_periodic")
    with pytest.raises(ValueError, match="schedule changed"):
        JointCosmosDataLoaderStateCallback(changed).load_state_dict(state)
    # Old checkpoints had only global_id and per-stream worker state.
    JointCosmosDataLoaderStateCallback(changed).load_state_dict({"global_id": 7})
    assert changed._global_id == 7


def test_detached_joint_checkpoint_tracks_and_restores_inner_positions() -> None:
    callback = JointCosmosDataLoaderStateCallback()
    callback.on_training_step_batch_end(
        model=None,
        data_batch={
            "dataset_name": "a:0",
            "sample_worker_id": torch.tensor([0, 0]),
            "sample_epoch": torch.tensor([2, 2]),
            "sample_index": torch.tensor([4, 7]),
        },
        output_batch={},
        loss=torch.tensor(0.0),
    )
    state = callback.state_dict()
    assert state["global_id"] == 1
    assert state["a:0"][0] == {"epoch": 2, "index": 7}
    assert "__schedule__" not in state
    assert "__fingerprints__" not in state

    restored = JointCosmosDataLoaderStateCallback()
    epoch_key = "COSMOS_DL_STATE_a:0_WORKER_0_EPOCH"
    index_key = "COSMOS_DL_STATE_a:0_WORKER_0_INDEX"
    try:
        restored.load_state_dict(state)
        assert restored._global_id == 1
        assert os.environ[epoch_key] == "2"
        assert os.environ[index_key] == "7"
    finally:
        os.environ.pop(epoch_key, None)
        os.environ.pop(index_key, None)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    _ids = {
        "<|im_start|>": 10,
        "<|im_end|>": 14,
        "<|vision_start|>": 11,
        "<|vision_end|>": 15,
        "<|image_pad|>": 13,
        "<|video_pad|>": 16,
        "<think>": 17,
        "</think>": 18,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def apply_chat_template(self, messages, *_args, **kwargs) -> list[int]:
        if not kwargs.get("add_generation_prompt") and messages[-1]["role"] == "assistant":
            return [20, 21, 22, 60, 14, 61]
        return [20, 21, 22]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [17, 18] if text == "<think></think>" else []

    def get_vocab(self) -> dict[str, int]:
        return self._ids


class _FakeQwenProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **_kwargs):
        assert messages[0]["content"][0]["type"] == "image"
        assert not isinstance(messages[0]["content"][0]["image"], str)
        return {
            "input_ids": torch.tensor([10, 11, 12, 13, 14]),
            "attention_mask": torch.ones(5, dtype=torch.bool),
        }

    def add_assistant_tokens_mask(self, _input_ids):
        return torch.tensor([False, False, True, True, True])


def test_understanding_native_chat_and_special_token_loss_mask() -> None:
    catalog = ParquetCatalogDataset(SAMPLES / "understanding_onevision", expected_type="understanding")
    result = CatalogUnderstandingProcessor(_FakeQwenProcessor()).process(catalog[0])
    assert result["text_token_ids"].tolist() == [10, 11, 12, 13, 14]
    assert result["sequence_plan"].text_loss_mask == [False, False, True, False, True]


def test_understanding_samples_pack_as_independent_causal_spans() -> None:
    plans = [
        SequencePlan(has_text=True, text_loss_mask=[False, True, True]),
        SequencePlan(has_text=True, text_loss_mask=[False, False, True, True]),
    ]
    packed = pack_understanding_sequence(
        plans,
        [[10, 11, 12], [20, 21, 22, 23]],
        {"eos_token_id": 2, "start_of_generation": 3},
    )
    assert packed.sample_lens == [3, 4]
    assert packed.split_lens == [3, 4]
    assert packed.attn_modes == ["causal", "causal"]
    assert packed.ce_loss_indexes.tolist() == [0, 1, 4, 5]
    assert packed.label_ids.tolist() == [11, 12, 22, 23]


class _MultiTokenRoleTokenizer:
    all_special_ids = [10, 11, 42]
    _ids = {
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<think>": 40,
        "</think>": 41,
        "<|image_pad|>": 42,
        "<|vision_start|>": 43,
        "<|vision_end|>": 44,
        "<|video_pad|>": 45,
    }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "assistant\n"
        assert add_special_tokens is False
        return [31, 32, 33]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def get_vocab(self) -> dict[str, int]:
        return self._ids


class _ProcessorHolder:
    tokenizer = _MultiTokenRoleTokenizer()


def test_qwen_assistant_mask_matches_multitoken_role_and_excludes_specials() -> None:
    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.processor = _ProcessorHolder()
    tokens = torch.tensor([10, 31, 32, 33, 40, 50, 41, 42, 11])
    assert processor.add_assistant_tokens_mask(tokens).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_toml_specs_build_existing_cosmos_joint_loader() -> None:
    path = str(SAMPLES / "understanding_onevision")
    loader = build_catalog_joint_dataloader(
        [{"path": path, "ratio": 1}],
        components={
            "understanding": {
                "processor": CatalogUnderstandingProcessor(_FakeQwenProcessor()),
                "batcher": SimpleBatcher(1),
                "collator": VFMListCollator(),
            }
        },
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["dataset_name"] == "llava_onevision:0"
    assert batch["text_token_ids"][0][0].shape == (5,)
    assert batch["sequence_plan"][0].text_loss_mask == [False, False, True, False, True]


def test_catalog_rejects_dataset_without_matching_fixed_profile_at_startup() -> None:
    path = str(SAMPLES / "understanding_onevision")

    with pytest.raises(
        ValueError,
        match=r"understanding_onevision.*llava_onevision.*task 'understanding'.*understanding_tokens",
    ):
        build_catalog_joint_dataloader(
            [{"path": path, "ratio": 1}],
            components={
                "understanding": {
                    "processor": CatalogUnderstandingProcessor(_FakeQwenProcessor()),
                    "batcher": SimpleBatcher(1),
                    "collator": VFMListCollator(),
                    "fixed_profile_available": False,
                }
            },
            num_workers=0,
        )


def test_joint_catalog_loader_splits_worker_budget_across_streams() -> None:
    path = str(SAMPLES / "understanding_onevision")
    loader = build_catalog_joint_dataloader(
        [{"path": path, "ratio": 1} for _ in range(3)],
        components={
            "understanding": {
                "processor": CatalogUnderstandingProcessor(_FakeQwenProcessor()),
                "batcher": SimpleBatcher(1),
                "collator": VFMListCollator(),
            }
        },
        num_workers=4,
        prefetch_factor=2,
    )

    assert [stream.num_workers for stream in loader._loaders] == [2, 1, 1]


class _FakeDecoder:
    def decode(self, _path: str, **kwargs):
        count = int(kwargs.get("valid_frames", kwargs.get("uniform_sample_count", 5)))
        frames = tuple(np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(count))
        return frames, {
            "target_fps": float(kwargs.get("target_fps", 24.0)),
            "requested_timestamps_sec": [0.0] * count,
            "timestamps_sec": [0.0] * count,
        }


def _clip_target(path: Path, *, valid_frames: int = 5, **extra) -> dict:
    return {
        "type": "gen_video",
        "gen_video": str(path),
        "trim_head_frames": 0,
        "valid_frames": valid_frames,
        "fps": 24.0,
        **extra,
    }


class _FakeGenerationProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        self.messages = messages
        media = messages[0]["content"][0]
        assert media["type"] == "image"
        assert isinstance(media["image"], Image.Image)
        return UserDict(
            {
                "input_ids": torch.tensor([20, 21, 22]),
                "attention_mask": torch.ones(3, dtype=torch.bool),
                "pixel_values": torch.ones((2, 3), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            }
        )

    def add_assistant_tokens_mask(self, input_ids):
        return torch.zeros_like(input_ids, dtype=torch.bool)


@pytest.mark.parametrize("cfg_drop_ratio", [0.0, 1.0])
def test_generation_user_image_uses_same_vlm_processor_without_target_prefix(
    tmp_path: Path, cfg_drop_ratio: float
) -> None:
    prompt_image = tmp_path / "prompt.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(prompt_image)
    row = {
        "id": "image-prompt",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(prompt_image)},
                        {"type": "text", "text": "continue"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "target.mp4")],
                },
            ]
        },
    }
    chat_processor = _FakeGenerationProcessor()
    result = CatalogVisualGenerationProcessor(
        chat_processor,
        video_decoder=_FakeDecoder(),
        cfg_drop_ratio=cfg_drop_ratio,
        conditioning_config={0: 0, 1: 0, 2: 1},
    ).process(row)
    assert result["text_token_ids"].tolist() == [20, 21, 22]
    assert result["pixel_values"].shape == (2, 3)
    assert result["image_grid_thw"].tolist() == [[1, 1, 2]]
    # User-media VLM context never implicitly becomes a clean DiT target prefix.
    assert result["conditioning_requested_latent_frames"] == 0
    assert result["sequence_plan"].condition_frame_indexes_vision == []

    assert result["ai_caption"] == ("" if cfg_drop_ratio else "continue")
    assert chat_processor.messages[0]["content"][1]["text"] == ("" if cfg_drop_ratio else "continue")


class _FakeVideoGenerationProcessor(_FakeGenerationProcessor):
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        self.messages = messages
        media = messages[0]["content"][0]
        assert media["type"] == "video"
        assert media["video"] and isinstance(media["video"][0], Image.Image)
        return UserDict(
            {
                "input_ids": torch.tensor([20, 21, 22]),
                "attention_mask": torch.ones(3, dtype=torch.bool),
                "pixel_values_videos": torch.ones((2, 3), dtype=torch.float32),
                "video_grid_thw": torch.tensor([[1, 1, 2]]),
            }
        )


@pytest.mark.parametrize("cfg_drop_ratio", [0.0, 1.0])
def test_generation_user_video_is_vlm_context_not_target_prefix(tmp_path: Path, cfg_drop_ratio: float) -> None:
    row = {
        "id": "video-prompt",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(tmp_path / "prompt.mp4"), "num_frames": 5},
                        {"type": "text", "text": "continue"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "target.mp4")],
                },
            ]
        },
    }
    chat_processor = _FakeVideoGenerationProcessor()
    result = CatalogVisualGenerationProcessor(
        chat_processor,
        video_decoder=_FakeDecoder(),
        cfg_drop_ratio=cfg_drop_ratio,
        conditioning_config={0: 0, 1: 0, 2: 1},
    ).process(row)
    assert result["pixel_values_videos"].shape == (2, 3)
    assert result["video_grid_thw"].tolist() == [[1, 1, 2]]
    assert result["conditioning_requested_latent_frames"] == 0
    assert result["sequence_plan"].condition_frame_indexes_vision == []

    assert result["ai_caption"] == ("" if cfg_drop_ratio else "continue")
    assert chat_processor.messages[0]["content"][1]["text"] == ("" if cfg_drop_ratio else "continue")


def test_joint_catalog_loader_mixes_three_tasks_one_to_one_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single joint loader can schedule all three task batch schemas before model dispatch."""

    action_catalog = SAMPLES / "action_droid.parquet"
    action_root = Path(json.loads(pq.read_table(action_catalog, columns=["data"])["data"][0].as_py())["root"])
    if not (action_root / "success/meta/info.json").is_file():
        pytest.skip("official DROID golden subset is not installed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    loader = build_catalog_joint_dataloader(
        [
            {"path": str(SAMPLES / "understanding_onevision"), "ratio": 1},
            {"path": str(SAMPLES / "visual_generation_action100m"), "ratio": 1},
            {"path": str(action_catalog), "ratio": 1},
        ],
        components={
            "understanding": {
                "processor": CatalogUnderstandingProcessor(_FakeQwenProcessor()),
                "batcher": SimpleBatcher(1),
                "collator": VFMListCollator(),
            },
            "visual_generation": {
                "processor": CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_FakeDecoder()),
                "batcher": SimpleBatcher(1),
                "collator": VFMListCollator(),
            },
            "action": {
                "batcher": SimpleBatcher(1),
                "collator": VFMListCollator(),
            },
        },
        seed=1234,
        num_workers=0,
        tokenizer_config=None,
    )

    batches = list(itertools.islice(iter(loader), 6))
    expected_streams = {"llava_onevision:0", "action100m:1", "droid:2"}
    for cycle_start in (0, 3):
        assert {batch["dataset_name"] for batch in batches[cycle_start : cycle_start + 3]} == expected_streams

    by_stream = {batch["dataset_name"]: batch for batch in batches[:3]}
    understanding = by_stream["llava_onevision:0"]
    assert understanding["text_token_ids"][0][0].shape == (5,)
    assert understanding["sequence_plan"][0].text_loss_mask == [False, False, True, False, True]

    generation = by_stream["action100m:1"]
    assert len(generation["video"]) == len(generation["sequence_plan"]) == 1
    assert "action" not in generation

    action = by_stream["droid:2"]
    assert len(action["video"]) == len(action["action"]) == len(action["sequence_plan"]) == 1
    assert action["action"][0][0].shape == (33, 64)


class _UnifiedFakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        add_generation_prompt = bool(kwargs.get("add_generation_prompt"))
        ids = torch.tensor([20, 21, 22] if add_generation_prompt else [10, 50, 14])
        result = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
        media = [
            part
            for message in messages
            for part in message.get("content", [])
            if part.get("type") in {"image", "video"}
        ]
        if media:
            assert not isinstance(media[0][media[0]["type"]], str)
            if media[0]["type"] == "image":
                result["pixel_values"] = torch.ones((2, 3), dtype=torch.float32)
                result["image_grid_thw"] = torch.tensor([[1, 1, 2]])
            else:
                result["pixel_values_videos"] = torch.ones((2, 3), dtype=torch.float32)
                result["video_grid_thw"] = torch.tensor([[1, 1, 2]])
        return result

    def add_assistant_tokens_mask(self, input_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if input_ids.numel() and int(input_ids[-1]) == 14:
            mask[-2:] = True
        return mask

    def tokenize_text(self, *_args, **_kwargs):
        return [20, 21, 22]


def test_toml_builds_and_iterates_three_task_catalog_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator import catalog_processors as catalog_processor_module
    from cosmos_framework.data.generator import processors as processor_module
    from cosmos_framework.utils.lazy_config import instantiate

    action_catalog = SAMPLES / "action_droid.parquet"
    action_root = Path(json.loads(pq.read_table(action_catalog, columns=["data"])["data"][0].as_py())["root"])
    if not (action_root / "success/meta/info.json").is_file():
        pytest.skip("official DROID golden subset is not installed")

    visual_catalog = tmp_path / "visual.parquet"
    visual_rows = []
    for requested in (0, 1, 2, None):
        target = _clip_target(tmp_path / f"target-{requested}.mp4", valid_frames=9)
        if requested is not None:
            target["conditioning_latent_frames"] = requested
        visual_rows.append(
            {
                "id": f"visual-{requested}",
                "type": "visual_generation",
                "dataset": "visual",
                "data": json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                            {"role": "assistant", "content": [target]},
                        ]
                    }
                ),
                "provenance": "{}",
            }
        )
    pq.write_table(pa.Table.from_pylist(visual_rows), visual_catalog, row_group_size=4)

    recipe = tmp_path / "catalog.toml"
    recipe.write_text(
        f'''[job]
task = "vfm"
experiment = "vision_sft_nano"
project = "cosmos3"
group = "test"
name = "catalog-loader-test"
wandb_mode = "disabled"

[model.tokenizer]
vae_path = "${{oc.env:WAN_VAE_PATH}}"

[checkpoint]
load_path = "${{oc.env:BASE_CHECKPOINT_PATH}}"

[dataloader_train]
seed = 1234
max_sequence_length = 45056
max_samples_per_batch = 1

[dataloader_train.visual_generation]
conditioning_config = {{"0" = 0.7, "1" = 0.2, "2" = 0.1}}

[[dataloader_train.datasets]]
path = "{SAMPLES / "understanding_onevision"}"
ratio = 1

[[dataloader_train.datasets]]
path = "{visual_catalog}"
ratio = 1

[[dataloader_train.datasets]]
path = "{action_catalog}"
ratio = 1
'''
    )
    monkeypatch.setenv("DATASET_PATH", str(tmp_path))
    monkeypatch.setenv("WAN_VAE_PATH", str(tmp_path / "vae.pth"))
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(processor_module, "build_processor", lambda *_args, **_kwargs: _UnifiedFakeProcessor())
    monkeypatch.setattr(
        catalog_processor_module,
        "build_video_decoder_from_config",
        lambda _config=None: _RequestedFrameDecoder(),
    )

    config = load_experiment_from_toml(recipe)
    loader = instantiate(config.dataloader_train)
    assert isinstance(loader, JointCosmosDataLoader)
    batches = list(itertools.islice(iter(loader), 12))
    expected_streams = {"llava_onevision:0", "visual:1", "droid:2"}
    for cycle_start in range(0, 12, 3):
        assert {batch["dataset_name"] for batch in batches[cycle_start : cycle_start + 3]} == expected_streams
    assert all("packed_sequence" in batch for batch in batches)
    assert all(
        batch["packed_sequence"].vision is None for batch in batches if batch["dataset_name"] == "llava_onevision:0"
    )
    assert all(
        batch["packed_sequence"].vision is not None
        for batch in batches
        if batch["dataset_name"] in {"visual:1", "droid:2"}
    )

    visual_batches = [batch for batch in batches if batch["dataset_name"] == "visual:1"]
    requested = {int(batch["conditioning_requested_latent_frames"][0].item()) for batch in visual_batches}
    assert {0, 1, 2}.issubset(requested)
    assert all(len(batch["sequence_plan"]) == 1 for batch in visual_batches)
    assert any(batch["dataset_name"] == "llava_onevision:0" and "sequence_plan" in batch for batch in batches)
    assert any(batch["dataset_name"] == "droid:2" and "action" in batch for batch in batches)


def test_t2v_has_assistant_prefix_and_no_text_ce(tmp_path: Path) -> None:
    row = {
        "id": "t2v",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "unused.mp4")],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_FakeDecoder()).process(row)
    assert result["sequence_plan"].supervise_text is False
    assert result["sequence_plan"].has_vision is True
    builder = PackedSequenceBuilder()
    builder.pack_text_tokens(
        result["text_token_ids"].tolist(),
        {"eos_token_id": 2, "start_of_generation": 11},
        has_generation=True,
        supervise_text=False,
    )
    assert builder.text_ids == result["text_token_ids"].tolist() + [2, 11]
    assert builder.ce_loss_indexes == []
    assert builder.label_ids == []


class _RequestedFrameDecoder:
    def __init__(self) -> None:
        self.last_kwargs = None

    def decode(self, _path: str, **kwargs):
        self.last_kwargs = kwargs
        count = int(kwargs["valid_frames"])
        frames = tuple(np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(count))
        return frames, {
            "target_fps": float(kwargs["target_fps"]),
            "requested_timestamps_sec": [0.0] * count,
            "timestamps_sec": [0.0] * count,
        }


def test_generation_strictly_applies_catalog_trim_head_frames(tmp_path: Path) -> None:
    decoder = _RequestedFrameDecoder()
    row = {
        "id": "strict-trim",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "unused.mp4", trim_head_frames=30)],
                },
            ]
        },
        "provenance": {"extra": {"trim_head_frames_affects_sampling": False}},
    }

    CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=decoder).process(row)

    assert decoder.last_kwargs["trim_head_frames"] == 30


def test_video_window_is_center_cropped_to_vae_temporal_shape(tmp_path: Path) -> None:
    decoder = _RequestedFrameDecoder()
    row = {
        "id": "temporal-center-crop",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "unused.mp4", valid_frames=12)],
                },
            ]
        },
    }

    result = CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=decoder).process(row)

    assert decoder.last_kwargs["valid_frames"] == 12
    assert result["num_frames"] == 9
    assert result["decode_metadata"]["temporal_center_crop"] == {
        "input_frames": 12,
        "output_frames": 9,
        "removed_head_frames": 1,
        "removed_tail_frames": 2,
        "temporal_compression_factor": 4,
    }


@pytest.mark.parametrize(("requested", "indexes"), [(0, []), (1, [0]), (2, [0, 1])])
def test_t2v_explicit_conditioning_latent_frames(tmp_path: Path, requested: int, indexes: list[int]) -> None:
    row = {
        "id": f"t2v-{requested}",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [
                        _clip_target(
                            tmp_path / "unused.mp4",
                            valid_frames=9,
                            conditioning_latent_frames=requested,
                        )
                    ],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_RequestedFrameDecoder()).process(row)
    assert result["conditioning_requested_latent_frames"] == requested
    assert result["conditioning_effective_latent_frames"] == requested
    assert result["sequence_plan"].condition_frame_indexes_vision == indexes


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"conditioning": {"kind": "target_prefix", "num_latent_frames": 1}},
        {"conditioning_latent_frames": 3},
        {"conditioning_latent_frames": True},
    ],
)
def test_t2v_rejects_legacy_or_invalid_conditioning(tmp_path: Path, invalid_fields: dict) -> None:
    target = _clip_target(tmp_path / "unused.mp4", valid_frames=9, **invalid_fields)
    row = {
        "id": "invalid-conditioning",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {"role": "assistant", "content": [target]},
            ]
        },
    }
    with pytest.raises(CatalogSchemaError):
        CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_RequestedFrameDecoder()).process(row)


def test_t2v_short_target_downgrades_conditioning(tmp_path: Path) -> None:
    row = {
        "id": "short",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "unused.mp4", conditioning_latent_frames=2)],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_RequestedFrameDecoder()).process(row)
    assert result["conditioning_requested_latent_frames"] == 2
    assert result["conditioning_effective_latent_frames"] == 1
    assert result["sequence_plan"].condition_frame_indexes_vision == [0]


def test_missing_conditioning_is_epoch_scoped_and_reproducible(tmp_path: Path) -> None:
    row = {
        "id": "epoch-mixture",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [
                        _clip_target(
                            tmp_path / "unused.mp4",
                            valid_frames=9,
                            conditioning_latent_frames=None,
                        )
                    ],
                },
            ]
        },
    }
    processor = CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_RequestedFrameDecoder(), seed=1234)
    first = [
        processor.process({**row, "_dp_epoch": epoch})["conditioning_requested_latent_frames"] for epoch in range(20)
    ]
    second = [
        processor.process({**row, "_dp_epoch": epoch})["conditioning_requested_latent_frames"] for epoch in range(20)
    ]
    assert first == second
    assert set(first).issubset({0, 1, 2})
    assert len(set(first)) > 1


def test_generation_assistant_text_supervises_text_and_im_end_before_vision(tmp_path: Path) -> None:
    row = {
        "id": "t2v-reasoning",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "first plan the motion"},
                        _clip_target(tmp_path / "unused.mp4"),
                    ],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(_FakeTokenizer(), video_decoder=_FakeDecoder()).process(row)
    plan = result["sequence_plan"]
    text_ids = result["text_token_ids"].tolist()
    assert plan.supervise_text is True
    assert plan.text_loss_mask == [False, False, False, True, True]
    assert text_ids[-1] == 14

    builder = PackedSequenceBuilder()
    builder.pack_text_tokens(
        text_ids,
        {"eos_token_id": 14, "start_of_generation": 11},
        has_generation=True,
        supervise_text=True,
        text_loss_mask=plan.text_loss_mask,
    )
    assert builder.text_ids == text_ids + [11]
    assert builder.label_ids == [60, 14]
    assert 11 not in builder.label_ids


def test_t2i_uses_the_same_optional_text_grammar(tmp_path: Path) -> None:
    image_path = tmp_path / "target.png"
    Image.new("RGB", (48, 32), color=(10, 20, 30)).save(image_path)
    row = {
        "id": "t2i",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make an image"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "compose the scene"},
                        {"type": "gen_image", "gen_image": str(image_path)},
                    ],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(_FakeTokenizer()).process(row)
    assert result["images"].ndim == 3
    assert "video" not in result
    assert result["sequence_plan"].supervise_text is True
    assert result["text_token_ids"][-1].item() == 14


def test_edge_qwen3_template_closes_assistant_before_vision_start(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer_root = Path("/share/project/liyue/model_ckpts/Cosmos3-Edge")
    if not (tokenizer_root / "tokenizer.json").is_file():
        pytest.skip("Cosmos3-Edge tokenizer is not installed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    row = {
        "id": "edge-t2v",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [_clip_target(tmp_path / "unused.mp4")],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(tokenizer, video_decoder=_FakeDecoder()).process(row)
    text_ids = result["text_token_ids"].tolist()
    assert tokenizer.decode(text_ids).endswith("<|im_start|>assistant\n")
    builder = PackedSequenceBuilder()
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    builder.pack_text_tokens(
        text_ids,
        {
            "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "start_of_generation": vision_start,
        },
        has_generation=True,
        supervise_text=False,
    )
    assert tokenizer.decode(builder.text_ids).endswith("<|im_start|>assistant\n<|im_end|><|vision_start|>")
    assert builder.label_ids == []


def test_action_caption_closes_assistant_before_vision_without_text_ce() -> None:
    builder = PackedSequenceBuilder()
    builder.pack_text_tokens(
        [20, 21, 22],
        {"eos_token_id": 14, "start_of_generation": 11},
        has_generation=True,
        supervise_text=False,
        text_loss_mask=None,
    )
    assert builder.text_ids == [20, 21, 22, 14, 11]
    assert builder.label_ids == []
    assert builder.ce_loss_indexes == []


def test_edge_qwen3_generation_reasoning_trains_im_end_not_vision_start(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer_root = Path("/share/project/liyue/model_ckpts/Cosmos3-Edge")
    if not (tokenizer_root / "tokenizer.json").is_file():
        pytest.skip("Cosmos3-Edge tokenizer is not installed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    row = {
        "id": "edge-reasoning-t2v",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "instruction"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "reasoning / plan / description"},
                        _clip_target(tmp_path / "unused.mp4"),
                    ],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(tokenizer, video_decoder=_FakeDecoder()).process(row)
    plan = result["sequence_plan"]
    builder = PackedSequenceBuilder()
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    builder.pack_text_tokens(
        result["text_token_ids"].tolist(),
        {"eos_token_id": im_end, "start_of_generation": vision_start},
        has_generation=True,
        supervise_text=plan.supervise_text,
        text_loss_mask=plan.text_loss_mask,
    )
    assert tokenizer.decode(builder.text_ids).endswith(
        "<|im_start|>assistant\nreasoning / plan / description<|im_end|><|vision_start|>"
    )
    assert tokenizer.decode(builder.label_ids) == "reasoning / plan / description<|im_end|>"
    assert vision_start not in builder.label_ids


def test_official_droid_catalog_builds_two_windows_and_collates(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_path = SAMPLES / "action_droid.parquet"
    root = Path(json.loads(pq.read_table(catalog_path)["data"][0].as_py())["root"])
    if not (root / "success/meta/info.json").is_file():
        pytest.skip("official DROID golden subset is not installed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    dataset = build_action_dataset_from_catalog(catalog_path, tokenizer_config=None)
    assert dataset._dataset._sample_stride == 1
    first, second = dataset[0], dataset[1]
    assert first["video"].shape[1] == 33
    assert first["action"].shape == second["action"].shape == (33, 64)
    assert first["sequence_plan"].supervise_text is False
    batch = VFMListCollator().collate([first, second])
    assert len(batch["video"]) == len(batch["action"]) == 2


@pytest.mark.parametrize("prefix", [0, 1, 2])
@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_cfg_dropout_preserves_target_prefix(tmp_path: Path, prefix: int, ratio: float) -> None:
    row = {
        "id": "cfg",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "make a video"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "a detailed explanation"},
                        _clip_target(tmp_path / "target.mp4", valid_frames=13, conditioning_latent_frames=prefix),
                    ],
                },
            ]
        },
    }
    result = CatalogVisualGenerationProcessor(
        _FakeTokenizer(),
        video_decoder=_RequestedFrameDecoder(),
        cfg_drop_ratio=ratio,
    ).process(row)
    assert result["cfg_dropped"] == bool(ratio)
    assert result["ai_caption"] == ("" if ratio else "make a video")
    assert result["sequence_plan"].condition_frame_indexes_vision == list(range(prefix))
    assert result["sequence_plan"].supervise_text == (not bool(ratio))
    assert row["data"]["messages"][0]["content"][0]["text"] == "make a video"


def test_cfg_dropout_reproducible_across_epoch_and_order(tmp_path: Path) -> None:
    row = {
        "id": "cfg",
        "type": "visual_generation",
        "dataset": "test",
        "data": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "caption"}]},
                {"role": "assistant", "content": [_clip_target(tmp_path / "target.mp4")]},
            ]
        },
    }
    processor = CatalogVisualGenerationProcessor(
        _FakeTokenizer(),
        video_decoder=_FakeDecoder(),
        cfg_drop_ratio=0.1,
    )
    first = {epoch: processor.process({**row, "_dp_epoch": epoch})["cfg_dropped"] for epoch in range(100)}
    restored = CatalogVisualGenerationProcessor(
        _FakeTokenizer(),
        video_decoder=_FakeDecoder(),
        cfg_drop_ratio=0.1,
    )
    assert first == {
        epoch: restored.process({**row, "_dp_epoch": epoch})["cfg_dropped"] for epoch in reversed(range(100))
    }
    assert 1 < sum(first.values()) < 25


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_cfg_dropout_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError):
        DataloaderTrainConfig.model_validate({"visual_generation": {"cfg_drop_ratio": ratio}})
    with pytest.raises(ValueError):
        CatalogVisualGenerationProcessor(_FakeTokenizer(), cfg_drop_ratio=ratio)
