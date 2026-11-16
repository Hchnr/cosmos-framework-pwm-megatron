# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_framework.inference.args import OmniSampleOverrides
from cosmos_framework.inference.catalog import InferenceCatalogError, _nearest_video_bucket, load_inference_inputs


def _write_catalog(
    path: Path, *, task_type: str = "visual_generation", conditioning_latent_frames: int | None = None
) -> None:
    rows = []
    for index in range(2):
        rows.append(
            {
                "id": f"eval:{index:03d}",
                "type": task_type,
                "dataset": "eval",
                "data": json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": json.dumps({"caption": f"prompt {index}"})}],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "gen_video",
                                        "gen_video": "condition.mp4" if conditioning_latent_frames else None,
                                        "fps": 16.0,
                                        "valid_frames": 81,
                                        "width": 352,
                                        "height": 192,
                                        **(
                                            {"conditioning_latent_frames": conditioning_latent_frames}
                                            if conditioning_latent_frames is not None
                                            else {}
                                        ),
                                    }
                                ],
                            },
                        ]
                    }
                ),
                "provenance": json.dumps(
                    {"extra": {"eval_index": index, "eval_mode": "interactive", "seed": 100 + index}}
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_load_t2v_catalog_directory(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "parquet"
    _write_catalog(catalog_dir / "part-00000.parquet")

    samples = load_inference_inputs([catalog_dir], global_overrides=OmniSampleOverrides(num_steps=50, guidance=5.0))

    assert [sample.name for sample in samples] == ["001_interactive", "002_interactive"]
    assert [sample.seed for sample in samples] == [100, 101]
    assert [sample.prompt for sample in samples] == ["prompt 0", "prompt 1"]
    assert all(sample.model_mode == "text2video" for sample in samples)
    assert all(sample.fps == 16 and sample.num_frames == 81 for sample in samples)
    assert all(sample.resolution == "256" and sample.aspect_ratio == "16,9" for sample in samples)
    assert all(sample.num_steps == 50 and sample.guidance == 5.0 for sample in samples)
    assert samples[0].extra == {
        "catalog": {
            "id": "eval:000",
            "dataset": "eval",
            "source": str(catalog_dir / "part-00000.parquet"),
            "source_width": 352,
            "source_height": 192,
            "output_width": 320,
            "output_height": 192,
            "conditioning_latent_frames": 0,
        }
    }


def test_nearest_video_bucket_preserves_source_aspect_ratio() -> None:
    assert _nearest_video_bucket(640, 480) == ("480", "4,3", (736, 544))


@pytest.mark.parametrize(
    ("conditioning_latent_frames", "model_mode", "condition_indexes"),
    [(1, "image2video", [0]), (2, "video2video", [0, 1])],
)
def test_load_clean_prefix_catalog(
    tmp_path: Path, conditioning_latent_frames: int, model_mode: str, condition_indexes: list[int]
) -> None:
    path = tmp_path / "eval.parquet"
    (tmp_path / "condition.mp4").touch()
    _write_catalog(path, conditioning_latent_frames=conditioning_latent_frames)

    samples = load_inference_inputs([path])

    assert [sample.model_mode for sample in samples] == [model_mode, model_mode]
    assert [sample.vision_path for sample in samples] == [str(tmp_path / "condition.mp4")] * 2
    assert [sample.condition_frame_indexes_vision for sample in samples] == [condition_indexes, condition_indexes]
    assert [sample.condition_video_keep for sample in samples] == ["first", "first"]


def test_catalog_rejects_non_visual_generation_rows(tmp_path: Path) -> None:
    path = tmp_path / "action.parquet"
    _write_catalog(path, task_type="action")

    with pytest.raises(InferenceCatalogError, match="only visual_generation video rows are supported"):
        load_inference_inputs([path])


def test_regular_json_inputs_still_use_existing_loader(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"model_mode": "text2video", "prompt": "plain prompt"}))

    samples = load_inference_inputs([path])

    assert len(samples) == 1
    assert samples[0].name == "sample"
    assert samples[0].prompt == "plain prompt"
