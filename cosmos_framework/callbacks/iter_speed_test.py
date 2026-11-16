# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.callbacks.iter_speed as iter_speed_module
from cosmos_framework.callbacks.iter_speed import IterSpeed, _vfm_batch_size_metrics


def _model() -> SimpleNamespace:
    def is_image_batch(data_batch: dict) -> bool:
        if "images" in data_batch or "video" in data_batch:
            return "images" in data_batch
        vlm_inputs = data_batch.get("vlm_inputs", ())
        has_images = any(item.get("pixel_values") is not None for item in vlm_inputs)
        has_videos = any(item.get("pixel_values_videos") is not None for item in vlm_inputs)
        return has_images and not has_videos

    return SimpleNamespace(
        input_image_key="images",
        input_video_key="video",
        input_video_latent_key="video_latent",
        is_image_batch=is_image_batch,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("data_batch", "expected"),
    [
        ({"images": [torch.zeros(1), torch.zeros(1)]}, {"image_batch_size": 2}),
        ({"video": [torch.zeros(1), torch.zeros(1), torch.zeros(1)]}, {"video_batch_size": 3}),
        ({"video_latent": [torch.zeros(1), torch.zeros(1)]}, {"video_batch_size": 2}),
        ({"vlm_inputs": [{"pixel_values": torch.zeros(4, 8)}]}, {"image_batch_size": 1}),
        ({"vlm_inputs": [{"pixel_values_videos": torch.zeros(4, 8)}]}, {"video_batch_size": 1}),
        ({"input_ids": torch.ones(1, 8, dtype=torch.long)}, {}),
    ],
)
def test_vfm_batch_size_metrics_supports_generation_and_understanding_batches(
    data_batch: dict, expected: dict[str, int]
) -> None:
    assert _vfm_batch_size_metrics(_model(), data_batch) == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_offline_vae_batch_uses_latent_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = IterSpeed(every_n=1, hit_thres=0, save_s3=False)
    callback.time = 1.0
    trainer = SimpleNamespace(sample_counter=52)
    model = SimpleNamespace(
        input_image_key="images",
        input_video_key="video",
        input_video_latent_key="video_latent",
        is_image_batch=lambda data_batch: False,
    )
    logged: list[dict[str, float | int]] = []

    monkeypatch.setattr(iter_speed_module.time, "time", lambda: 2.0)
    monkeypatch.setattr(iter_speed_module.wandb, "run", object())
    monkeypatch.setattr(iter_speed_module.wandb, "log", lambda values, step: logged.append(values | {"step": step}))

    IterSpeed.every_n_impl.__wrapped__(
        callback,
        trainer,
        model,
        {"video_latent": [torch.empty(1), torch.empty(1)]},
        {},
        torch.tensor(0.25),
        iteration=52,
    )

    assert logged == [{"timer/iter_speed": 1.0, "sample_counter": 52, "video_batch_size": 2, "step": 52}]
