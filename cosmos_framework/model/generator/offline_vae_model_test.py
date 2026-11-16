# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def _model() -> OmniMoTModel:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.input_video_latent_key = "video_latent"
    model.config = SimpleNamespace(state_ch=48)
    return model


def test_direct_offline_latents_match_online_model_shape_and_dtype() -> None:
    latent = torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16)

    result = _model()._direct_vision_latents([[latent]])

    assert len(result) == 1
    assert result[0].shape == (1, 48, 2, 4, 6)
    assert result[0].dtype == torch.float32
    assert result[0].is_contiguous()


def test_get_data_and_condition_bypasses_pixel_and_vae_processing() -> None:
    model = _model()
    model.config.vision_vae_mode = "offline"
    model.input_video_key = "video"
    model.input_image_key = "images"
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.tokenizer_sound_gen = None
    model._normalize_video_databatch_inplace = mock.Mock(side_effect=AssertionError("pixel normalization ran"))
    model._encode_vision_x0_tokens = mock.Mock(side_effect=AssertionError("VAE encoding ran"))
    model._normalize_action_databatch = mock.Mock(return_value=(None, None))
    model._normalize_sound_databatch_inplace = mock.Mock()
    latent = torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16)

    result = model.get_data_and_condition(
        {
            "video_latent": [[latent]],
            "conditioning_fps": [torch.tensor(24.0)],
        }
    )

    assert result.raw_state_vision is None
    assert result.x0_tokens_vision[0].shape == (1, 48, 2, 4, 6)
    assert result.x0_tokens_vision[0].dtype == torch.float32
    model._normalize_video_databatch_inplace.assert_not_called()
    model._encode_vision_x0_tokens.assert_not_called()


@pytest.mark.parametrize(
    "latent",
    [
        torch.zeros(48, 2, 4, 6, dtype=torch.float32),
        torch.zeros(16, 2, 4, 6, dtype=torch.bfloat16),
        torch.zeros(2, 48, 2, 4, 6, dtype=torch.bfloat16),
    ],
)
def test_direct_offline_latents_reject_invalid_contract(latent: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError)):
        _model()._direct_vision_latents([[latent]])


def test_offline_vae_resolutions_are_derived_from_latent_shapes() -> None:
    latents = [
        torch.empty(1, 48, 2, 16, 28, device="meta"),
        torch.empty(1, 48, 2, 30, 52, device="meta"),
        torch.empty(1, 48, 2, 45, 80, device="meta"),
    ]

    resolutions = OmniMoTModel._get_offline_vae_resolutions(latents, spatial_downsample_factor=16)

    assert resolutions == ["256", "480", "720"]


def test_offline_vae_resolutions_reject_non_model_latent_shape() -> None:
    latent = torch.empty(48, 2, 30, 52, device="meta")

    with pytest.raises(ValueError, match=r"shape \[1,C,T,H,W\]"):
        OmniMoTModel._get_offline_vae_resolutions([latent], spatial_downsample_factor=16)
