# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest import mock

import torch

from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample


def test_draw_sample_callback_is_fully_skipped_in_offline_vae_mode() -> None:
    callback = EveryNDrawSample(every_n=1)
    model = SimpleNamespace(config=SimpleNamespace(vision_vae_mode="offline"))

    callback.on_train_start(model)
    callback.trainer = mock.Mock()
    callback.on_training_step_end(
        model,
        {"video_latent": [[torch.zeros(48, 2, 4, 6, dtype=torch.bfloat16)]]},
        {},
        torch.tensor(0.0),
        iteration=1,
    )

    assert callback._offline_vae_disabled is True
    assert callback.trainer.mock_calls == []
