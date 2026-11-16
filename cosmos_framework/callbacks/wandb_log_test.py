# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.wandb_log import WandbCallback, _LossRecord


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        ({"dataset_name": "AgiBotWorld-Beta:0"}, ["AgiBotWorld-Beta:0"]),
        ({}, ["default"]),
        ({"__url__": ["s3://bucket/data/shard.tar"]}, ["s3://bucket/data"]),
        ({"dataset_name": ["alpha", "alpha", "beta"]}, ["alpha", "beta"]),
        ({"dataset_name": ("alpha", "alpha")}, ["alpha"]),
    ],
)
def test_loss_records_use_whole_dataset_names(batch, expected):
    # Exercise accumulation on CPU, before the distributed logging boundary.
    callback = SimpleNamespace(
        final_loss_log=_LossRecord(),
        final_loss_log_per_dataset={},
        final_all_loss_log={},
        unstable_count=torch.zeros(1),
        config=SimpleNamespace(trainer=SimpleNamespace(logging_iter=1)),
        logging_iter_multipler=2,
    )
    loss = torch.tensor(2.5, requires_grad=True)
    WandbCallback.on_training_step_end(callback, None, batch, {}, loss, iteration=1)
    assert list(callback.final_loss_log_per_dataset) == expected
    for record in callback.final_loss_log_per_dataset.values():
        assert record.iter_count == 1
        assert record.loss.item() == 2.5
        assert not record.loss.requires_grad
    assert callback.final_loss_log.iter_count == 1
    assert callback.final_loss_log.loss.item() == 2.5
