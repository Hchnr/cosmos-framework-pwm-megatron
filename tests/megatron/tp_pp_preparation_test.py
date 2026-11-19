# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Owner preparation accepts the native empty-memory/per-sample VLM contract."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationLoss, GenerationPreparation


def test_owner_preparation_reuses_native_helper_without_trainable_graph():
    owner = GenerationPreparation.__new__(GenerationPreparation)
    nn.Module.__init__(owner)
    sequence = object()
    prepared = {
        "packed_sequence": sequence,
        "gen_data_clean": SimpleNamespace(batch_size=2, is_image_batch=False),
        "gen_data_noised": object(),
        "memory_info": {"skip_text": False, "initial_temporal_offset": 0},
        "vlm_inputs": [{}, {}],
        "timesteps_vision": torch.tensor([0.3, 0.8]),
        "timesteps_action": None,
        "timesteps_sound": None,
    }

    def native_helper(batch, iteration):
        assert batch == {"identity": 12} and iteration == 1201
        assert not torch.is_grad_enabled()
        return prepared

    owner.prepare_generation_training_inputs = native_helper
    actual = owner.prepare({"identity": 12}, 1201)
    assert actual["packed_sequence"] is sequence and actual["sample_count"] == 2
    assert actual["loss_inputs"]["gen_data_noised"] is prepared["gen_data_noised"]
    prepared["vlm_inputs"] = [{"pixels": torch.ones(1)}, {}]
    with pytest.raises(ValueError, match="visual reasoner"):
        owner.prepare({"identity": 12}, 1201)


def test_pipeline_loss_cannot_fall_back_to_physical_world_count_collective():
    loss = GenerationLoss.__new__(GenerationLoss)
    nn.Module.__init__(loss)
    loss.config = SimpleNamespace(rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=True))
    with pytest.raises(ValueError, match="logical micro-step"):
        loss({}, {})


@pytest.mark.parametrize("local_rank", [0, 1])
def test_vae_load_and_sync_use_owner_group_even_when_global_zero_is_absent(monkeypatch, local_rank):
    from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan

    group, loaded, synced = object(), [], []
    monkeypatch.setattr(wan, "WanVAE_", lambda **kwargs: nn.Linear(2, 2))
    monkeypatch.setattr(wan, "TRAINING", True)
    monkeypatch.setattr(wan, "INTERNAL", True)
    monkeypatch.setattr(wan, "get_rank", lambda supplied: local_rank if supplied is group else 99)
    monkeypatch.setattr(
        wan.dist, "get_global_rank", lambda supplied, rank: 4 if supplied is group and rank == 0 else -1
    )

    def load(path, **kwargs):
        loaded.append(path)
        return {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}

    monkeypatch.setattr(wan.easy_io, "load", load)
    monkeypatch.setattr(wan, "sync_model_states", lambda model, **kwargs: synced.append(kwargs))
    model = wan._video_vae("/readonly/vae.pth", device="cpu", process_group=group)
    assert loaded == (["/readonly/vae.pth"] if local_rank == 0 else [])
    assert synced == [{"process_group": group, "src": 4}]
    assert not next(model.parameters()).is_meta
