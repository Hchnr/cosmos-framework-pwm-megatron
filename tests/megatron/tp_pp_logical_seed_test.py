# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real CPU RNG regression for logical preparation; no process group or GPU."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, RectifiedFlow


class _Flow:
    sample_train_time = RectifiedFlow.sample_train_time

    def __init__(self):
        self.noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.device, self.dtype = torch.device("cpu"), torch.float32

    @staticmethod
    def train_time_sampler(batch_size, *, device, dtype, generator, shifts):
        return torch.rand(batch_size, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def get_interpolation(epsilon, original, sigmas):
        return epsilon, epsilon


def _model(strategy="base"):
    flow = _Flow()
    return SimpleNamespace(
        config=SimpleNamespace(
            action_gen=True,
            causal_training_strategy=strategy,
            rectified_flow_training_config=SimpleNamespace(
                use_discrete_rf=False, shift=1, shift_action=1, shift_sound=1
            ),
        ),
        rectified_flow_video=flow,
        rectified_flow_image=flow,
        rectified_flow_action=flow,
        rectified_flow_sound=flow,
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        tensor_kwargs={"device": "cpu", "dtype": torch.float32},
        parallel_dims=None,
    )


def _draw(monkeypatch, *, physical_rank, logical_rank, seed_rank, strategy="base"):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: physical_rank)
    model = _model(strategy)
    with torch.random.fork_rng(devices=[]):
        # LogicalTrainingRNG restores the original logical rank's global RNG.
        torch.manual_seed(1000 + logical_rank)
        vision, _ = OmniMoTModel._get_train_noise_level_vision(model, 1, False, [2], iteration=12, seed_rank=seed_rank)
        action, _ = OmniMoTModel._get_train_noise_level_action(model, 1, iteration=12, seed_rank=seed_rank)
        # Sound's native helper deliberately omits iteration and uses global RNG.
        sound, _ = OmniMoTModel._get_train_noise_level_sound(model, 1)
        clean = SimpleNamespace(
            x0_tokens_vision=[torch.zeros(1, 1, 2, 2, 2)],
            x0_tokens_action=[torch.zeros(2, 4)],
            x0_tokens_sound=[torch.zeros(1, 2)],
            is_image_batch=False,
            raw_action_dim=None,
            batch_size=1,
        )
        packed = SimpleNamespace(
            vision=SimpleNamespace(condition_mask=[torch.zeros(2, 1, 1)]),
            action=SimpleNamespace(condition_mask=[torch.zeros(2, 1)], input_valid_mask=None),
            sound=SimpleNamespace(condition_mask=[torch.zeros(2, 1)]),
        )
        noised = OmniMoTModel._add_noise_to_input(
            model, clean, packed, torch.full((1, 1), 0.5), iteration=12, seed_rank=seed_rank
        )
        return (vision, action, sound, noised.epsilon_vision[0], noised.epsilon_action[0], noised.epsilon_sound[0])


@pytest.mark.parametrize("strategy", ["base", "diffusion_forcing"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_original_seed_rank_matches_serial_and_parallel_owners(monkeypatch, strategy, deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    warning_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        for logical_rank in range(8):
            native = _draw(
                monkeypatch, physical_rank=logical_rank, logical_rank=logical_rank, seed_rank=None, strategy=strategy
            )
            serial = _draw(
                monkeypatch, physical_rank=0, logical_rank=logical_rank, seed_rank=logical_rank, strategy=strategy
            )
            parallel = _draw(
                monkeypatch,
                physical_rank=logical_rank % 4,
                logical_rank=logical_rank,
                seed_rank=logical_rank,
                strategy=strategy,
            )
            assert all(torch.equal(a, b) for a, b in zip(native, serial, strict=True))
            assert all(torch.equal(a, b) for a, b in zip(native, parallel, strict=True))
        if deterministic:
            # Negative control: physical rank alone gives a different native stream.
            wrong = _draw(monkeypatch, physical_rank=0, logical_rank=3, seed_rank=None, strategy=strategy)
            right = _draw(monkeypatch, physical_rank=3, logical_rank=3, seed_rank=None, strategy=strategy)
            assert not torch.equal(wrong[0], right[0])
            assert not torch.equal(wrong[3], right[3])
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warning_only)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "2"])
def test_invalid_explicit_seed_rank_fails_closed(invalid):
    flow = _Flow()
    with pytest.raises(ValueError, match="seed_rank"):
        flow.sample_train_time(1, seed_rank=invalid)
    with pytest.raises(ValueError, match="seed_rank"):
        OmniMoTModel._add_noise_to_input(_model(), None, None, None, seed_rank=invalid)
