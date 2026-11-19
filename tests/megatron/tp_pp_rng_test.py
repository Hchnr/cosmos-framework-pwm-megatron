# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Logical noising RNG follows native draws across interleaving and restart."""

import random

import numpy as np
import pytest
import torch

from cosmos_framework.training_backends.megatron_tp_pp.rng import LogicalTrainingRNG
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, preserve_training_rng, set_rand_state_dict


def draws():
    return [random.random(), float(np.random.random()), torch.rand(3), torch.randn(5, device="cuda").cpu()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native training RNG includes the CUDA Philox state")
def test_logical_rng_matches_native_interleaving_loader_seed_draws_and_restart():
    with preserve_training_rng():
        initial = {}
        for rank, seed in ((1, 31), (3, 73)):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            initial[rank] = get_rand_state_dict()
        expected, base_seeds = {}, {}
        for rank, state in initial.items():
            set_rand_state_dict(state)
            base_seeds[rank] = [int(torch.empty((), dtype=torch.int64).random_().item()) for _ in range(58)]
            expected[rank] = [draws() for _ in range(3)]
        outside = get_rand_state_dict()
        manager = LogicalTrainingRNG(initial)
        for rank in (3, 1):
            assert manager.draw_loader_base_seeds(rank, 58) == base_seeds[rank]
        counts = {1: 0, 3: 0}
        for rank in (3, 1, 1, 3):
            with manager.activate(rank):
                actual = draws()
            for value, target in zip(actual, expected[rank][counts[rank]], strict=True):
                torch.testing.assert_close(value, target, rtol=0, atol=0)
            counts[rank] += 1
        restarted = LogicalTrainingRNG(manager.state_dict())
        for rank in (1, 3):
            with restarted.activate(rank):
                actual = draws()
            for value, target in zip(actual, expected[rank][2], strict=True):
                torch.testing.assert_close(value, target, rtol=0, atol=0)
        for name, value in get_rand_state_dict().items():
            torch.testing.assert_close(value, outside[name], rtol=0, atol=0)
