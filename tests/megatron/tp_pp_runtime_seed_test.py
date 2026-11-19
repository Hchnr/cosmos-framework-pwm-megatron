# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TP/PP must seed host RNGs before initializing MCore's CUDA streams."""

import random
from types import SimpleNamespace

import numpy as np
import torch

from cosmos_framework.training_backends.megatron_tp_pp import runtime
from cosmos_framework.utils import misc


def test_initialize_seeds_host_rng_by_rank_before_model_parallel_cuda_seed(monkeypatch):
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import random as tp_random

    rank = [0]
    calls = []
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _: "0.20.0")
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(runtime.dist, "get_process_group_ranks", lambda group: [0, 1, 2, 3])
    monkeypatch.setattr(misc.distributed, "get_rank", lambda: rank[0])
    monkeypatch.setattr(ps, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(ps, "initialize_model_parallel", lambda **kwargs: calls.append("groups"))
    for name in ("get_tensor_model_parallel_group", "get_pipeline_model_parallel_group", "get_data_parallel_group"):
        monkeypatch.setattr(ps, name, lambda: object())
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(("cuda_all", seed)))
    monkeypatch.setattr(tp_random, "model_parallel_cuda_manual_seed", lambda seed: calls.append(("mcore", seed)))
    config = SimpleNamespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        data_parallel_size=lambda world: 1,
        microbatches=lambda world: 3,
    )

    def run_once(physical_rank):
        rank[0] = physical_rank
        calls.clear()
        runtime.initialize(config, seed=42)
        assert calls == ["groups", ("cuda_all", 42 + physical_rank), ("mcore", 42)]
        return random.random(), np.random.random(), torch.rand(1).item()

    old_random, old_numpy, old_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        rank0 = run_once(0)
        assert run_once(0) == rank0
        rank1 = run_once(1)
        assert rank1 == run_once(1)
        assert rank0 != rank1
        assert rank0[0] == random.Random(42).random()
        assert rank0[1] == np.random.RandomState(42).random_sample()
        assert rank0[2] == torch.rand(1, generator=torch.Generator().manual_seed(42)).item()
    finally:
        random.setstate(old_random)
        np.random.set_state(old_numpy)
        torch.set_rng_state(old_torch)


def test_checkpoint_restore_overrides_fresh_seed_for_host_rngs(monkeypatch):
    from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict

    monkeypatch.setattr(misc.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    cuda_state = torch.tensor([12, 34], dtype=torch.uint8)
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state.clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: restored_cuda.append(state.clone()))
    old_random, old_numpy, old_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        misc.set_random_seed(seed=42, by_rank=True)
        saved = get_rand_state_dict()
        expected = (random.random(), np.random.random(), torch.rand(1).item())
        misc.set_random_seed(seed=99, by_rank=True)
        set_rand_state_dict(saved)
        assert (random.random(), np.random.random(), torch.rand(1).item()) == expected
        assert len(restored_cuda) == 1 and torch.equal(restored_cuda[0], cuda_state)
    finally:
        random.setstate(old_random)
        np.random.set_state(old_numpy)
        torch.set_rng_state(old_torch)
