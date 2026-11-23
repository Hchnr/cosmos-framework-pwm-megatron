# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""TP/PP must reject clipping settings with different native semantics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.grad_clip import _clip_grads_with_global_norm
from cosmos_framework.configs.toml_config import sft_config
from cosmos_framework.training_backends.megatron_tp_pp import train

RECIPE = (
    Path(__file__).resolve().parents[2]
    / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
)


def _load_recipe(monkeypatch, tmp_path, overrides=()):
    monkeypatch.setattr(sft_config, "_get_action128_roots", lambda datasets: [])
    monkeypatch.setenv("HF_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("WAN_VAE_PATH", str(tmp_path / "vae.pth"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", str(tmp_path / "base"))
    monkeypatch.setenv("MEGATRON_TP_PP_IMPORT_PATH", str(tmp_path / "bundle"))
    return sft_config.load_experiment_from_toml(RECIPE, extra_overrides=list(overrides))


@pytest.fixture
def config(monkeypatch, tmp_path):
    return _load_recipe(monkeypatch, tmp_path)


@pytest.mark.parametrize("clip_norm", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_unsupported_clip_norm_fails_before_distributed_setup(config, monkeypatch, clip_norm):
    config.trainer.callbacks.grad_clip.clip_norm = clip_norm
    monkeypatch.setattr(train.dist, "get_world_size", lambda: pytest.fail("queried distributed state"))
    with pytest.raises(ValueError, match="finite positive grad_clip.clip_norm"):
        train._validate(config, SimpleNamespace())


def test_default_recipe_retains_supported_clip_norm(config, monkeypatch):
    assert config.trainer.callbacks.grad_clip.clip_norm == 1.0
    monkeypatch.setattr(train.dist, "get_world_size", lambda: 4)
    topology = SimpleNamespace(data_parallel_size=lambda world_size: 1, logical_data_parallel_size=192)
    train._validate(config, topology)


def test_real_recipe_zero_clip_override_fails_before_distributed_setup(monkeypatch, tmp_path):
    config = _load_recipe(monkeypatch, tmp_path, ["trainer.callbacks.grad_clip.clip_norm=0.0"])
    assert config.trainer.callbacks.grad_clip.clip_norm == 0.0
    monkeypatch.setattr(train.dist, "get_world_size", lambda: pytest.fail("queried distributed state"))
    with pytest.raises(ValueError, match="finite positive grad_clip.clip_norm"):
        train._validate(config, SimpleNamespace())


def test_native_zero_clip_really_zeroes_finite_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    _clip_grads_with_global_norm({"default": [parameter]}, 0.0, torch.tensor(5.0))
    assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
