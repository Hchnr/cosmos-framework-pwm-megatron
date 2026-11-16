# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from torch import nn

from cosmos_framework.model.generator.mot import parallelize_unified_mot as unified_parallel
from cosmos_framework.model.generator.mot import parallelize_vfm_network as vfm_parallel


class _LanguageModel(nn.Module):
    def __init__(self, *, train_visual: bool) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self._reasoner_visual_artifact = True
        self._reasoner_visual_trainable = train_visual


class _VFM(nn.Module):
    def __init__(self, *, train_visual: bool) -> None:
        super().__init__()
        self.language_model = _LanguageModel(train_visual=train_visual)

    def generate_reasoner_text(self) -> None:
        pass


@pytest.mark.parametrize("train_visual", [False, True])
def test_activation_checkpointing_wraps_visual_only_when_trainable(
    monkeypatch: pytest.MonkeyPatch,
    train_visual: bool,
) -> None:
    model = _LanguageModel(train_visual=train_visual)
    language_blocks = list(model.model.layers)
    visual_blocks = list(model.visual.blocks)
    wrapped: list[nn.Module] = []

    def _record_wrapper(block: nn.Module, _config: object) -> nn.Module:
        wrapped.append(block)
        return block

    monkeypatch.setattr(unified_parallel, "_apply_ac_to_transformer_block", _record_wrapper)

    unified_parallel.apply_ac(model, SimpleNamespace(mode="full"))

    expected = language_blocks + (visual_blocks if train_visual else [])
    assert wrapped == expected


@pytest.mark.parametrize("train_visual", [False, True])
def test_fsdp_shards_visual_only_when_trainable(
    monkeypatch: pytest.MonkeyPatch,
    train_visual: bool,
) -> None:
    model = _LanguageModel(train_visual=train_visual)
    language_blocks = list(model.model.layers)
    visual_blocks = list(model.visual.blocks)
    mesh = object()
    shard = Mock()
    register = Mock()
    monkeypatch.setattr(unified_parallel, "fsdp_mesh", lambda _parallel_dims: mesh)
    monkeypatch.setattr(unified_parallel, "fully_shard", shard)
    monkeypatch.setattr(unified_parallel, "register_fsdp_forward_method", register)

    unified_parallel.apply_fsdp(model, SimpleNamespace())

    expected_blocks = (visual_blocks if train_visual else []) + language_blocks
    assert shard.call_args_list == [call(block, mesh=mesh) for block in expected_blocks]
    assert register.call_args_list == [call(block, "reasoner_forward") for block in language_blocks]


@pytest.mark.parametrize("train_visual", [False, True])
def test_root_fsdp_ignores_only_frozen_visual_parameters(
    monkeypatch: pytest.MonkeyPatch,
    train_visual: bool,
) -> None:
    model = _VFM(train_visual=train_visual)
    mesh = object()
    root_shard = Mock(side_effect=lambda *, module, **_kwargs: module)
    monkeypatch.setattr(
        vfm_parallel,
        "parallelize_unified_mot",
        lambda language_model, **_kwargs: language_model,
    )
    monkeypatch.setattr(vfm_parallel, "fsdp_mesh", lambda _parallel_dims: mesh)
    monkeypatch.setattr(vfm_parallel, "fully_shard", root_shard)
    monkeypatch.setattr(vfm_parallel, "register_fsdp_forward_method", Mock())

    result = vfm_parallel.parallelize_vfm_network(
        model,
        parallel_dims=SimpleNamespace(cp_enabled=False, dp_enabled=True),
        compile_config=SimpleNamespace(enabled=False),
        ac_config=SimpleNamespace(mode="full"),
    )

    assert result is model
    assert root_shard.call_args.kwargs["mesh"] is mesh
    ignored = root_shard.call_args.kwargs["ignored_params"]
    expected = set() if train_visual else set(model.language_model.visual.parameters())
    assert ignored == expected
