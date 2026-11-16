# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cosmos_framework.scripts import train


class _FakeConfig:
    def __init__(self, *, train_error: BaseException | None = None):
        self.events: list[str] = []
        self.model = "model-config"
        self.dataloader_train = "train-loader-config"
        self.dataloader_val = "val-loader-config"
        self.trainer = SimpleNamespace(type=lambda _config: _FakeTrainer(self.events, train_error))

    def freeze(self) -> None:
        self.events.append("freeze")


class _FakeTrainer:
    def __init__(self, events: list[str], train_error: BaseException | None):
        events.append("trainer")
        self.events = events
        self.train_error = train_error

    def train(self, model, dataloader_train, dataloader_val) -> None:  # noqa: ANN001
        self.events.append(f"train:{model}:{dataloader_train}:{dataloader_val}")
        if self.train_error is not None:
            raise self.train_error


@pytest.fixture(autouse=True)
def _patch_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_configure(config: _FakeConfig, *, deterministic: bool) -> None:
        config.events.append(f"configure:{deterministic}")

    monkeypatch.setattr(train, "configure_compile_cache", fake_configure)
    monkeypatch.setattr(train, "model_init", nullcontext)
    monkeypatch.setattr(train, "data_loader_init", nullcontext)
    monkeypatch.setattr(train, "log_reproducible_setup", lambda _config, _args: None)
    monkeypatch.setattr(train, "instantiate", lambda value: f"instance:{value}")
    monkeypatch.setattr(
        "cosmos_framework.callbacks.cosmos_dataloader_state.attach_cosmos_dataloader_state_callback",
        lambda _trainer, _dataloader: False,
    )


def _args(*, deterministic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(deterministic=deterministic, attach_vscode_debugger=False)


def test_compile_cache_is_configured_before_trainer_and_model_construction() -> None:
    config = _FakeConfig()

    train._run_training(config, _args())

    assert config.events == [
        "configure:False",
        "freeze",
        "trainer",
        "train:instance:model-config:instance:train-loader-config:instance:val-loader-config",
    ]


def test_deterministic_flag_is_forwarded_to_compile_cache_configuration() -> None:
    config = _FakeConfig()

    train._run_training(config, _args(deterministic=True))

    assert config.events[0] == "configure:True"


def test_training_failure_propagates_without_compile_cache_finalize() -> None:
    config = _FakeConfig(train_error=RuntimeError("training failed"))

    with pytest.raises(RuntimeError, match="training failed"):
        train._run_training(config, _args())

    assert config.events[0] == "configure:False"


def test_training_explicitly_opts_in_to_fp8_build(monkeypatch) -> None:
    config = _FakeConfig()
    config.model = SimpleNamespace(config=SimpleNamespace(fp8_training=SimpleNamespace(enabled=True)))
    calls = []

    def instantiate(value, **kwargs):
        calls.append((value, kwargs))
        return "instance"

    monkeypatch.setattr(train, "instantiate", instantiate)
    train._run_training(config, _args())
    assert calls[0] == (config.model, {"enable_fp8_training": True})
    assert all(not kwargs for _, kwargs in calls[1:])
