# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest

from cosmos_framework.configs.base.defaults.fixed_sequence_length import (
    FixedSequenceLengthConfig,
    resolve_fixed_sequence_targets,
    validate_fixed_sequence_length_contract,
)
from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.utils.config import Config


def _model_config(
    *,
    fixed: bool,
    compile_enabled: bool,
    compile_dynamic: bool,
    und_tokens: int = 24,
    gen_tokens: int = 40,
    understanding_tokens: int = 0,
    cp_size: int = 1,
    compiled_region: str = "language",
    max_tokens: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        fixed_sequence_length=SimpleNamespace(
            enabled=fixed,
            understanding_tokens=understanding_tokens,
            und_tokens=und_tokens,
            gen_tokens=gen_tokens,
        ),
        compile=SimpleNamespace(
            enabled=compile_enabled,
            compile_dynamic=compile_dynamic,
            compiled_region=compiled_region,
        ),
        parallelism=SimpleNamespace(context_parallel_shard_degree=cp_size),
        max_num_tokens_after_packing=max_tokens,
    )


@pytest.mark.parametrize(
    ("fixed", "compile_enabled", "compile_dynamic"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_valid_compile_truth_table(fixed: bool, compile_enabled: bool, compile_dynamic: bool) -> None:
    validate_fixed_sequence_length_contract(
        _model_config(
            fixed=fixed,
            compile_enabled=compile_enabled,
            compile_dynamic=compile_dynamic,
        )
    )


def test_variable_sequence_static_compile_fails() -> None:
    with pytest.raises(ValueError, match="variable sequence length only supports compile_dynamic=true"):
        validate_fixed_sequence_length_contract(_model_config(fixed=False, compile_enabled=True, compile_dynamic=False))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"und_tokens": 1}, "greater than one"),
        ({"gen_tokens": 1}, "greater than one"),
        ({"understanding_tokens": 1}, "greater than one"),
        ({"understanding_tokens": -1}, r"zero \(disabled\) or greater than one"),
        ({"und_tokens": 0}, "configured together"),
        ({"gen_tokens": 0}, "configured together"),
        ({"cp_size": 0}, "must be positive"),
        ({"cp_size": 16, "und_tokens": 24}, "must be divisible"),
        ({"max_tokens": 65}, "must equal the largest enabled fixed profile"),
        ({"compiled_region": "all"}, "only supports compiled_region='language'"),
    ],
)
def test_invalid_fixed_contract_fails(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_fixed_sequence_length_contract(
            _model_config(
                fixed=True,
                compile_enabled=True,
                compile_dynamic=False,
                **kwargs,
            )
        )


def test_standalone_vlm_without_fixed_contract_is_unchanged() -> None:
    validate_fixed_sequence_length_contract(
        SimpleNamespace(
            compile=SimpleNamespace(enabled=True, compile_dynamic=False),
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "has_generation", "expected"),
    [
        ({"understanding_tokens": 48, "und_tokens": 0, "gen_tokens": 0}, False, (48, 0, "understanding")),
        ({"understanding_tokens": 0, "und_tokens": 24, "gen_tokens": 40}, True, (24, 40, "generation")),
        ({"understanding_tokens": 64, "und_tokens": 24, "gen_tokens": 40}, False, (64, 0, "understanding")),
        ({"understanding_tokens": 64, "und_tokens": 24, "gen_tokens": 40}, True, (24, 40, "generation")),
    ],
)
def test_resolve_task_profile(kwargs: dict[str, int], has_generation: bool, expected: tuple[int, int, str]) -> None:
    fixed = SimpleNamespace(enabled=True, **kwargs)
    targets = resolve_fixed_sequence_targets(fixed, has_generation=has_generation)
    assert targets is not None
    assert (targets.causal, targets.full, targets.profile) == expected


def test_omni_mot_default_is_disabled() -> None:
    config = OmniMoTModelConfig()
    assert config.fixed_sequence_length == FixedSequenceLengthConfig()
    assert not config.fixed_sequence_length.enabled


def test_top_level_config_rejects_invalid_contract_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_cuda_allocation_is_reached(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA allocation was reached before fixed-sequence validation")

    monkeypatch.setattr("torch.ByteTensor", fail_if_cuda_allocation_is_reached)
    config = Config(
        model={
            "config": _model_config(
                fixed=False,
                compile_enabled=True,
                compile_dynamic=False,
            )
        },
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    with pytest.raises(ValueError, match="Static decoder compile requires"):
        config.validate()


@pytest.mark.parametrize("action_tokens", [0, -1, 40, 41])
def test_small_tower_requires_separate_action_capacity(action_tokens):
    model = _model_config(fixed=True, compile_enabled=True, compile_dynamic=False)
    model.action_tower_preset = "width_only_1b"
    model.fixed_sequence_length.action_tokens = action_tokens
    with pytest.raises(ValueError, match="action_tokens"):
        validate_fixed_sequence_length_contract(model)


def test_small_tower_action_capacity_is_part_of_gen_budget():
    model = _model_config(fixed=True, compile_enabled=True, compile_dynamic=False)
    model.action_tower_preset = "width_only_1b"
    model.fixed_sequence_length.action_tokens = 8
    validate_fixed_sequence_length_contract(model)
    targets = resolve_fixed_sequence_targets(model.fixed_sequence_length, has_generation=True)
    assert targets.full == 40
