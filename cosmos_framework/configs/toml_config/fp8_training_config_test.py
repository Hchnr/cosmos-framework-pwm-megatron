# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig, load_experiment_from_toml
from cosmos_framework.inference.common.public_model_config import build_public_model_config

BASELINE = Path("examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml")


@pytest.fixture(autouse=True)
def _isolate_catalog_discovery(monkeypatch):
    # Config mapping must be testable without the production USR filesystem.
    monkeypatch.setattr("cosmos_framework.configs.toml_config.sft_config._get_action128_roots", lambda datasets: [])


def baseline():
    return tomllib.loads(BASELINE.read_text())


def test_default_disabled_and_old_recipe_unchanged():
    cfg = SFTExperimentConfig.model_validate(baseline())
    assert not cfg.model.fp8_training.enabled
    assert cfg.model.precision == "bfloat16"
    assert cfg.checkpoint.load_training_state
    assert cfg.trainer.grad_accum_iter == 2


@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
def test_enabled_config_maps_to_runtime(tmp_path, recipe):
    path = tmp_path / "fp8.toml"
    path.write_text(BASELINE.read_text() + f'\n[model.fp8_training]\nenabled=true\nrecipe="{recipe}"\n')
    config = load_experiment_from_toml(path)
    assert config.model.config.fp8_training.enabled
    assert config.model.config.fp8_training.recipe == recipe
    assert config.model.config.quantization.method is None


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("job", "task", "vlm", "only for VFM"),
        ("model", "precision", "float32", "bfloat16"),
        ("model", "lora_enabled", True, "LoRA or EMA"),
        ("model", "tower_structure", "three_tower", "two-tower"),
        ("model", "action_gen", True, "two-tower"),
    ],
)
def test_unsupported_config_rejected(section, key, value, match):
    raw = baseline()
    raw["model"]["fp8_training"] = {"enabled": True}
    raw[section][key] = value
    with pytest.raises(ValueError, match=match):
        SFTExperimentConfig.model_validate(raw)


def test_export_omits_training_policy_without_mutating_config():
    config = {"config": {"fp8_training": {"enabled": True, "_type": "private.training.type"}, "precision": "bfloat16"}}
    before = copy.deepcopy(config)
    assert build_public_model_config(config) == {"config": {"precision": "bfloat16"}}
    assert config == before
    config["enable_fp8_training"] = True
    assert "enable_fp8_training" not in build_public_model_config(config)


def test_cli_override_cannot_bypass_fp8_validation(tmp_path):
    path = tmp_path / "fp8.toml"
    path.write_text(BASELINE.read_text())
    with pytest.raises(ValueError, match="bfloat16"):
        load_experiment_from_toml(
            path, extra_overrides=["model.config.fp8_training.enabled=true", "model.config.precision=float32"]
        )


def test_disabled_vlm_section_does_not_leak_unknown_fields():
    from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

    overrides = build_hydra_overrides(
        {"job": {"task": "vlm", "experiment": "test"}, "model": {"fp8_training": {"enabled": False}}}
    )
    assert all("fp8_training" not in item for item in overrides)
