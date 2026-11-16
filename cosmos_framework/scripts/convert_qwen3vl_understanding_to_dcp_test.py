# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.inference.model import Cosmos3OmniConfig
from cosmos_framework.scripts.convert_qwen3vl_understanding_to_dcp import (
    _build_target_config,
    _initialize_moe_gen_pathway,
    _validate_loaded_keys,
    _validate_target_config,
)

_SOURCE = Path("/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct")
_RECIPE = Path("examples/toml/sft_config/qwen3vl_v2_0a_understanding_qwen4b.toml")


def test_build_target_config_round_trips_without_instantiating_model(tmp_path: Path) -> None:
    config = _build_target_config(_SOURCE, _RECIPE)

    assert config.model["config"]["vision_gen"] is False
    assert config.model["config"]["action_gen"] is False
    assert config.model["config"]["sound_gen"] is False
    assert config.model["config"]["predict_text_tokens"] is True
    assert config.model["config"]["load_vlm_visual"] is True
    assert config.model["config"]["load_vision_tokenizer"] is False
    assert config.model["config"]["vlm_config"]["pretrained_weights"]["enabled"] is False

    config.save_pretrained(tmp_path)
    restored = Cosmos3OmniConfig.from_pretrained(tmp_path)
    assert restored.model == config.model


def test_validate_target_config_rejects_generation_head() -> None:
    model_dict = {
        "config": {
            "load_vlm_visual": True,
            "load_vision_tokenizer": False,
            "vision_gen": True,
            "action_gen": False,
            "sound_gen": False,
            "predict_text_tokens": True,
            "lora_enabled": False,
            "ema": {"enabled": False},
            "vlm_config": {
                "pretrained_weights": {
                    "enabled": False,
                    "backbone_path": str(_SOURCE),
                },
                "model_instance": {"config": {"include_visual": True}},
            },
        }
    }

    with pytest.raises(ValueError, match="not pure Understanding"):
        _validate_target_config(model_dict, _SOURCE)


def test_validate_loaded_keys_requires_full_visual_coverage() -> None:
    target = {"model.layers.0.weight", "visual.patch_embed.weight", "lm_head.weight"}
    source = {"model.language_model.layers.0.weight", "model.visual.patch_embed.weight"}

    with pytest.raises(ValueError, match="visual tensor"):
        _validate_loaded_keys(
            target,
            {"model.layers.0.weight"},
            source,
            tied_embeddings=True,
        )


def test_validate_loaded_keys_allows_tied_lm_head() -> None:
    target = {"model.layers.0.weight", "visual.patch_embed.weight", "lm_head.weight"}
    loaded = {"model.layers.0.weight", "visual.patch_embed.weight"}
    source = {"model.language_model.layers.0.weight", "model.visual.patch_embed.weight"}

    assert _validate_loaded_keys(target, loaded, source, tied_embeddings=True) == (1, 1)


def test_initialize_moe_gen_pathway_uses_authoritative_copy() -> None:
    class ToyUnifiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.proj_moe_gen = nn.Parameter(torch.tensor([-1.0, -1.0]))
            self.config = SimpleNamespace(tie_word_embeddings=False)

        def init_moe(self) -> None:
            self.proj_moe_gen.data.copy_(self.proj.data)

    model = ToyUnifiedModel()
    initialized = _initialize_moe_gen_pathway(model, {"proj"})

    assert initialized == {"proj_moe_gen"}
    assert torch.equal(model.proj_moe_gen, model.proj)
