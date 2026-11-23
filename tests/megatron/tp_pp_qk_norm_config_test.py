# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""TP/PP must not silently replace the native model's Q/K-norm policy."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.configs.toml_config import sft_config
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, PackedAttentionMoT
from cosmos_framework.model.generator.omni_mot_model import _prepare_full_model_instance
from cosmos_framework.training_backends.megatron_tp_pp import model as tp_model
from cosmos_framework.utils.lazy_config import instantiate

RECIPE = (
    Path(__file__).resolve().parents[2]
    / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"
)


def _load_without_catalog_io(monkeypatch, tmp_path, overrides):
    # Resolving the production recipe should not need actual training data or
    # source checkpoint contents for this model-construction contract.
    monkeypatch.setattr(sft_config, "_get_action128_roots", lambda datasets: [])
    monkeypatch.setenv("HF_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("WAN_VAE_PATH", str(tmp_path / "vae.pth"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", str(tmp_path / "base"))
    monkeypatch.setenv("MEGATRON_TP_PP_IMPORT_PATH", str(tmp_path / "bundle"))
    return sft_config.load_experiment_from_toml(RECIPE, extra_overrides=overrides)


def _native_language_config(config):
    instance, _, _ = _prepare_full_model_instance(
        config.model.config.vlm_config.model_instance, load_vlm_visual=False, include_generation_tower=True
    )
    return instantiate(instance["config"])


@pytest.mark.parametrize("field", ["qk_norm_for_text", "qk_norm_for_diffusion"])
def test_production_recipe_hydra_override_must_not_be_silently_ignored(monkeypatch, tmp_path, field):
    prefix = "+" if field == "qk_norm_for_diffusion" else ""
    override = f"{prefix}model.config.vlm_config.model_instance.config.{field}=false"
    config = _load_without_catalog_io(monkeypatch, tmp_path, [override])
    assert getattr(_native_language_config(config), field) is False
    # Reject immediately after config instantiation, before GPU stage creation.
    monkeypatch.setattr(tp_model, "PipelineStage", lambda *args, **kwargs: pytest.fail("allocated a stage"))
    with pytest.raises(ValueError, match="qk_norm_for_text=True and qk_norm_for_diffusion=True"):
        tp_model.build_stage(config.model.config, SimpleNamespace())

    # Negative control for the previous implementation: bypassing only the
    # new guard reaches the real TP factory, which hard-codes both norms on.
    class ReachedStage(Exception):
        pass

    observed = {}

    def capture_stage(_config, layer_factory, **kwargs):
        layer = layer_factory(0)  # The real production factory constructs on meta.
        observed["text"] = layer.self_attn.q_norm
        observed["diffusion"] = layer.self_attn.q_norm_moe_gen
        raise ReachedStage

    with monkeypatch.context() as old:
        old.setattr(tp_model, "_validate_qk_norm_config", lambda _: None)
        old.setattr(tp_model, "transformer_config", lambda *args, **kwargs: None)
        old.setattr(tp_model, "PipelineStage", capture_stage)
        with pytest.raises(ReachedStage):
            tp_model.build_stage(config.model.config, SimpleNamespace())
    ignored = observed["text" if field == "qk_norm_for_text" else "diffusion"]
    assert not isinstance(ignored, nn.Identity)


def test_production_recipe_default_satisfies_norm_contract(monkeypatch, tmp_path):
    config = _load_without_catalog_io(monkeypatch, tmp_path, [])
    native = _native_language_config(config)
    assert native.qk_norm_for_text is True and native.qk_norm_for_diffusion is True
    tp_model._validate_qk_norm_config(native)


@pytest.mark.parametrize("field", ["qk_norm_for_text", "qk_norm_for_diffusion"])
def test_disabling_qk_norm_changes_native_attention_math(field):
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    options = {"qk_norm_for_text": True, "qk_norm_for_diffusion": True}
    active = PackedAttentionMoT(config, layer_idx=0, layer_types=LayerTypes("qwen3_vl_dense"), **options)
    options[field] = False
    disabled = PackedAttentionMoT(config, layer_idx=0, layer_types=LayerTypes("qwen3_vl_dense"), **options)
    name = "q_norm" if field == "qk_norm_for_text" else "q_norm_moe_gen"
    assert isinstance(getattr(disabled, name), nn.Identity)
    values = torch.tensor([[[2.0, 1.0]]])
    assert torch.equal(getattr(disabled, name)(values), values)
    assert not torch.equal(getattr(active, name)(values), values)


def test_missing_norm_policy_fails_closed():
    with pytest.raises(ValueError, match="qk_norm_for_text=True"):
        tp_model._validate_qk_norm_config(SimpleNamespace(qk_norm_for_text=True))
