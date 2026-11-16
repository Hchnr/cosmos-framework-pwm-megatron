# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter

from cosmos_framework.scripts._merge_understanding_dcp import (
    _allocate,
    _model_family_signature,
    _tensor_metadata,
    merge_understanding_dcp,
)


def _save_model(
    root: Path,
    state: dict[str, torch.Tensor],
    *,
    config: bool = False,
    family: str = "test_model",
) -> None:
    model = root / "model"
    model.mkdir(parents=True)
    dcp.save(state, storage_writer=FileSystemWriter(model))
    if config:
        payload = {
            "model": {
                "config": {
                    "vlm_config": {
                        "model_instance": {
                            "_target": family,
                            "config": {"base_config": {"json_file": f"{family}.json"}},
                        }
                    }
                }
            }
        }
        (model / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        (root / "checkpoint.json").write_text("{}\n", encoding="utf-8")


def _load_model(root: Path) -> dict[str, torch.Tensor]:
    model = root / "model"
    state = _allocate(_tensor_metadata(model))
    dcp.load(state, storage_reader=FileSystemReader(model), no_dist=True)
    return state


def test_merge_understanding_dcp_preserves_gen_and_updates_und_and_ema(tmp_path: Path) -> None:
    full = tmp_path / "full"
    slim = tmp_path / "slim"
    output = tmp_path / "merged"
    _save_model(
        full,
        {
            "net.language_model.model.embed_tokens.weight": torch.full((2, 2), 1.0),
            "net.language_model.model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 2.0),
            "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": torch.full((2, 2), 3.0),
            "net.time_embedder.weight": torch.full((2, 2), 4.0),
            "net_ema.language_model.model.embed_tokens.weight": torch.full((2, 2), 5.0),
            "net_ema.language_model.model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 6.0),
            "net_ema.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": torch.full((2, 2), 7.0),
            "net_ema.time_embedder.weight": torch.full((2, 2), 8.0),
        },
        config=True,
    )
    _save_model(
        slim,
        {
            "net.language_model.model.embed_tokens.weight": torch.full((2, 2), 11.0),
            "net.language_model.model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 12.0),
        },
    )

    report = merge_understanding_dcp(full, slim, output)
    merged = _load_model(output)

    assert report.understanding_tensors == 2
    assert report.ema_tensors_updated == 2
    torch.testing.assert_close(merged["net.language_model.model.embed_tokens.weight"], torch.full((2, 2), 11.0))
    torch.testing.assert_close(
        merged["net.language_model.model.layers.0.self_attn.q_proj.weight"], torch.full((2, 2), 12.0)
    )
    torch.testing.assert_close(
        merged["net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"], torch.full((2, 2), 3.0)
    )
    torch.testing.assert_close(merged["net.time_embedder.weight"], torch.full((2, 2), 4.0))
    torch.testing.assert_close(merged["net_ema.language_model.model.embed_tokens.weight"], torch.full((2, 2), 11.0))
    torch.testing.assert_close(
        merged["net_ema.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"],
        torch.full((2, 2), 7.0),
    )
    assert (output / "model" / "config.json").is_file()
    assert (output / "checkpoint.json").is_file()


def test_merge_rejects_generation_weights_in_slim_checkpoint(tmp_path: Path) -> None:
    full = tmp_path / "full"
    slim = tmp_path / "slim"
    _save_model(
        full,
        {
            "net.language_model.model.embed_tokens.weight": torch.ones(1),
            "net.language_model.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": torch.ones(1),
        },
        config=True,
    )
    _save_model(
        slim,
        {
            "net.language_model.model.embed_tokens.weight": torch.ones(1),
            "net.language_model.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": torch.ones(1),
        },
    )

    with pytest.raises(ValueError, match="still contains generation parameters"):
        merge_understanding_dcp(full, slim, tmp_path / "merged")


def test_merge_rejects_different_model_families(tmp_path: Path) -> None:
    full = tmp_path / "full"
    slim = tmp_path / "slim"
    common = {
        "net.language_model.model.embed_tokens.weight": torch.ones(1),
        "net.language_model.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
    }
    _save_model(full, common, config=True, family="qwen")
    _save_model(slim, common, config=True, family="nemotron")

    with pytest.raises(ValueError, match="model families differ"):
        merge_understanding_dcp(full, slim, tmp_path / "merged")


def test_model_family_signature_normalizes_qwen_serialization_variants(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    common = {"net.language_model.model.embed_tokens.weight": torch.ones(1)}
    _save_model(legacy, common, config=True, family="qwen3_vl_text_for_causal_lm")
    _save_model(current, common, config=True, family="placeholder")

    legacy_config_path = legacy / "model" / "config.json"
    legacy_config = json.loads(legacy_config_path.read_text(encoding="utf-8"))
    legacy_instance = legacy_config["model"]["config"]["vlm_config"]["model_instance"]
    legacy_instance["config"]["base_config"]["json_file"] = (
        "cosmos3://vfm/models/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    legacy_config_path.write_text(json.dumps(legacy_config), encoding="utf-8")

    current_config_path = current / "model" / "config.json"
    current_config = json.loads(current_config_path.read_text(encoding="utf-8"))
    current_instance = current_config["model"]["config"]["vlm_config"]["model_instance"]
    current_instance["_target_"] = "cosmos_framework.model.generator.mot.unified_mot.Qwen3VLTextForCausalLM"
    del current_instance["_target"]
    current_instance["config"]["base_config"]["json_file"] = (
        "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    current_config_path.write_text(json.dumps(current_config), encoding="utf-8")

    assert _model_family_signature(legacy / "model") == _model_family_signature(current / "model")
