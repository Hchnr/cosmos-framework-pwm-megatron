# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Contract tests for the frozen V2 Qwen3-VL Understanding migration recipes."""

from __future__ import annotations

from pathlib import Path

import tomllib

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, _prepare_full_model_instance

_REPO_ROOT = Path(__file__).parents[3]
_TOML_PATH = _REPO_ROOT / "examples/toml/sft_config/qwen3vl_v2_0a_understanding_qwen4b.toml"
_LAUNCHER_PATH = _REPO_ROOT / "examples/launch_sft_qwen3vl_v2_understanding_qwen4b.sh"
_NANO_TOML_PATH = _REPO_ROOT / "examples/toml/sft_config/qwen3vl_v2_0a_understanding_nano8b.toml"
_NANO_LAUNCHER_PATH = _REPO_ROOT / "examples/launch_sft_qwen3vl_v2_understanding_nano8b.sh"
_INCLUDE_VISUAL_OVERRIDE = "+model.config.vlm_config.model_instance.config.include_visual=true"
_PACKING_TOKEN_OVERRIDE = "+model.config.vlm_config.model_instance.config.max_num_tokens_after_packing=49152"
_DISABLE_SAMPLE_REG_OVERRIDE = "~trainer.callbacks.every_n_sample_reg=null"
_DISABLE_SAMPLE_EMA_OVERRIDE = "~trainer.callbacks.every_n_sample_ema=null"
_NANO_PROCESSOR_OVERRIDE = (
    "model.config.vlm_config.tokenizer.pretrained_model_name=/share/project/eai_pwm/models/Cosmos3-Nano"
)
_PARQUET_ROOT = Path(
    "/share/project/eai_pwm/prepared_datasets/parquet_datasets/understanding/qwen3vl_v2_0a_20260814_reference_v1"
)
_EXPECTED_DATASETS = [
    ("vlm_robotics_fullscan_v2r1_p0_episodic_embodied", 6136),
    ("vlm_robotics_fullscan_v2r1_p0_metric_3d_scene", 1356),
    ("vlm_robotics_fullscan_v2r1_p0_multiview_memory", 1182),
    ("vlm_robotics_fullscan_v2r1_p0_temporal_physical", 4907),
    ("vlm_robotics_fullscan_v2r1_p1_episodic_embodied", 57),
    ("vlm_robotics_fullscan_v2r1_p1_free_space_affordance", 61),
    ("vlm_robotics_fullscan_v2r1_p1_metric_3d_scene", 879),
    ("vlm_robotics_fullscan_v2r1_p1_multiview_memory", 214),
    ("vlm_robotics_fullscan_v2r1_p1_temporal_physical", 2370),
    ("vlm_robotics_affordance_support_v1_robot_placement", 140),
    ("vlm_robotics_affordance_support_v1_semantic_affordance", 1),
    ("vlm_robotics_fullscan_v2r1_p2_visual_structured_retention", 4944),
    ("vlm_robotics_general_retention_v1_safe_general", 2472),
    ("gqa_train_balanced_reference", 446),
    ("mindcube_train_spatial_qa_reference", 45),
    ("sharerobot_planning_qa_v1_train", 4589),
    ("sharerobot_affordance_bbox_text_v1_train", 29),
    ("sharerobot_trajectory_points_text_v1_train", 31),
    ("msr3d_strict_object_image_qa_v1_train", 362),
    ("robobench_official_prompt_all_training_v1_train", 27),
]
_EXCLUDED_DATASETS = {
    "vlm_robotics_general_retention_v1_vstar_risk_general",
    "vlm_robotics_onevision_optional_p2_recovery",
    "model_judge_intphys_provisional_p1_capability",
}


def test_frozen_v2_recipe_contract() -> None:
    with _TOML_PATH.open("rb") as file:
        raw = tomllib.load(file)

    assert raw["job"]["experiment"] == "vision_sft_qwen4b_gen"

    model = raw["model"]
    assert model["precision"] == "bfloat16"
    assert model["max_num_tokens_after_packing"] == 49152
    assert model["load_vlm_visual"] is True
    assert model["train_vlm_visual"] is True
    assert model["load_vision_tokenizer"] is False
    assert model["vision_gen"] is False
    assert model["action_gen"] is False
    assert model["predict_text_tokens"] is True
    assert model["ema"]["enabled"] is False

    assert raw["optimizer"]["keys_to_select"] == []
    assert raw["optimizer"]["lr"] == 1.0e-5
    assert raw["optimizer"]["weight_decay"] == 0.0
    assert raw["scheduler"] == {
        "cycle_lengths": [40000],
        "f_max": [1.0],
        "f_min": [1.0],
        "f_start": [0.0],
        "verbosity_interval": 0,
        "warm_up_steps": [200],
    }
    assert raw["trainer"]["grad_accum_iter"] == 2
    assert raw["trainer"]["max_iter"] == 40000
    assert raw["checkpoint"]["load_path"] == ""
    assert raw["checkpoint"]["save_iter"] == 3000
    assert raw["dataloader_train"]["max_samples_per_batch"] == 48
    assert raw["dataloader_train"]["packing_pool_size"] == 48
    assert raw["dataloader_train"]["max_sequence_length"] == 49152
    assert raw["dataloader_train"]["enable_map_style_resume"] is False
    assert raw["dataloader_train"]["seed"] == 42

    specs = raw["dataloader_train"]["datasets"]
    actual = [(Path(spec["path"]).name, spec["ratio"]) for spec in specs]
    assert actual == _EXPECTED_DATASETS
    assert sum(ratio for _, ratio in actual) == 30248
    assert all(Path(spec["path"]).parent == _PARQUET_ROOT for spec in specs)
    assert not ({name for name, _ in actual} & _EXCLUDED_DATASETS)
    assert _INCLUDE_VISUAL_OVERRIDE in _LAUNCHER_PATH.read_text()
    assert _PACKING_TOKEN_OVERRIDE in _LAUNCHER_PATH.read_text()
    assert _DISABLE_SAMPLE_REG_OVERRIDE in _LAUNCHER_PATH.read_text()
    assert _DISABLE_SAMPLE_EMA_OVERRIDE in _LAUNCHER_PATH.read_text()
    assert "PREFER_PROJECT_CUDA_LIBS=1" in _LAUNCHER_PATH.read_text()
    assert '"trainer.max_iter=$SMOKE_MAX_ITER"' in _LAUNCHER_PATH.read_text()
    assert '"trainer.grad_accum_iter=$SMOKE_GRAD_ACCUM_ITER"' in _LAUNCHER_PATH.read_text()


def test_recipe_loads_as_qwen4b_pure_understanding(monkeypatch) -> None:
    qwen_path = "/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct"
    monkeypatch.setenv("QWEN3_VL_4B_PATH", qwen_path)

    config = load_experiment_from_toml(
        _TOML_PATH,
        extra_overrides=[
            _INCLUDE_VISUAL_OVERRIDE,
            _PACKING_TOKEN_OVERRIDE,
            _DISABLE_SAMPLE_REG_OVERRIDE,
            _DISABLE_SAMPLE_EMA_OVERRIDE,
        ],
    )

    model = config.model.config
    assert model.load_vlm_visual is True
    assert model.train_vlm_visual is True
    assert model.load_vision_tokenizer is False
    assert model.vision_gen is False
    assert model.action_gen is False
    assert model.predict_text_tokens is True
    assert model.ema.enabled is False
    assert model.vlm_config.tokenizer.tokenizer_type == qwen_path
    assert model.vlm_config.pretrained_weights.backbone_path == qwen_path

    prepared, _, external_visual = _prepare_full_model_instance(
        model.vlm_config.model_instance,
        load_vlm_visual=model.load_vlm_visual,
        include_generation_tower=False,
    )
    assert external_visual is False
    assert prepared["config"]["include_visual"] is True
    assert prepared["config"]["include_generation_tower"] is False

    assert list(config.optimizer.keys_to_select) == []
    assert config.optimizer.lr == 1.0e-5
    assert config.trainer.grad_accum_iter == 2
    assert config.trainer.max_iter == 40000
    assert "every_n_sample_reg" not in config.trainer.callbacks
    assert "every_n_sample_ema" not in config.trainer.callbacks
    assert config.checkpoint.load_path == ""
    assert config.dataloader_train.max_samples_per_batch == 48
    assert config.dataloader_train.packing_pool_size == 48
    assert config.dataloader_train.max_sequence_length == 49152
    assert config.dataloader_train.enable_map_style_resume is False
    assert "dataloader_state" not in config.trainer.callbacks
    assert [(Path(spec["path"]).name, spec["ratio"]) for spec in config.catalog_datasets] == _EXPECTED_DATASETS


def test_qwen4b_production_recipe_supports_fixed_static_understanding(tmp_path, monkeypatch) -> None:
    """Validate fixed UND using the team's most-used VFLM-only recipe."""

    qwen_path = "/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct"
    monkeypatch.setenv("QWEN3_VL_4B_PATH", qwen_path)
    recipe = _TOML_PATH.read_text()
    recipe = recipe.replace(
        "[model.compile]\nenabled         = true\ncompile_dynamic = true",
        "[model.compile]\nenabled         = true\ncompile_dynamic = false",
    )
    recipe = recipe.replace(
        "[model.ema]",
        "[model.fixed_sequence_length]\n"
        "enabled = true\n"
        "understanding_tokens = 49152\n"
        "und_tokens = 0\n"
        "gen_tokens = 0\n\n"
        "[model.ema]",
    )
    fixed_recipe = tmp_path / "qwen3vl_v2_0a_understanding_qwen4b_fixed_static.toml"
    fixed_recipe.write_text(recipe)

    config = load_experiment_from_toml(
        fixed_recipe,
        extra_overrides=[
            _INCLUDE_VISUAL_OVERRIDE,
            _PACKING_TOKEN_OVERRIDE,
            _DISABLE_SAMPLE_REG_OVERRIDE,
            _DISABLE_SAMPLE_EMA_OVERRIDE,
        ],
    )

    assert config.model.config.fixed_sequence_length.enabled is True
    assert config.model.config.fixed_sequence_length.understanding_tokens == 49152
    assert config.model.config.fixed_sequence_length.und_tokens == 0
    assert config.model.config.fixed_sequence_length.gen_tokens == 0
    assert config.model.config.compile.compile_dynamic is False
    assert config.dataloader_train.max_sequence_length == 49152
    assert all(Path(spec["path"]).is_dir() for spec in config.catalog_datasets)


def test_nano8b_frozen_v2_recipe_contract() -> None:
    with _NANO_TOML_PATH.open("rb") as file:
        raw = tomllib.load(file)

    assert raw["job"]["task"] == "vfm"
    assert raw["job"]["experiment"] == "understanding_sft_nano"

    model = raw["model"]
    assert model["precision"] == "bfloat16"
    assert model["max_num_tokens_after_packing"] == 8192
    assert model["load_vlm_visual"] is True
    assert model.get("train_vlm_visual", False) is False
    assert model["load_vision_tokenizer"] is False
    assert model["vision_gen"] is False
    assert model["action_gen"] is False
    assert model["predict_text_tokens"] is True
    assert model["ema"]["enabled"] is False
    assert model["compile"]["enabled"] is False

    assert raw["optimizer"]["keys_to_select"] == []
    assert raw["optimizer"]["lr"] == 1.0e-6
    assert raw["optimizer"]["weight_decay"] == 0.1
    assert raw["checkpoint"]["load_path"] == "${oc.env:BASE_CHECKPOINT_PATH}"
    assert raw["dataloader_train"]["max_samples_per_batch"] == 48
    assert raw["dataloader_train"]["packing_pool_size"] == 48
    assert raw["dataloader_train"]["max_sequence_length"] == 8192
    assert raw["dataloader_train"]["enable_map_style_resume"] is True
    assert raw["dataloader_train"]["map_style_resume_mode"] == "at_most_once"
    assert raw["dataloader_train"]["seed"] == 42

    specs = raw["dataloader_train"]["datasets"]
    actual = [(Path(spec["path"]).name, spec["ratio"]) for spec in specs]
    assert actual == _EXPECTED_DATASETS
    assert sum(ratio for _, ratio in actual) == 30248
    assert all(Path(spec["path"]).parent == _PARQUET_ROOT for spec in specs)
    assert not ({name for name, _ in actual} & _EXCLUDED_DATASETS)

    launcher = _NANO_LAUNCHER_PATH.read_text()
    assert "/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP" in launcher
    assert _INCLUDE_VISUAL_OVERRIDE in launcher
    assert "model.config.vlm_config.tokenizer.pretrained_model_name=$HF_MODEL_PATH" in launcher
    assert "REQUIRE_WAN_VAE=0" in launcher
    assert "PREFER_PROJECT_CUDA_LIBS=1" in launcher
    assert _DISABLE_SAMPLE_REG_OVERRIDE in launcher
    assert _DISABLE_SAMPLE_EMA_OVERRIDE in launcher
    assert '"trainer.max_iter=$SMOKE_MAX_ITER"' in launcher
    assert '"trainer.grad_accum_iter=$SMOKE_GRAD_ACCUM_ITER"' in launcher
    assert '"checkpoint.save_iter=$SMOKE_CHECKPOINT_SAVE_ITER"' in launcher


def test_nano8b_recipe_loads_as_omnimot_pure_understanding(monkeypatch) -> None:
    checkpoint_path = "/share/project/liyue/model_ckpts/Cosmos3-Nano-DCP"
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", checkpoint_path)

    config = load_experiment_from_toml(
        _NANO_TOML_PATH,
        extra_overrides=[
            _INCLUDE_VISUAL_OVERRIDE,
            _NANO_PROCESSOR_OVERRIDE,
            _DISABLE_SAMPLE_REG_OVERRIDE,
            _DISABLE_SAMPLE_EMA_OVERRIDE,
        ],
    )

    assert config.model._target_ in {
        OmniMoTModel,
        "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel",
    }
    model = config.model.config
    assert model.load_vlm_visual is True
    assert model.train_vlm_visual is False
    assert model.load_vision_tokenizer is False
    assert model.vision_gen is False
    assert model.action_gen is False
    assert model.sound_gen is False
    assert model.predict_text_tokens is True
    assert model.ema.enabled is False
    assert model.vlm_config.model_name == "Qwen/Qwen3-VL-8B-Instruct"
    assert model.vlm_config.pretrained_weights.enabled is False
    assert model.vlm_config.tokenizer.pretrained_model_name == "/share/project/eai_pwm/models/Cosmos3-Nano"

    prepared, _, external_visual = _prepare_full_model_instance(
        model.vlm_config.model_instance,
        load_vlm_visual=model.load_vlm_visual,
        include_generation_tower=False,
    )
    assert external_visual is False
    assert prepared["config"]["include_visual"] is True
    assert prepared["config"]["include_generation_tower"] is False

    assert config.checkpoint.load_path == checkpoint_path
    assert config.optimizer.lr == 1.0e-6
    assert config.optimizer.weight_decay == 0.1
    assert [(Path(spec["path"]).name, spec["ratio"]) for spec in config.catalog_datasets] == _EXPECTED_DATASETS
    assert config.dataloader_train.max_samples_per_batch == 48
    assert config.dataloader_train.packing_pool_size == 48
    assert config.dataloader_train.max_sequence_length == 8192
    assert config.dataloader_train.enable_map_style_resume is True
    assert config.dataloader_train.map_style_resume_mode == "at_most_once"
    assert config.trainer.callbacks.dataloader_state._target_ in {
        JointCosmosDataLoaderStateCallback,
        "cosmos_framework.callbacks.cosmos_dataloader_state.JointCosmosDataLoaderStateCallback",
    }
    assert "every_n_sample_reg" not in config.trainer.callbacks
    assert "every_n_sample_ema" not in config.trainer.callbacks
