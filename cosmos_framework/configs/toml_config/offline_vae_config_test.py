# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_catalog_discovery(monkeypatch):
    # Schema tests use placeholder paths; catalog discovery is tested separately.
    monkeypatch.setattr("cosmos_framework.configs.toml_config.sft_config._get_action128_roots", lambda datasets: [])


from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig, load_experiment_from_toml

_SHA = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
_FIXED_T2V_RECIPES = (
    Path(__file__).parents[3] / "examples" / "toml" / "sft_config" / "vision_sft_nano_t2v_4gpu_fixed_static.toml",
    Path(__file__).parents[3] / "examples" / "toml" / "sft_config" / "vision_sft_nano_t2v_8gpu_fixed_static.toml",
)


def test_standard_offline_catalog_does_not_require_producer_manifest() -> None:
    cfg = SFTExperimentConfig.model_validate(
        {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "dataloader_train": {
                "visual_generation": {"vae_mode": "offline"},
                "datasets": [{"path": "/catalog.parquet", "ratio": 1}],
            },
        }
    )
    assert cfg.dataloader_train.datasets[0].vae_manifest_path is None


def test_offline_mode_is_rejected_for_vlm() -> None:
    with pytest.raises(ValidationError, match="supported only for VFM"):
        SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vlm", "experiment": "reasoner_sft"},
                "dataloader_train": {"visual_generation": {"vae_mode": "offline"}},
            }
        )


def test_loader_derives_no_vae_and_removes_pixel_callbacks(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "dataset",
                "vae_name": "wan2pt2_tokenizer",
                "vae_tag": "wan2pt2_tokenizer-t480-test",
                "vae_weights_sha256": _SHA,
            }
        )
    )
    recipe = tmp_path / "offline.toml"
    recipe.write_text(
        f'''\
[job]
task = "vfm"
experiment = "vision_sft_nano"

[model]
load_vlm_visual = false

[model.tokenizer]
vae_path = "${{oc.env:THIS_ENV_MUST_NOT_BE_REQUIRED}}"

[trainer.callbacks.compile_tokenizer]
enabled = true
warmup_resolutions = ["256"]

[dataloader_train]
max_sequence_length = 45056

[dataloader_train.visual_generation]
vae_mode = "offline"
expected_vae_name = "wan2pt2_tokenizer"
expected_vae_weights_sha256 = "{_SHA}"

[[dataloader_train.datasets]]
path = "/catalog.parquet"
ratio = 1
vae_manifest_path = "{manifest}"
'''
    )

    config = load_experiment_from_toml(recipe)

    assert config.model.config.load_vision_tokenizer is False
    assert config.model.config.vision_vae_mode == "offline"
    assert config.dataloader_train.vae_mode == "offline"
    assert "compile_tokenizer" not in config.trainer.callbacks
    assert "every_n_sample_reg" not in config.trainer.callbacks
    assert "every_n_sample_ema" not in config.trainer.callbacks


@pytest.mark.parametrize("recipe", _FIXED_T2V_RECIPES, ids=lambda path: path.stem)
def test_fixed_static_recipe_accepts_one_offline_override(
    recipe: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_manifest = tmp_path / "action-manifest.json"
    koala_manifest = tmp_path / "koala-manifest.json"
    for path, dataset in ((action_manifest, "action100m"), (koala_manifest, "koala")):
        path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "vae_name": "wan2pt2_tokenizer",
                    "vae_tag": "wan2pt2_tokenizer-t480-test",
                    "vae_weights_sha256": _SHA,
                }
            )
        )
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/checkpoint")
    monkeypatch.setenv("ACTION100M_DATASET_PATH", "/action-catalog")
    monkeypatch.setenv("KOALA_DATASET_PATH", "/koala-catalog")
    monkeypatch.setenv("ACTION100M_VAE_MANIFEST_PATH", str(action_manifest))
    monkeypatch.setenv("KOALA_VAE_MANIFEST_PATH", str(koala_manifest))
    monkeypatch.delenv("WAN_VAE_PATH", raising=False)

    config = load_experiment_from_toml(
        recipe,
        extra_overrides=["dataloader_train.visual_generation.vae_mode=offline"],
    )

    assert config.model.config.load_vision_tokenizer is False
    assert config.model.config.vision_vae_mode == "offline"
    assert config.dataloader_train.vae_mode == "offline"
    assert config.dataloader_train.expected_vae_weights_sha256 == _SHA
    assert [item["vae_manifest_path"] for item in config.dataloader_train.dataset_specs] == [
        str(action_manifest),
        str(koala_manifest),
    ]


def test_usr_recipe_resolves_production_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USR_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("LOCAL_RANK", "3")
    recipe = Path(__file__).parents[3] / "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static.toml"
    config = load_experiment_from_toml(recipe)
    assert config.job.project == "cosmos3-t2v"
    assert config.job.name == "nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga4-cfg01-cond731"
    assert config.trainer.grad_accum_iter == 4
    assert config.optimizer.lr == 1e-4
    import tomllib

    assert config.checkpoint.save_iter == tomllib.loads(recipe.read_text())["checkpoint"]["save_iter"]
    assert (
        config.model.config.vlm_config.tokenizer.pretrained_model_name
        == "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct"
    )
    assert config.checkpoint.load_training_state is False
    assert config.checkpoint.load_path == "/share/project/eai_pwm/models/Cosmos3-Nano-Qwen3-VL-8B-Instruct-Hybrid-DCP"
    assert config.trainer.run_validation is False
    assert config.dataloader_train.cfg_drop_ratio == 0.1
    assert dict(config.dataloader_train.conditioning_config) == pytest.approx({0: 7 / 11, 1: 3 / 11, 2: 1 / 11})
    assert config.dataloader_train.max_samples_per_batch is None
    fixed = config.model.config.fixed_sequence_length
    assert fixed.und_tokens == 2176
    assert fixed.gen_tokens == 65536
    assert fixed.und_tokens + fixed.gen_tokens == config.dataloader_train.max_sequence_length
    assert config.dataloader_train.max_caption_tokens == 2048
    assert len(config.dataloader_train.dataset_specs) == 12
    assert config.model.config.load_vision_tokenizer is False
    assert "every_n_sample_reg" not in config.trainer.callbacks
    assert "every_n_sample_ema" not in config.trainer.callbacks
    assert config.trainer.compile_cache.path.endswith("/rank3")


def test_usr_high_capacity_recipe_resolves_v21_training_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USR_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCAL_RANK", "3")
    recipe = (
        Path(__file__).parents[3]
        / "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml"
    )
    config = load_experiment_from_toml(recipe)

    assert config.job.name == (
        "nano-usr-v2.1-r7-1056-resume23000-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541"
    )
    assert config.model.config.parallelism.data_parallel_shard_degree == 16
    assert config.trainer.grad_accum_iter == 2
    assert config.model.config.compile.mix_order_reduction_split_size == 32
    fixed = config.model.config.fixed_sequence_length
    assert fixed.und_tokens == 3_072
    assert fixed.gen_tokens == 98_304
    assert fixed.und_tokens + fixed.gen_tokens == 101_376
    assert config.model.config.max_num_tokens_after_packing == 101_376
    assert config.dataloader_train.max_sequence_length == 101_376
    assert (
        config.model.config.vlm_config.tokenizer.pretrained_model_name
        == "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct"
    )
    assert config.checkpoint.load_training_state is True
    assert config.checkpoint.keys_not_to_resume == []
    assert config.checkpoint.load_path == "/share/project/eai_pwm/models/Cosmos3-Nano-USR-v2.1-iter23000-Resume-DCP"
    assert len(config.dataloader_train.dataset_specs) == 29
    assert all(spec["path"].startswith(f"{tmp_path}/") for spec in config.dataloader_train.dataset_specs)
    assert sum(spec["ratio"] for spec in config.dataloader_train.dataset_specs) == 4_246_463

    import tomllib

    raw_dataset_paths = [
        dataset["path"] for dataset in tomllib.loads(recipe.read_text())["dataloader_train"]["datasets"]
    ]
    replacement_root = (
        "/share/project/liyue/data/parquet-data/train-usr-v2.1-20260928/full-r3-80-r6b-5-r6-15-epochs4-seed0"
    )
    assert all(replacement_root in path for path in raw_dataset_paths)


@pytest.mark.parametrize("explicit_overrides", [False, True])
def test_usr_launcher_uses_toml_defaults_unless_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_overrides: bool
) -> None:
    import os
    import subprocess

    root = Path(__file__).parents[3]
    recipe = tmp_path / "edited.toml"
    recipe.write_text(
        (root / "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static.toml")
        .read_text()
        .replace('resolution = "480"', 'resolution = "256"')
        .replace("max_iter                = 100000", "max_iter                = 17")
    )
    for key in ("MAX_ITER", "TRAIN_RESOLUTION", "HF_MODEL_PATH", "BASE_CHECKPOINT_PATH", "USR_DATASET_ROOT"):
        monkeypatch.delenv(key, raising=False)
    capture = tmp_path / "args.txt"
    stub = tmp_path / "torchrun"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGV_CAPTURE"\n')
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "VIRTUAL_ENV": str(tmp_path),
        "NNODES": "1",
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "SFT_TOML_FILE": str(recipe),
        "ARGV_CAPTURE": str(capture),
        "WANDB_API_KEY": "",
        "EXTRA_TAIL_OVERRIDES": "",
        "DETAILED_NVTX_GRANULARITY": "",
    }
    command = ["bash", str(root / "examples/launch_sft_vision_nano_usr_128gpu.sh")]
    if explicit_overrides:
        env.update(MAX_ITER="2", TRAIN_RESOLUTION="480", EXTRA_TAIL_OVERRIDES="trainer.max_iter=3")
        command.append("trainer.max_iter=4")
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    args = capture.read_text().splitlines()
    overrides = args[args.index("--") + 1 :] if "--" in args else []
    if not explicit_overrides:
        assert overrides == []
    config = load_experiment_from_toml(recipe, extra_overrides=overrides)
    assert config.trainer.max_iter == (4 if explicit_overrides else 17)
    assert config.model.config.resolution == ("480" if explicit_overrides else "256")
    assert config.dataloader_train.resolution == config.model.config.resolution
    assert config.trainer.grad_accum_iter == 4
    assert (
        config.model.config.vlm_config.tokenizer.pretrained_model_name
        == "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct"
    )
    assert config.checkpoint.load_path == "/share/project/eai_pwm/models/Cosmos3-Nano-Qwen3-VL-8B-Instruct-Hybrid-DCP"
    assert config.dataloader_train.dataset_specs[0].path.endswith("full-seed0/agibot-world")
