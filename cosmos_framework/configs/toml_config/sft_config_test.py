# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Loader tests for the free-form ``[custom]`` escape-hatch section of the SFT TOML."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import (
    SFTExperimentConfig,
    _resolve_catalog_datasets_for_inspection,
    load_experiment_from_toml,
)
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

# Representative payload: scalars, a nested sub-table, and an array-of-tables.
_CUSTOM_PAYLOAD = {
    "scalar_int": 5,
    "scalar_str": "hello",
    "flag": True,
    "ratio": 0.3,
    "sampling": {"bug_ratio": 0.3, "nested": {"deep": 1}},
    "items": [
        {"path": "/data/a", "weight": 1.0},
        {"path": "/data/b", "weight": 2.0},
    ],
}


def test_catalog_inspection_resolves_env_without_mutating_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_CATALOG_ROOT", "/tmp/catalog-root")
    datasets = [{"path": "${oc.env:TEST_CATALOG_ROOT}/sample", "ratio": 3}]

    resolved = _resolve_catalog_datasets_for_inspection(datasets)

    assert resolved == [{"path": "/tmp/catalog-root/sample", "ratio": 3}]
    assert datasets == [{"path": "${oc.env:TEST_CATALOG_ROOT}/sample", "ratio": 3}]


# --------------------------------------------------------------------------- #
# 1. pydantic schema validation                                               #
# --------------------------------------------------------------------------- #
class TestSchemaValidation:
    @pytest.mark.parametrize("task", ["vfm", "vlm"])
    def test_periodic_checkpoint_retention_reaches_hydra(self, task: str) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": task, "experiment": "vision_sft_nano"},
                "checkpoint": {"save_iter": 200, "keep_last_n": 3, "keep_every_n_steps": 2000},
            }
        )
        overrides = build_hydra_overrides(cfg.model_dump(exclude_unset=True))
        assert "checkpoint.keep_last_n=3" in overrides
        assert "checkpoint.keep_every_n_steps=2000" in overrides
        assert "checkpoint.save_iter=200" in overrides

    @pytest.mark.parametrize("interval", [0, -2000, True, 2000.0, "2000"])
    def test_periodic_checkpoint_retention_requires_positive_integer(self, interval) -> None:
        with pytest.raises(ValidationError, match="keep_every_n_steps"):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "checkpoint": {"keep_every_n_steps": interval},
                }
            )

    def test_periodic_checkpoint_retention_defaults_disabled(self) -> None:
        cfg = SFTExperimentConfig.model_validate({"job": {"task": "vfm", "experiment": "vision_sft_nano"}})
        assert cfg.checkpoint.keep_every_n_steps is None
        assert all(
            "keep_every_n_steps" not in item for item in build_hydra_overrides(cfg.model_dump(exclude_unset=True))
        )

    def test_custom_section_validates_arbitrary_nested_content(self) -> None:
        """Arbitrary nested [custom] content passes through untouched."""
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "custom": _CUSTOM_PAYLOAD,
        }
        cfg = SFTExperimentConfig.model_validate(raw)
        # The framework stores it verbatim — no coercion, no inner validation.
        assert cfg.custom == _CUSTOM_PAYLOAD

    def test_no_custom_section_defaults_empty(self) -> None:
        cfg = SFTExperimentConfig.model_validate({"job": {"task": "vfm", "experiment": "vision_sft_nano"}})
        assert cfg.custom == {}

    def test_unknown_top_level_key_raises(self) -> None:
        """Any unknown top-level section that is NOT `custom` still raises."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "bogus_section": {"x": 1},
                }
            )

    def test_unknown_key_inside_optimizer_raises(self) -> None:
        """A typo inside a KNOWN section is still a hard error (extra='forbid')."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "optimizer": {"lr": 1.0e-4, "not_a_real_key": 1},
                }
            )

    def test_custom_does_not_loosen_sibling_validation(self) -> None:
        """Presence of [custom] must not relax extra='forbid' elsewhere."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "custom": _CUSTOM_PAYLOAD,
                    "trainer": {"max_iter": 10, "typo_here": True},
                }
            )

    def test_compile_cache_schema_accepts_configured_path(self) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                "trainer": {
                    "compile_cache": {
                        "enabled": True,
                        "path": ".compile_cache/custom",
                    }
                },
            }
        )
        assert cfg.trainer.compile_cache.path == ".compile_cache/custom"

    def test_trainer_compile_limits_reach_hydra(self) -> None:
        from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "action_policy_droid_nano"},
                "trainer": {"compile_config": {"recompile_limit": 128, "use_duck_shape": False}},
            }
        )
        overrides = build_hydra_overrides(cfg.model_dump(exclude_unset=True))
        assert "trainer.compile_config.recompile_limit=128" in overrides
        assert "trainer.compile_config.use_duck_shape=false" in overrides

    @pytest.mark.parametrize("limit", [0, -1])
    def test_trainer_compile_limit_must_be_positive(self, limit: int) -> None:
        with pytest.raises(ValidationError, match="recompile_limit"):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "action_policy_droid_nano"},
                    "trainer": {"compile_config": {"recompile_limit": limit}},
                }
            )

    def test_compile_tokenizer_schema_accepts_aspect_ratio_filter(self) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "vision_sft_edge"},
                "trainer": {
                    "callbacks": {
                        "compile_tokenizer": {
                            "enabled": True,
                            "warmup_resolutions": ["256"],
                            "aspect_ratio": "1,1",
                        }
                    }
                },
            }
        )
        assert cfg.trainer.callbacks.compile_tokenizer.aspect_ratio == "1,1"

    def test_fixed_sequence_static_compile_contract_accepts_valid_vfm(self) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "vision_sft_edge"},
                "model": {
                    "max_num_tokens_after_packing": 45056,
                    "fixed_sequence_length": {
                        "enabled": True,
                        "und_tokens": 19456,
                        "gen_tokens": 25600,
                    },
                    "compile": {
                        "enabled": True,
                        "compile_dynamic": False,
                        "compiled_region": "language",
                    },
                },
                "dataloader_train": {"max_sequence_length": 45056},
            }
        )
        assert cfg.model.fixed_sequence_length.und_tokens == 19456

    @pytest.mark.parametrize(
        "fixed",
        [
            {"enabled": True, "understanding_tokens": 45056},
            {"enabled": True, "understanding_tokens": 45056, "und_tokens": 19456, "gen_tokens": 25600},
        ],
    )
    def test_fixed_sequence_accepts_understanding_profiles_and_secondary_sample_cap(
        self, fixed: dict[str, object]
    ) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "vision_sft_edge"},
                "model": {
                    "max_num_tokens_after_packing": 45056,
                    "fixed_sequence_length": fixed,
                },
                "dataloader_train": {
                    "max_sequence_length": 45056,
                    "max_samples_per_batch": 48,
                },
            }
        )

        assert cfg.dataloader_train.max_samples_per_batch == 48

    def test_vfm_variable_sequence_static_compile_rejected(self) -> None:
        with pytest.raises(ValidationError, match="fixed_sequence_length.enabled=true"):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_edge"},
                    "model": {"compile": {"enabled": True, "compile_dynamic": False}},
                }
            )

    def test_fixed_sequence_rejects_sample_count_batching(self) -> None:
        with pytest.raises(ValidationError, match="max_sequence_length token-budget mode"):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "model": {
                        "max_num_tokens_after_packing": 34,
                        "fixed_sequence_length": {"enabled": True, "und_tokens": 25, "gen_tokens": 9},
                    },
                    "dataloader_train": {"max_samples_per_batch": 2},
                }
            )

    def test_vlm_static_compile_remains_outside_fixed_sequence_contract(self) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vlm", "experiment": "reasoner_sft"},
                "model": {"compile": {"enabled": True, "compile_dynamic": False}},
            }
        )
        assert not cfg.model.compile.compile_dynamic

    @pytest.mark.parametrize("removed_key", ["mode", "shared_dir", "local_dir"])
    def test_compile_cache_schema_rejects_removed_fields(self, removed_key: str) -> None:
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "trainer": {"compile_cache": {"enabled": True, removed_key: "removed"}},
                }
            )

    def test_catalog_worker_options_validate(self) -> None:
        cfg = SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                "dataloader_train": {
                    "datasets": [{"path": "/tmp/catalog.parquet", "ratio": 1}],
                    "num_workers": 2,
                    "prefetch_factor": 2,
                    "pin_memory": False,
                },
            }
        )

        assert cfg.dataloader_train.num_workers == 2
        assert cfg.dataloader_train.prefetch_factor == 2
        assert cfg.dataloader_train.pin_memory is False


# --------------------------------------------------------------------------- #
# 2. build_hydra_overrides must NOT emit [custom] as per-leaf overrides        #
# --------------------------------------------------------------------------- #
class TestBuildHydraOverrides:
    def test_custom_not_emitted_as_overrides(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "optimizer": {"lr": 1.0e-5},
            "custom": _CUSTOM_PAYLOAD,
        }
        overrides = build_hydra_overrides(raw)
        # Nothing under custom (verbatim or remapped) should appear.
        assert all("custom" not in o for o in overrides), overrides

    def test_other_keys_still_emitted(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "optimizer": {"lr": 1.0e-5},
            "custom": {"a": 1},
        }
        overrides = build_hydra_overrides(raw)
        assert "experiment=vision_sft_nano" in overrides
        assert any(o.startswith("optimizer.lr=") for o in overrides), overrides

    def test_fixed_sequence_config_routes_to_vfm_and_is_skipped_for_vlm(self) -> None:
        model = {
            "fixed_sequence_length": {
                "enabled": True,
                "understanding_tokens": 45056,
                "und_tokens": 19456,
                "gen_tokens": 25600,
            }
        }
        vfm = build_hydra_overrides({"job": {"task": "vfm", "experiment": "vision_sft_edge"}, "model": model})
        vlm = build_hydra_overrides({"job": {"task": "vlm", "experiment": "reasoner_sft"}, "model": model})

        assert "model.config.fixed_sequence_length.enabled=true" in vfm
        assert "model.config.fixed_sequence_length.understanding_tokens=45056" in vfm
        assert all("fixed_sequence_length" not in override for override in vlm)


# --------------------------------------------------------------------------- #
# 3. end-to-end load_experiment_from_toml on the shipped vision_sft_nano recipe #
# --------------------------------------------------------------------------- #
_BASE_TOML = """\
[job]
task         = "vfm"
experiment   = "vision_sft_nano"
project      = "cosmos3"
group        = "sft"
name         = "sft_config_custom_test"
wandb_mode   = "disabled"

[model.tokenizer]
vae_path = "${oc.env:WAN_VAE_PATH}"

[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
"""

_CUSTOM_TOML_BLOCK = """\

[custom]
scalar_int = 5
scalar_str = "hello"
flag       = true
ratio      = 0.3

[custom.sampling]
bug_ratio = 0.3

[custom.sampling.nested]
deep = 1

[[custom.items]]
path   = "/data/a"
weight = 1.0

[[custom.items]]
path   = "/data/b"
weight = 2.0
"""


def _load_or_skip(toml_path: Path):
    """Run the real loader, skipping if the training stack can't be imported."""
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    try:
        return load_experiment_from_toml(str(toml_path))
    except ImportError as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"training stack not importable here: {exc!r}")


@pytest.fixture
def _dummy_recipe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # vision_sft_nano interpolates these env vars into path strings at resolve time.
    monkeypatch.setenv("DATASET_PATH", "/tmp/dummy_dataset")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/dummy_vae.pth")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/dummy_ckpt")


@pytest.fixture(autouse=True)
def _composition_catalog_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Composition tests use placeholder paths, not physical training catalogs.

    Real Action128 publication gates and spec injection are exercised by
    data/test/action128_test.py, including the production three-tower recipe.
    Keep file I/O out of these isolated configuration/override tests.
    """
    monkeypatch.setattr(
        "cosmos_framework.configs.toml_config.sft_config._get_action128_roots",
        lambda datasets: [],
    )


class TestEndToEndLoader:
    def test_production_periodic_checkpoint_retention(self, monkeypatch, _dummy_recipe_env):
        monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
        config = _load_or_skip(Path("examples/toml/sft_config/action128_wam_nano_160gpu.toml"))
        assert config.checkpoint.save_iter == 200
        assert config.checkpoint.keep_last_n == 3
        assert config.checkpoint.keep_every_n_steps == 2000

    def test_production_recipe_local_shape_overrides(self, monkeypatch, _dummy_recipe_env):
        from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

        monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
        config = load_experiment_from_toml(
            "examples/toml/sft_config/action_policy_1b.toml",
            extra_overrides=[
                "model.config.parallelism.data_parallel_shard_degree=2",
                "trainer.max_iter=2",
                "dataloader_train.max_samples_per_batch=4",
                "model.config.fixed_sequence_length.und_tokens=512",
                "model.config.fixed_sequence_length.gen_tokens=2816",
                "model.config.fixed_sequence_length.action_tokens=128",
                "model.config.max_num_tokens_after_packing=3328",
                "dataloader_train.max_sequence_length=3328",
                "dataloader_train.num_workers=2",
            ],
        )
        assert config.model.config.fixed_sequence_length.gen_tokens == 2816
        assert config.model.config.fixed_sequence_length.action_tokens == 128
        assert config.model.config.max_num_tokens_after_packing == 3328
        assert config.dataloader_train.fixed_gen_tokens == 2816
        assert config.dataloader_train.fixed_und_tokens == 512
        assert config.dataloader_train.max_sequence_length == 3328
        assert config.dataloader_train.max_samples_per_batch == 4
        assert config.dataloader_train.num_workers == 2

    def test_local_shape_override_rejects_mismatched_loader_budget(self):
        from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

        with pytest.raises(ValidationError, match="largest enabled fixed profile"):
            load_experiment_from_toml(
                "examples/toml/sft_config/action_policy_1b.toml",
                extra_overrides=["dataloader_train.max_sequence_length=3328"],
            )

    def test_large_pack_ac_recipe(self, monkeypatch, _dummy_recipe_env):
        monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
        config = _load_or_skip(Path("examples/toml/sft_config/action_policy_1b.toml"))
        model = config.model.config
        assert model.action_tower_preset == "width_only_1b"
        assert model.action_tower_attention_backend == "flash3_video"
        assert model.activation_checkpointing.mode == "full"
        assert model.fixed_sequence_length.enabled
        assert model.fixed_sequence_length.und_tokens == 11776
        assert model.fixed_sequence_length.gen_tokens == 62592
        assert model.fixed_sequence_length.action_tokens == 2944
        assert config.dataloader_train.max_samples_per_batch == 92
        assert config.dataloader_train.max_sequence_length == 74368
        assert config.trainer.grad_accum_iter == 1
        assert config.optimizer.lr == 0.00002

    def test_production_action_1b_recipe(self, monkeypatch, _dummy_recipe_env):
        monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
        config = _load_or_skip(Path("examples/toml/sft_config/action_policy_1b.toml"))
        assert config.model.config.action_tower_preset == "width_only_1b"
        assert config.model.config.action_tower_init == "checkpoint"
        assert config.model.config.action_tower_attention_backend == "flash3_video"
        assert config.model.config.parallelism.data_parallel_shard_degree == -1
        assert config.trainer.grad_accum_iter == 1
        assert config.dataloader_train.num_workers == 18
        assert config.checkpoint.save_iter == 1000
        assert config.trainer.logging_iter == 1
        assert config.job.wandb_mode == "online"

    def test_action_tower_recipe_adds_new_lr_patterns(
        self, tmp_path: Path, monkeypatch, _dummy_recipe_env: None
    ) -> None:
        monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
        recipe = Path("examples/toml/sft_config/action_policy_1b.toml")
        path = tmp_path / "new_pattern.toml"
        path.write_text(
            recipe.read_text().replace("lr_multipliers = {", "lr_multipliers = { action_time_embedder = 2.0,")
        )
        config = _load_or_skip(path)
        assert config.optimizer.lr_multipliers["action_time_embedder"] == 2.0
        assert config.model.config.tower_structure == "three_tower"
        assert config.model.config.action_tower_attention_backend == "flash3_video"
        assert config.model.config.action_tower_projection_mode == "routed"
        assert config.model.config.action_tower_execution == "native"

    def test_load_with_custom_section(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "with_custom.toml"
        toml_path.write_text(_BASE_TOML + _CUSTOM_TOML_BLOCK)

        config = _load_or_skip(toml_path)

        expected = {
            "scalar_int": 5,
            "scalar_str": "hello",
            "flag": True,
            "ratio": 0.3,
            "sampling": {"bug_ratio": 0.3, "nested": {"deep": 1}},
            "items": [
                {"path": "/data/a", "weight": 1.0},
                {"path": "/data/b", "weight": 2.0},
            ],
        }
        # Injected verbatim as a plain dict after Hydra resolution, so a project
        # can run MyProjectConfig.model_validate(config.custom) directly.
        assert config.custom == expected

    def test_load_without_custom_section_defaults_empty(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "no_custom.toml"
        toml_path.write_text(_BASE_TOML)

        config = _load_or_skip(toml_path)

        assert config.custom == {}

    def test_catalog_cli_override_updates_injected_loader(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "catalog_override.toml"
        toml_path.write_text(
            _BASE_TOML
            + """

[dataloader_train]
max_samples_per_batch = 32

[[dataloader_train.datasets]]
path = "/tmp/catalog.parquet"
ratio = 1
"""
        )

        config = load_experiment_from_toml(
            toml_path,
            extra_overrides=["dataloader_train.max_samples_per_batch=1"],
        )

        assert config.dataloader_train.max_samples_per_batch == 1


# --------------------------------------------------------------------------- #
# 4. Every shipped example TOML validates + builds overrides                   #
# --------------------------------------------------------------------------- #
_EXAMPLE_TOML_DIR = Path(__file__).parents[3] / "examples" / "toml" / "sft_config"
_EXAMPLE_TOMLS = sorted(_EXAMPLE_TOML_DIR.glob("*.toml"))


class TestExampleTomlConfigs:
    """Every TOML shipped under ``examples/toml/sft_config/`` must both validate
    against ``SFTExperimentConfig`` (``extra="forbid"`` rejects any block the
    schema does not register) and produce a Hydra override list. Catches, e.g., a
    recipe TOML carrying a ``[model.*]`` / ``[dataloader_train.*]`` sub-block the
    schema never modelled, which would raise ``ValidationError`` at load time.
    """

    def test_examples_dir_is_nonempty(self) -> None:
        assert _EXAMPLE_TOMLS, f"no example TOMLs found under {_EXAMPLE_TOML_DIR}"

    @pytest.mark.parametrize("toml_path", _EXAMPLE_TOMLS, ids=lambda p: p.name)
    def test_example_toml_validates_and_builds_overrides(self, toml_path: Path) -> None:
        import tomllib

        raw = tomllib.loads(toml_path.read_text())

        # 1. Schema validation — extra="forbid" flags un-registered TOML blocks.
        try:
            SFTExperimentConfig.model_validate(raw)
        except ValidationError as exc:  # pragma: no cover - failure detail
            pytest.fail(f"{toml_path.name} failed schema validation:\n{exc}")

        # 2. Override construction — every leaf must route through PATH_REMAPS.
        overrides = build_hydra_overrides(raw)
        assert overrides[0] == "--"
        assert any(o.startswith("experiment=") for o in overrides), overrides


def test_qwen4b_gen_recipe_builds_catalog_and_selects_only_gen(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Qwen4B recipe must use Parquet and keep the UND expert frozen."""
    monkeypatch.setenv("DATASET_PATH", "/tmp/bridge.parquet")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/Wan2.2_VAE.pth")
    monkeypatch.setenv("QWEN3_VL_4B_PATH", "/tmp/Qwen3-VL-4B-Instruct")
    recipe = _EXAMPLE_TOML_DIR / "vision_sft_qwen4b_gen.toml"

    config = _load_or_skip(recipe)

    # Catalog metadata intentionally preserves the OmegaConf interpolation;
    # Hydra resolves it when the LazyCall is instantiated.
    assert config.catalog_datasets == [{"path": "${oc.env:DATASET_PATH}", "ratio": 1}]
    target = config.dataloader_train.get("_target_")
    target_name = getattr(target, "__name__", str(target).rsplit(".", 1)[-1])
    assert target_name == "build_catalog_training_dataloader"
    assert config.optimizer.keys_to_select == ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]
    assert config.dataloader_train.conditioning_config == {0: 1.0, 1: 0.0, 2: 0.0}


def test_nano_t2v_recipe_uses_catalog_worker_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/Wan2.2_VAE.pth")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/Cosmos3-Nano-DCP")
    monkeypatch.setenv("ACTION100M_DATASET_PATH", "/tmp/action100m")
    monkeypatch.setenv("KOALA_DATASET_PATH", "/tmp/koala-36m")
    recipe = _EXAMPLE_TOML_DIR / "vision_sft_nano_t2v_8gpu.toml"

    config = _load_or_skip(recipe)

    assert config.catalog_datasets == [
        {"path": "${oc.env:ACTION100M_DATASET_PATH}", "ratio": 16},
        {"path": "${oc.env:KOALA_DATASET_PATH}", "ratio": 25},
    ]
    assert config.dataloader_train.num_workers == 6
    assert config.dataloader_train.prefetch_factor == 4
    assert config.dataloader_train.pin_memory is False
    assert config.dataloader_train.media_config.max_frame_pixels == 33_554_432


@pytest.mark.parametrize(
    "recipe_name",
    [
        "vision_sft_nano_t2v_4gpu_fixed_static.toml",
        "vision_sft_nano_t2v_8gpu_fixed_static.toml",
    ],
)
def test_nano_t2v_fixed_static_recipe_wires_catalog_admission(
    recipe_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/Wan2.2_VAE.pth")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/Cosmos3-Nano-DCP")
    monkeypatch.setenv("ACTION100M_DATASET_PATH", "/tmp/action100m")
    monkeypatch.setenv("KOALA_DATASET_PATH", "/tmp/koala-36m")
    recipe = _EXAMPLE_TOML_DIR / recipe_name

    config = _load_or_skip(recipe)

    assert config.model.config.compile.compile_dynamic is False
    assert config.model.config.fixed_sequence_length.enabled is True
    assert config.dataloader_train.fixed_sequence_length_enabled is True
    assert config.dataloader_train.fixed_understanding_tokens == 0
    assert config.dataloader_train.fixed_und_tokens == 1_086
    assert config.dataloader_train.fixed_gen_tokens == 64_450
    assert config.trainer.compile_cache.enabled is True
    assert config.trainer.callbacks.compile_tokenizer.enabled is True
    scheduler_lengths = [
        len(config.scheduler.cycle_lengths),
        len(config.scheduler.f_max),
        len(config.scheduler.f_min),
        len(config.scheduler.f_start),
        len(config.scheduler.warm_up_steps),
    ]
    assert len(set(scheduler_lengths)) == 1
    assert sum(config.scheduler.cycle_lengths) >= config.trainer.max_iter


def test_nano_t2v_launch_files_do_not_embed_personal_paths() -> None:
    examples_dir = _EXAMPLE_TOML_DIR.parents[1]
    launch_files = [
        examples_dir / "launch_sft_vision_nano_t2v_8gpu.sh",
        examples_dir / "launch_sft_vision_nano_t2v_128gpu.sh",
        _EXAMPLE_TOML_DIR / "vision_sft_nano_t2v_8gpu.toml",
    ]

    for launch_file in launch_files:
        assert "/share/project/liyue" not in launch_file.read_text()


def test_full_gen_is_not_a_training_configuration():
    with pytest.raises(ValidationError, match="action_tower_projection_mode"):
        SFTExperimentConfig.model_validate(
            {
                "job": {"task": "vfm", "experiment": "action_policy_droid_nano"},
                "model": {"tower_structure": "three_tower", "action_tower_projection_mode": "full_gen"},
            }
        )


@pytest.mark.parametrize(
    ("recipe", "expected_weight"),
    [
        ("action128_wam_nano_160gpu", 1.0),
        ("action128_wam_nano_2tower_160gpu", 1.0),
        ("action_policy_1b", 10.0),
    ],
)
def test_action_modality_weights_reach_runtime_config(recipe, expected_weight, monkeypatch, _dummy_recipe_env):
    monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
    config = load_experiment_from_toml(f"examples/toml/sft_config/{recipe}.toml")
    rf = config.model.config.rectified_flow_training_config
    assert rf.loss_scale == expected_weight
    assert rf.action_loss_weight == expected_weight


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_modality_weights_reject_invalid_values(value):
    for key in ("loss_scale", "action_loss_weight"):
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate({"model": {"rectified_flow_training_config": {key: value}}})


def test_shared_prefetch_fields_are_injected_after_hydra_and_validate_topology():
    from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

    payload = {
        "job": {"task": "vfm", "experiment": "vision_sft_nano"},
        "dataloader_train": {
            "datasets": [{"path": "/tmp/catalog.parquet", "ratio": 1}],
            "execution_backend": "shared_prefetch",
            "logical_workers": [2],
            "worker_assignment": [0, 0],
            "worker_timeout": 120,
            "microbatch_timeout_sec": 20,
            "microbatch_timeout_max_skips": 2,
            "receive_in_background": False,
            "microbatch_skip_log_dir": "/tmp/skipped_samples",
            "num_workers": 1,
            "prefetch_factor": 2,
        },
    }
    cfg = SFTExperimentConfig.model_validate(payload)
    assert cfg.dataloader_train.logical_workers == [2]
    overrides = build_hydra_overrides(payload)
    assert not any(
        "worker_assignment" in item
        or "execution_backend" in item
        or "logical_workers" in item
        or "microbatch_" in item
        or "receive_in_background" in item
        for item in overrides
    )
    for value in ([-1], [True], [1.5]):
        payload["dataloader_train"]["logical_workers"] = value
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(payload)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_microbatch_timeout_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        SFTExperimentConfig.model_validate({"dataloader_train": {"microbatch_timeout_sec": value}})


def test_production_58src_microbatch_deadline_reaches_factory(monkeypatch, _dummy_recipe_env):
    monkeypatch.setenv("HF_MODEL_PATH", "/tmp/dummy_tokenizer")
    monkeypatch.setenv("OUTPUT_ROOT", "/tmp/58src-output")
    config = load_experiment_from_toml(
        "examples/toml/sft_config/action128_pretrain_state_h1_58src_192gpu.toml",
        extra_overrides=["dataloader_train.microbatch_timeout_sec=60"],
    )
    assert config.dataloader_train.microbatch_timeout_sec == 60
    assert config.dataloader_train.microbatch_timeout_max_skips == 0
    from omegaconf import OmegaConf

    assert OmegaConf.to_container(config.dataloader_train, resolve=True)["microbatch_skip_log_dir"] == (
        "/tmp/58src-output/logs/skipped_samples"
    )
