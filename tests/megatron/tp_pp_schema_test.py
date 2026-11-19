"""The production recipe preserves baseline compute/data and rejects silent drift."""

import copy
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"


def recipe():
    return tomllib.loads(RECIPE.read_text())


def test_tp_pp_recipe_keeps_baseline_compute_and_data_contract():
    raw = recipe()
    parsed = SFTExperimentConfig.model_validate(raw)
    baseline = tomllib.loads((RECIPE.parent / "action128_pretrain_state_h1_58src_192gpu.toml").read_text())
    model = copy.deepcopy(raw["model"])
    model.pop("parallelism")
    original_model = copy.deepcopy(baseline["model"])
    original_model.pop("parallelism")
    assert model == original_model
    assert raw["optimizer"] == baseline["optimizer"] and raw["scheduler"] == baseline["scheduler"]
    data = copy.deepcopy(raw["dataloader_train"])
    data.pop("action_state_normalization_paths")  # Existing production shell supplies this exact old State128 path.
    assert data == baseline["dataloader_train"]
    assert parsed.trainer.grad_accum_iter == 1
    assert parsed.model.parallelism.megatron.reference_micro_steps_per_update == baseline["trainer"]["grad_accum_iter"]
    assert parsed.model.parallelism.megatron.global_packed_microbatches_per_update == 576
    assert parsed.checkpoint.action128_preserve_source_normalization


@pytest.mark.parametrize("kind", ["hsdp", "cp", "double_ga", "partial_batch", "async", "model_only", "unknown"])
def test_tp_pp_schema_rejects_incompatible_runtime_semantics(kind):
    raw = recipe()
    if kind == "hsdp":
        raw["model"]["parallelism"]["data_parallel_shard_degree"] = 16
    elif kind == "cp":
        raw["model"]["parallelism"]["context_parallel_shard_degree"] = 2
    elif kind == "double_ga":
        raw["trainer"]["grad_accum_iter"] = 3
    elif kind == "partial_batch":
        raw["model"]["parallelism"]["megatron"]["global_packed_microbatches_per_update"] = 12
    elif kind == "async":
        raw["checkpoint"]["dcp_async_mode_enabled"] = True
    elif kind == "model_only":
        raw["checkpoint"]["load_training_state"] = False
    else:
        raw["model"]["parallelism"]["megatron"]["misspelled_tp"] = 4
    with pytest.raises(ValidationError):
        SFTExperimentConfig.model_validate(raw)
