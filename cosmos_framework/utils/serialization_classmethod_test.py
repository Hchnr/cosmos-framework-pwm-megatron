"""A dry-run YAML must retain the concrete owner of inherited classmethods."""

from dataclasses import dataclass

import pytest
import yaml
from omegaconf import DictConfig

from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig
from cosmos_framework.utils.lazy_config import LazyCall, instantiate
from cosmos_framework.utils.serialization import _yaml_safe, to_qualitified_name, to_yaml


@dataclass
class ConfigContainer:
    config: DictConfig


@pytest.mark.parametrize("serialize", [_yaml_safe, to_qualitified_name])
def test_inherited_classmethod_owner(serialize):
    assert serialize(Qwen3VLMoTConfig.from_json_file).endswith(".Qwen3VLMoTConfig.from_json_file")


def test_dryrun_yaml_config_roundtrip():
    config = LazyCall(Qwen3VLMoTConfig.from_json_file)(
        json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    loaded = instantiate(yaml.safe_load(to_yaml(ConfigContainer(config)))["config"])
    assert isinstance(loaded, Qwen3VLMoTConfig)
    assert loaded.full_config.model_type == "qwen3_vl"
