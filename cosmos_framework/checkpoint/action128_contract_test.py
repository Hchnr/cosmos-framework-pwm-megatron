"""Read Action128 contracts from full, model-only and legacy checkpoints on CPU."""

import json
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.distributed.checkpoint as dcp
import yaml

from cosmos_framework.checkpoint.action128 import read_source_contract
from cosmos_framework.utils.easy_io import easy_io


def _source(path, *, backend_key=None):
    return SimpleNamespace(path=str(path), backend_key=backend_key)


def _checkpointer():
    return SimpleNamespace(get_storage_reader=lambda path, source: dcp.FileSystemReader(path))


def _contract(name, mode="action128_seam"):
    return {"version": 1, "spec": {"name": name}, "vision_spatial_position_mode": mode}


def _write_config(path, contract, *, omit_spatial_mode=False):
    model = {"action128_spec": contract["spec"]}
    if not omit_spatial_mode:
        model["diffusion_expert_config"] = {"vision_spatial_position_mode": contract["vision_spatial_position_mode"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"model": {"config": model}}))


@pytest.mark.parametrize("trainer_layout", ["missing", "empty_directory", "without_contract"])
def test_model_only_checkpoint_reads_local_config(tmp_path, trainer_layout):
    checkpoint = tmp_path / "model_only"
    expected = _contract("initialized_state32")
    _write_config(checkpoint / "config.yaml", expected)
    if trainer_layout == "empty_directory":
        (checkpoint / "trainer").mkdir()
    elif trainer_layout == "without_contract":
        dcp.save({"iteration": 10}, checkpoint_id=checkpoint / "trainer", no_dist=True)
    assert read_source_contract(_checkpointer(), _source(checkpoint)) == expected


def test_immutable_trainer_contract_takes_priority_over_both_configs(tmp_path):
    checkpoint = tmp_path / "checkpoints/iter_000000010"
    expected = _contract("immutable")
    _write_config(checkpoint / "config.yaml", _contract("checkpoint_config"))
    _write_config(tmp_path / "config.yaml", _contract("run_config"))
    dcp.save({"action128_contract": json.dumps(expected)}, checkpoint_id=checkpoint / "trainer", no_dist=True)
    assert read_source_contract(_checkpointer(), _source(checkpoint)) == expected


def test_model_only_config_takes_priority_over_ancestor_config(tmp_path):
    checkpoint = tmp_path / "checkpoints/iter_000000010"
    expected = _contract("checkpoint_config")
    _write_config(checkpoint / "config.yaml", expected)
    _write_config(tmp_path / "config.yaml", _contract("unrelated_ancestor"))
    assert read_source_contract(_checkpointer(), _source(checkpoint)) == expected


@pytest.mark.parametrize("trainer_metadata", [False, True])
def test_legacy_run_config_remains_supported(tmp_path, trainer_metadata):
    checkpoint = tmp_path / "checkpoints/iter_000000010"
    expected = _contract("legacy", mode="canvas")
    _write_config(tmp_path / "config.yaml", expected, omit_spatial_mode=True)
    if trainer_metadata:
        dcp.save({"iteration": 10}, checkpoint_id=checkpoint / "trainer", no_dist=True)
    assert read_source_contract(_checkpointer(), _source(checkpoint)) == expected


def test_missing_contract_and_configs_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="original config.yaml"):
        read_source_contract(_checkpointer(), _source(tmp_path / "checkpoints/iter_000000010"))


def test_corrupt_existing_trainer_metadata_does_not_fall_back(tmp_path):
    checkpoint = tmp_path / "model_only"
    _write_config(checkpoint / "config.yaml", _contract("fallback_must_not_hide_corruption"))
    (checkpoint / "trainer").mkdir()
    (checkpoint / "trainer/.metadata").write_bytes(b"invalid pickle")
    with pytest.raises(pickle.UnpicklingError):
        read_source_contract(_checkpointer(), _source(checkpoint))


def test_existing_trainer_metadata_read_failure_propagates(tmp_path):
    checkpoint = tmp_path / "model_only"
    _write_config(checkpoint / "config.yaml", _contract("fallback_must_not_hide_permissions"))
    (checkpoint / "trainer").mkdir()
    (checkpoint / "trainer/.metadata").touch()
    failure = PermissionError("trainer metadata access denied")
    reader = Mock()
    reader.read_metadata.side_effect = failure
    checkpointer = SimpleNamespace(get_storage_reader=lambda path, source: reader)
    with pytest.raises(PermissionError) as raised:
        read_source_contract(checkpointer, _source(checkpoint))
    assert raised.value is failure


def test_invalid_immutable_contract_does_not_fall_back(tmp_path):
    checkpoint = tmp_path / "model_only"
    _write_config(checkpoint / "config.yaml", _contract("fallback_must_not_hide_invalid_contract"))
    dcp.save({"action128_contract": "invalid JSON"}, checkpoint_id=checkpoint / "trainer", no_dist=True)
    with pytest.raises(json.JSONDecodeError):
        read_source_contract(_checkpointer(), _source(checkpoint))


def test_model_only_config_uses_source_backend(monkeypatch):
    source = _source("s3://checkpoint-bucket/model_only", backend_key="source_backend")
    expected = _contract("remote")
    config_path = f"{source.path}/config.yaml"
    exists = Mock(side_effect=lambda path, *, backend_key: path == config_path)
    load = Mock(
        return_value=yaml.safe_dump(
            {
                "model": {
                    "config": {
                        "action128_spec": expected["spec"],
                        "diffusion_expert_config": {
                            "vision_spatial_position_mode": expected["vision_spatial_position_mode"]
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setattr(easy_io, "exists", exists)
    monkeypatch.setattr(easy_io, "load", load)
    checkpointer = SimpleNamespace(
        get_storage_reader=Mock(side_effect=AssertionError("Missing metadata must not be read"))
    )
    assert read_source_contract(checkpointer, source) == expected
    assert all(call.kwargs["backend_key"] == "source_backend" for call in exists.call_args_list)
    load.assert_called_once_with(config_path, file_format="txt", backend_key="source_backend")
