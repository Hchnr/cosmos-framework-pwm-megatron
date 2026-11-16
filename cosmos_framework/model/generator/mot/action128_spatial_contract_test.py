"""Actual DCP load hooks protect Action128 spatial coordinate semantics."""

import hashlib
import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from cosmos_framework.model.generator.mot.action128 import Action128Normalization
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec


def test_published_projection_families_are_validated():
    from cosmos_framework.data.generator.action.projection_routing import PUBLISHED_PROJECTION_FAMILY_IDS

    spec = tiny_action128_spec()
    spec["projection_routing"]["keys"].update(PUBLISHED_PROJECTION_FAMILY_IDS)
    spec["registry"]["domain_projection_ids"][0] = 63
    Action128Normalization(spec)
    key = next(iter(PUBLISHED_PROJECTION_FAMILY_IDS))
    spec["projection_routing"]["keys"][key] = 61
    with pytest.raises(ValueError, match="fixed head mapping"):
        Action128Normalization(spec)


def test_older_projection_registry_retains_checkpoint_digest():
    spec = tiny_action128_spec()
    spec["projection_routing"]["keys"] = {
        key: value for key, value in spec["projection_routing"]["keys"].items() if value <= 21
    }
    source = Action128Normalization(spec)
    target = Action128Normalization(spec)
    target.load_state_dict(source.state_dict())
    digest = hashlib.blake2b(json.dumps(spec, sort_keys=True).encode(), digest_size=16).digest()
    assert target.contract_digest.tolist() == list(digest)


@pytest.mark.parametrize("mutation", ["renumber", "unknown", "missing_used_slot"])
def test_projection_registry_rejects_incompatible_mapping(mutation):
    spec = tiny_action128_spec()
    keys = spec["projection_routing"]["keys"]
    if mutation == "renumber":
        keys["eef20/shared"] = 1
    elif mutation == "unknown":
        keys["joint128/unregistered"] = 63
    else:
        del keys["eef20/shared"]
    with pytest.raises(ValueError, match="fixed head mapping"):
        Action128Normalization(spec)


@pytest.mark.parametrize("mode", ["canvas", "action128_seam"])
def test_same_mode_dcp_roundtrip(tmp_path, mode):
    source = Action128Normalization(tiny_action128_spec(), mode)
    target = Action128Normalization(tiny_action128_spec(), mode)
    state = {"model": get_model_state_dict(source)}
    dcp.save(state, checkpoint_id=tmp_path / mode)
    loaded = {"model": get_model_state_dict(target)}
    dcp.load(loaded, checkpoint_id=tmp_path / mode)
    set_model_state_dict(target, loaded["model"], options=StateDictOptions(strict=False))
    target.validate_contract()
    torch.testing.assert_close(target.contract_digest, source.contract_digest)


@pytest.mark.parametrize("source_mode,target_mode", [("canvas", "action128_seam"), ("action128_seam", "canvas")])
def test_cross_mode_dcp_load_fails_before_any_forward(tmp_path, source_mode, target_mode):
    source = Action128Normalization(tiny_action128_spec(), source_mode)
    target = Action128Normalization(tiny_action128_spec(), target_mode)
    dcp.save({"model": get_model_state_dict(source)}, checkpoint_id=tmp_path / "checkpoint")
    loaded = {"model": get_model_state_dict(target)}
    dcp.load(loaded, checkpoint_id=tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="spatial RoPE mode"):
        set_model_state_dict(target, loaded["model"], options=StateDictOptions(strict=False))


def test_legacy_digest_is_unchanged_and_warm_start_keeps_initialized_contract():
    spec = tiny_action128_spec()
    legacy = Action128Normalization(spec)
    digest = hashlib.blake2b(json.dumps(spec, sort_keys=True).encode(), digest_size=16).digest()
    assert legacy.contract_digest.tolist() == list(digest)
    target = Action128Normalization(spec, "action128_seam")
    # Mirrors warm-start planner omitting every action_normalization key.
    target.load_state_dict({}, strict=False)
    target.validate_contract()
    assert not torch.equal(target.contract_digest, legacy.contract_digest)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_model_dtype_and_legacy_buffer_load_preserve_published_precision(tmp_path, dtype):
    spec = tiny_action128_spec()
    spec["transforms"][0]["normalization_mask"][0] = True
    spec["transforms"][0]["scale"][0] = 0.123456
    spec["transforms"][0]["bias"][0] = 0.234567
    source = Action128Normalization(spec, "action128_seam")
    expected = source.denormalize(torch.zeros(1, 128), 0)
    source.to(dtype=dtype)
    assert source.scale.dtype == source.bias.dtype == torch.float32
    torch.testing.assert_close(source.denormalize(torch.zeros(1, 128), 0), expected, rtol=0, atol=0)
    legacy = source.state_dict()
    legacy["scale"] = legacy["scale"].to(dtype)
    legacy["bias"] = legacy["bias"].to(dtype)
    dcp.save({"model": legacy}, checkpoint_id=tmp_path)
    with torch.device("meta"):
        target = Action128Normalization(spec, "action128_seam")
    target.to(dtype=dtype).to_empty(device="cpu")
    target.reset_contract()
    loaded = {"model": target.state_dict()}
    dcp.load(loaded, checkpoint_id=tmp_path)
    target.load_state_dict(loaded["model"])
    assert target.scale.dtype == target.bias.dtype == torch.float32
    torch.testing.assert_close(target.denormalize(torch.zeros(1, 128), 0), expected, rtol=0, atol=0)


def test_dtype_conversion_does_not_hide_bad_contract():
    module = Action128Normalization(tiny_action128_spec())
    module.contract_digest.zero_()
    module.to(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="checkpoint registry"):
        module.validate_contract()
