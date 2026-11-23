# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""All-component canonical import format and source tensor reconstruction."""

import hashlib
import json
import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.convert_cosmos_checkpoint_to_megatron_tp_pp import convert
from tools.verify_cosmos_tp_pp_conversion import verify


def make_source(path, *, model_dtype=torch.float32, include_master=True, optimizer_edit=None, fractional=False):
    name = "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"
    row = "net.language_model.model.layers.1.self_attn.o_proj_moe_gen.weight"
    frozen = "net.language_model.model.embed_tokens.weight"
    base = torch.arange(96, dtype=torch.float32).view(8, 12)
    if fractional:
        base = base * 0.013 + 0.0007
    tensors = {
        name: base.to(model_dtype),
        row: (base + 2).to(model_dtype),
        frozen: torch.arange(24, dtype=torch.float32).view(6, 4).to(model_dtype),
    }
    optim = {}
    for key in (name, row):
        values = {
            "exp_avg": tensors[key].float() * 0.01,
            "exp_avg_sq": tensors[key].float() * 0.02 + 0.01,
        }
        if include_master:
            values["master_param"] = tensors[key].float() + 0.001
        for field, value in values.items():
            optim[f"state.{key}.{field}"] = value
        for field, value in {
            "lr": torch.tensor(1e-4),
            "initial_lr": torch.tensor(1e-4),
            "bias_correction": True,
            "betas": (0.9, 0.99),
            "eps": 1e-8,
            "weight_decay": 0.05,
            "step": torch.tensor([1200], dtype=torch.int32),
        }.items():
            optim[f"param_groups.{key}.{field}"] = value
    if optimizer_edit is not None:
        optimizer_edit(optim)
    for component, state in {
        "model": tensors,
        "optim": optim,
        "trainer": {"iteration": 1200},
        "scheduler": {"last_epoch": 1200},
    }.items():
        dcp.save(state, checkpoint_id=path / component)
    (path / "dataloader").mkdir()
    (path / "dataloader" / "rank_0.pkl").write_bytes(pickle.dumps({"global_id": 3600}))
    (path / ".save_complete").touch()
    return tensors, optim


def test_full_conversion_reconstructs_model_master_and_moments(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source)
    source_digest = hashlib.sha256((source / "model" / ".metadata").read_bytes()).hexdigest()
    manifest = convert(source, destination, tp=2, pp=2, layers=2)
    assert manifest["complete"] and (destination / ".save_complete").is_file()
    assert set(manifest["parameters"]) == set(tensors)
    for key, entry in manifest["parameters"].items():
        shards = [torch.load(destination / shard["path"], weights_only=False) for shard in entry["shards"]]
        fields = list(shards[0]["state"])
        for field in fields:
            pieces = [shard["state"][field] for shard in shards]
            actual = pieces[0] if entry["axis"] is None else torch.cat(pieces, dim=entry["axis"])
            expected = tensors[key] if field == "model" else optim[f"state.{key}.{field}"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if "master_param" in fields:
            assert not torch.equal(shards[0]["state"]["model"], shards[0]["state"]["master_param"])
            assert shards[0]["hyperparameters"]["step"].item() == 1200
    for component in ("trainer", "scheduler", "dataloader"):
        for path in (source / component).iterdir():
            assert path.read_bytes() == (destination / "logical_state" / component / path.name).read_bytes()
    assert hashlib.sha256((source / "model" / ".metadata").read_bytes()).hexdigest() == source_digest
    report = verify(destination)
    assert report["passed"] and report["full_conversion"]
    assert len(report["checked_parameters"]) == 3
    with pytest.raises(FileExistsError):
        convert(source, destination, tp=2, pp=2, layers=2)
    parameter = next(k for k, v in manifest["parameters"].items() if v["optimizer_fields"])
    shard_path = destination / manifest["parameters"][parameter]["shards"][0]["path"]
    corrupted = torch.load(shard_path, weights_only=False)
    corrupted["state"]["exp_avg"].add_(1)
    torch.save(corrupted, shard_path)
    with pytest.raises(ValueError, match="Source values differ"):
        verify(destination)


def test_partial_conversion_never_publishes_completion(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "partial"
    make_source(source)
    result = convert(source, destination, tp=2, pp=2, layers=2, only_layer=0)
    assert not result["complete"] and not (destination / ".save_complete").exists()


def _tree_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_fp32_checkpoint_without_master_converts_verifies_and_keeps_source(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    _, optim = make_source(source, include_master=False, fractional=True)
    before = _tree_hashes(source)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    assert _tree_hashes(source) == before
    for name, entry in converted["parameters"].items():
        if not entry["optimizer_fields"]:
            assert entry["master_origin"] is None
            continue
        assert entry["master_origin"] == "model_fp32"
        assert f"state.{name}.master_param" not in optim
        for shard in entry["shards"]:
            saved = torch.load(destination / shard["path"], weights_only=False)
            assert saved["state"]["master_param"].dtype == torch.float32
            assert torch.equal(saved["state"]["master_param"], saved["state"]["model"])
            assert shard["sha256"]["master_param"] == shard["sha256"]["model"]
    result = verify(destination)
    assert result["passed"] and _tree_hashes(source) == before
    active = next(name for name, entry in converted["parameters"].items() if entry["optimizer_fields"])
    assert "master_param" in result["checked_parameters"][active]["fields"]
    shard = converted["parameters"][active]["shards"][0]
    shard_path = destination / shard["path"]
    corrupted = torch.load(shard_path, weights_only=False)
    corrupted["state"]["master_param"].add_(1)
    torch.save(corrupted, shard_path)
    with pytest.raises(ValueError, match="Source values differ"):
        verify(destination)
    assert _tree_hashes(source) == before


def test_existing_optimizer_master_and_legacy_manifest_still_verify(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source, model_dtype=torch.bfloat16)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    for name, entry in converted["parameters"].items():
        if entry["optimizer_fields"]:
            assert entry["master_origin"] == "optimizer"
            assert optim[f"state.{name}.master_param"].dtype == torch.float32
            assert tensors[name].dtype == torch.bfloat16
        entry.pop("master_origin")
    (destination / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")
    assert verify(destination)["passed"]


@pytest.mark.parametrize(
    "case",
    [
        "bf16_no_master",
        "missing_first_moment",
        "missing_second_moment",
        "missing_step",
        "unknown_state",
        "unknown_hyper",
        "moment_dtype",
        "moment_shape",
        "negative_step",
        "fractional_step",
    ],
)
def test_missing_or_unknown_optimizer_state_fails_closed(tmp_path, case):
    source, destination = tmp_path / "source", tmp_path / "converted"
    name = "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"

    def edit(optim):
        prefix = f"state.{name}."
        if case == "missing_first_moment":
            del optim[prefix + "exp_avg"]
        elif case == "missing_second_moment":
            del optim[prefix + "exp_avg_sq"]
        elif case == "missing_step":
            del optim[f"param_groups.{name}.step"]
        elif case == "unknown_state":
            optim[prefix + "unknown"] = torch.zeros(8, 12)
        elif case == "unknown_hyper":
            optim[f"param_groups.{name}.unknown"] = 1
        elif case == "moment_dtype":
            optim[prefix + "exp_avg"] = optim[prefix + "exp_avg"].bfloat16()
        elif case == "moment_shape":
            optim[prefix + "exp_avg"] = optim[prefix + "exp_avg"][:-1]
        elif case == "negative_step":
            optim[f"param_groups.{name}.step"] = torch.tensor([-1], dtype=torch.int32)
        elif case == "fractional_step":
            optim[f"param_groups.{name}.step"] = torch.tensor([1.5], dtype=torch.float32)

    make_source(
        source,
        model_dtype=torch.bfloat16 if case == "bf16_no_master" else torch.float32,
        include_master=False,
        optimizer_edit=edit,
    )
    before = _tree_hashes(source)
    with pytest.raises(ValueError):
        convert(source, destination, tp=2, pp=2, layers=2)
    assert not (destination / ".save_complete").exists()
    assert _tree_hashes(source) == before


def test_forged_source_master_origin_is_rejected(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    make_source(source, include_master=False)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    active = next(name for name, entry in converted["parameters"].items() if entry["optimizer_fields"])
    converted["parameters"][active]["master_origin"] = "optimizer"
    (destination / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        verify(destination)


def test_forged_hyperparameter_manifest_is_rejected(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    make_source(source, include_master=False)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    active = next(name for name, entry in converted["parameters"].items() if entry["optimizer_fields"])
    converted["parameters"][active]["hyperparameter_fields"].remove(f"param_groups.{active}.step")
    (destination / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")
    with pytest.raises(ValueError, match="hyperparameter coverage"):
        verify(destination)
