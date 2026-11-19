"""All-component canonical import format and source tensor reconstruction."""

import hashlib
import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.convert_cosmos_checkpoint_to_megatron_tp_pp import convert
from tools.verify_cosmos_tp_pp_conversion import verify


def make_source(path):
    name = "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"
    row = "net.language_model.model.layers.1.self_attn.o_proj_moe_gen.weight"
    frozen = "net.language_model.model.embed_tokens.weight"
    tensors = {
        name: torch.arange(96, dtype=torch.float32).view(8, 12),
        row: torch.arange(96, dtype=torch.float32).view(8, 12) + 2,
        frozen: torch.arange(24, dtype=torch.float32).view(6, 4),
    }
    optim = {}
    for key in (name, row):
        for field, value in {
            "master_param": tensors[key] + 0.001,
            "exp_avg": tensors[key] * 0.01,
            "exp_avg_sq": tensors[key] * 0.02 + 0.01,
        }.items():
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
