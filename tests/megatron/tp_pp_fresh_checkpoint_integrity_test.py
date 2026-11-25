"""Real CPU DCP controls for the isolated metadata-integrity proof."""

import pickle
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from tp_pp_fresh_checkpoint_restore_test import add_moments, make_optimizer, template
from tp_pp_fresh_checkpoint_restore_test import gloo as gloo

from cosmos_framework.training_backends.megatron_tp_pp import checkpoint
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states


def save_optimizer(path, mode):
    from megatron.core import dist_checkpointing

    stage, optimizer = make_optimizer()
    if mode != "untrained":
        add_moments(stage, optimizer)
    if mode == "mixed":
        list(adam_parameter_states(stage, optimizer))[1].state.clear()
    if mode == "untrained":
        list(adam_parameter_states(stage, optimizer))
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(optimizer, loading=False)}, str(path))
    return stage, optimizer


def test_trained_dcp_missing_both_moments_rejected_by_persisted_digest(gloo, tmp_path):
    path = tmp_path / "trained"
    save_optimizer(path, "trained")
    digest = checkpoint._metadata_sha256(path)
    metadata_path = path / ".metadata"
    # This file was just created by this test. Alter only the temporary fixture.
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    before = set(metadata.state_dict_metadata)
    metadata.state_dict_metadata = {
        k: v for k, v in metadata.state_dict_metadata.items() if not k.endswith((".exp_avg", ".exp_avg_sq"))
    }
    assert len(before) - len(metadata.state_dict_metadata) == 2
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
    stage, target = make_optimizer()
    entries = list(adam_parameter_states(stage, target))
    before = {id(param): id(value) for param, value in target.optimizer.state.items()}
    with pytest.raises(ValueError, match="At least one rank") as failure:
        checkpoint._prepare_optimizer_for_load(target, path, device="cpu", metadata_sha256=digest)
    assert "SHA256 mismatch" in str(failure.value.__cause__)
    assert {id(param): id(value) for param, value in target.optimizer.state.items()} == before
    assert all(not entry.state for entry in entries)


@pytest.mark.parametrize("mode", ["untrained", "mixed"])
def test_legacy_checkpoint_with_ambiguous_missing_moments_fails_closed(gloo, tmp_path, mode):
    path = tmp_path / mode
    save_optimizer(path, mode)
    stage, target = make_optimizer()
    list(adam_parameter_states(stage, target))
    with pytest.raises(ValueError, match="At least one rank") as failure:
        checkpoint._prepare_optimizer_for_load(target, path, device="cpu")
    assert "Legacy checkpoint has ambiguous" in str(failure.value.__cause__)


def test_legacy_trained_complete_moments_remain_loadable(gloo, tmp_path):
    from megatron.core import dist_checkpointing

    path = tmp_path / "complete_legacy"
    stage, source = save_optimizer(path, "trained")
    expected = [
        (e.state["exp_avg"].clone(), e.state["exp_avg_sq"].clone()) for e in adam_parameter_states(stage, source)
    ]
    stage, target = make_optimizer()
    list(adam_parameter_states(stage, target))
    checkpoint._prepare_optimizer_for_load(target, path, device="cpu")
    state = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(path))
    target.load_state_dict(state["optimizer"])
    for entry, (avg, avg_sq) in zip(adam_parameter_states(stage, target), expected, strict=True):
        torch.testing.assert_close(entry.state["exp_avg"], avg, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg_sq"], avg_sq, rtol=0, atol=0)


@pytest.mark.parametrize("value", [None, "", "a" * 63, "A" * 64, 123])
def test_malformed_metadata_digest_is_not_treated_as_legacy(gloo, tmp_path, value):
    path = tmp_path / "trained"
    save_optimizer(path, "trained")
    stage, target = make_optimizer()
    list(adam_parameter_states(stage, target))
    with pytest.raises(ValueError, match="At least one rank") as failure:
        checkpoint._prepare_optimizer_for_load(target, path, device="cpu", metadata_sha256=value)
    assert "Invalid checkpoint DCP metadata SHA256" in str(failure.value.__cause__)


def test_metadata_symlink_rejected(tmp_path):
    source = tmp_path / "real"
    source.write_bytes(b"metadata")
    (tmp_path / ".metadata").symlink_to(source)
    with pytest.raises(OSError):
        checkpoint._metadata_sha256(tmp_path)


def test_metadata_change_during_preflight_rejected_without_mutating_optimizer(gloo, tmp_path, monkeypatch):
    from torch.distributed.checkpoint import FileSystemReader

    path = tmp_path / "trained"
    save_optimizer(path, "trained")
    digest = checkpoint._metadata_sha256(path)
    stage, target = make_optimizer()
    entries = list(adam_parameter_states(stage, target))
    before = {id(param): id(value) for param, value in target.optimizer.state.items()}
    real = FileSystemReader.read_metadata

    def changed_during_read(reader):
        result = real(reader)
        with (path / ".metadata").open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(FileSystemReader, "read_metadata", changed_during_read)
    with pytest.raises(ValueError, match="At least one rank") as failure:
        checkpoint._prepare_optimizer_for_load(target, path, device="cpu", metadata_sha256=digest)
    assert "SHA256 mismatch" in str(failure.value.__cause__)
    assert {id(param): id(value) for param, value in target.optimizer.state.items()} == before
    assert all(not entry.state for entry in entries)


@dataclass
class Topology:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    use_distributed_optimizer: bool = True


def driver(stage, optimizer, iteration):
    stage.sharded_state_dict = lambda: {}
    stage.load_state_dict = lambda value, strict: None
    optimizer._copy_main_params_to_model_params = lambda: None
    return SimpleNamespace(
        stage=stage,
        optimizer=optimizer,
        iteration=iteration,
        scheduler=None,
        topology=Topology(),
        model=SimpleNamespace(start_param_sync=lambda **kwargs: None),
    )


def test_real_save_publishes_metadata_digest_and_real_load_verifies_it(gloo, tmp_path, monkeypatch):
    # The production wrapper requests CUDA for NCCL control tensors. Route only
    # these tiny scalar controls to CPU; DCP and the optimizer remain real.
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpoint, "release_file_cache", lambda paths: None)
    stage, source = make_optimizer()
    add_moments(stage, source)
    expected = [
        (e.state["exp_avg"].clone(), e.state["exp_avg_sq"].clone()) for e in adam_parameter_states(stage, source)
    ]
    path = tmp_path / "published"
    checkpoint.save(driver(stage, source, 4), path, rank_state={"cursor": 7})
    import json

    manifest = json.loads((path / "manifest.json").read_text())
    assert (path / ".save_complete").is_file()
    assert manifest["dcp_metadata_sha256"] == checkpoint._metadata_sha256(path)
    stage, target = make_optimizer()
    list(adam_parameter_states(stage, target))
    recovered = driver(stage, target, 0)
    assert checkpoint.load(recovered, path) == {"cursor": 7}
    assert recovered.iteration == 4
    for entry, (avg, avg_sq) in zip(adam_parameter_states(stage, target), expected, strict=True):
        torch.testing.assert_close(entry.state["exp_avg"], avg, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg_sq"], avg_sq, rtol=0, atol=0)


def test_real_non_distributed_mcore_optimizer_loads_legacy_checkpoint(gloo, tmp_path, monkeypatch):
    import json

    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.optimizer.optimizer import FP32Optimizer

    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpoint, "release_file_cache", lambda paths: None)

    class Stage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 4.0]))

        def sharded_state_dict(self):
            return {"weight": ShardedTensor.from_rank_offsets("weight", self.weight)}

    def build():
        stage = Stage()
        inner = torch.optim.AdamW(stage.parameters(), lr=0.01)

        def initialize(optimizer, config):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    state = optimizer.state[param]
                    if not state:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

        optimizer = FP32Optimizer(inner, SimpleNamespace(config_logger_dir=""), initialize)
        result = SimpleNamespace(
            stage=stage,
            optimizer=optimizer,
            iteration=1,
            scheduler=None,
            topology=Topology(use_distributed_optimizer=False),
            model=SimpleNamespace(start_param_sync=lambda **kwargs: None),
        )
        return result

    source = build()
    source.stage.weight.grad = torch.tensor([1.0, 2.0])
    source.optimizer.optimizer.step()
    expected_weight = source.stage.weight.detach().clone()
    expected = {key: value.clone() for key, value in source.optimizer.optimizer.state[source.stage.weight].items()}
    path = tmp_path / "nondistributed_legacy"
    checkpoint.save(source, path, rank_state={"cursor": 3})
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["dcp_metadata_sha256"]
    manifest_path.write_text(json.dumps(manifest))
    target = build()
    with torch.no_grad():
        target.stage.weight.zero_()
    assert checkpoint.load(target, path) == {"cursor": 3}
    torch.testing.assert_close(target.stage.weight, expected_weight, rtol=0, atol=0)
    for key, value in expected.items():
        torch.testing.assert_close(target.optimizer.optimizer.state[target.stage.weight][key], value, rtol=0, atol=0)


def test_changed_metadata_after_dcp_read_poisons_driver(gloo, tmp_path, monkeypatch):
    from megatron.core import dist_checkpointing

    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    stage, source = make_optimizer()
    add_moments(stage, source)
    path = tmp_path / "published"
    checkpoint.save(driver(stage, source, 4), path, rank_state={"cursor": 7})
    stage, target = make_optimizer()
    add_moments(stage, target)
    for entry in adam_parameter_states(stage, target):
        entry.state["exp_avg"].fill_(42.0)
        entry.state["exp_avg_sq"].fill_(43.0)
    before = [(e.state["exp_avg"].clone(), e.state["exp_avg_sq"].clone()) for e in adam_parameter_states(stage, target)]
    called = []
    recovered = driver(stage, target, 0)
    stage.load_state_dict = lambda value, strict: called.append(True)
    real_load = dist_checkpointing.load

    def changed_after_read(*args, **kwargs):
        result = real_load(*args, **kwargs)
        with (path / ".metadata").open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(dist_checkpointing, "load", changed_after_read)
    with pytest.raises(ValueError, match="At least one rank cannot verify checkpoint DCP metadata"):
        checkpoint.load(recovered, path)
    assert not called
    assert recovered._checkpoint_restore_failed is True
    # MCore may already have loaded into existing tensors. Check this explicitly
    # rather than presenting the failed load as an atomic rollback.
    assert any(
        not torch.equal(entry.state["exp_avg"], avg) or not torch.equal(entry.state["exp_avg_sq"], avg_sq)
        for entry, (avg, avg_sq) in zip(adam_parameter_states(stage, target), before, strict=True)
    )
    with pytest.raises(RuntimeError, match="discard and rebuild"):
        checkpoint.load(recovered, path)
    with pytest.raises(RuntimeError, match="discard and rebuild"):
        checkpoint.save(recovered, tmp_path / "forbidden", rank_state={})


def test_invalid_metadata_preflight_leaves_driver_and_optimizer_unchanged(gloo, tmp_path, monkeypatch):
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    stage, source = make_optimizer()
    add_moments(stage, source)
    path = tmp_path / "published"
    checkpoint.save(driver(stage, source, 4), path, rank_state={})
    with (path / ".metadata").open("ab") as handle:
        handle.write(b"damaged")
    stage, target = make_optimizer()
    add_moments(stage, target)
    entries = list(adam_parameter_states(stage, target))
    before = [(entry.state, entry.state["exp_avg"].clone(), entry.state["exp_avg_sq"].clone()) for entry in entries]
    recovered = driver(stage, target, 0)
    with pytest.raises(ValueError, match="At least one rank"):
        checkpoint.load(recovered, path)
    assert not getattr(recovered, "_checkpoint_restore_failed", False)
    checkpoint._require_usable_driver(recovered)
    for entry, (state, avg, avg_sq) in zip(adam_parameter_states(stage, target), before, strict=True):
        assert entry.state is state
        torch.testing.assert_close(entry.state["exp_avg"], avg, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg_sq"], avg_sq, rtol=0, atol=0)


def test_failed_driver_rejects_training_before_cuda_access():
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    broken = object.__new__(ModelParallelDriver)
    broken._checkpoint_restore_failed = True
    with pytest.raises(RuntimeError, match="discard and rebuild"):
        broken.forward_backward([], None)
    with pytest.raises(RuntimeError, match="discard and rebuild"):
        broken.step()
    with pytest.raises(RuntimeError, match="discard and rebuild"):
        broken.update([], None)
