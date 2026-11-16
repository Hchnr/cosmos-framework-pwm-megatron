"""Action128 external initialization preserves learned heads and contracts."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.checkpoint.action128 import (
    action128_skip_keys,
    checkpoint_contract,
    read_source_contract,
    validate_checkpoint_digest,
    validate_contract_extension,
)
from cosmos_framework.checkpoint.base import CheckpointLoadSource
from cosmos_framework.checkpoint.dcp import AsyncMode, CustomLoadPlanner, DistributedCheckpointer
from cosmos_framework.data.generator.dataflow.loader import JointCosmosDataLoader
from cosmos_framework.model.generator.mot.action128 import (
    Action128DomainLinear,
    Action128Normalization,
    action128_contract_digest,
)
from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec


def _model(*, legacy=False, mode="action128_seam"):
    model = nn.Module()
    model.net = nn.Module()
    model.net.backbone = nn.Linear(2, 2)
    dim, slots = (64, 32) if legacy else (128, 64)
    model.net.action2llm = Action128DomainLinear(dim, 8, slots)
    model.net.llm2action = Action128DomainLinear(8, dim, slots)
    if not legacy:
        model.net.action_normalization = Action128Normalization(tiny_action128_spec(), mode)
    return model


def _load(path, target, **options):
    reader = dcp.FileSystemReader(path)
    skips = action128_skip_keys(reader.read_metadata().state_dict_metadata, [], **options)
    state = target.state_dict()
    dcp.load(state, storage_reader=reader, planner=CustomLoadPlanner(keys_to_skip_loading=skips))
    target.load_state_dict(state)
    return skips


@pytest.mark.parametrize("load_training_state", [False, True])
def test_external_action128_preserves_heads(tmp_path, load_training_state):
    source, target = _model(), _model()
    assert not torch.equal(source.net.action2llm.fc.weight, target.net.action2llm.fc.weight)
    dcp.save(source.state_dict(), checkpoint_id=tmp_path)
    assert _load(tmp_path, target, load_training_state=load_training_state) == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_legacy_base_loads_backbone_and_keeps_fresh_heads(tmp_path):
    source, target = _model(legacy=True), _model()
    initial = target.net.action2llm.fc.weight.detach().clone()
    dcp.save(source.state_dict(), checkpoint_id=tmp_path)
    assert "action2llm" in _load(tmp_path, target)
    torch.testing.assert_close(target.net.action2llm.fc.weight, initial)
    torch.testing.assert_close(target.net.backbone.weight, source.net.backbone.weight)
    target.net.action_normalization.validate_contract()


def test_contract_mismatch_needs_explicit_model_only_reset(tmp_path):
    source = _model(mode="canvas")
    dcp.save(source.state_dict(), checkpoint_id=tmp_path)
    with pytest.raises(ValueError, match="spatial RoPE mode"):
        _load(tmp_path, _model())
    target = _model()
    initial = target.net.action2llm.fc.weight.detach().clone()
    _load(tmp_path, target, mode="reset_heads")
    torch.testing.assert_close(target.net.action2llm.fc.weight, initial)
    target.net.action_normalization.validate_contract()


@pytest.mark.parametrize(
    "legacy,mode,skips", [(True, "auto", []), (False, "reset_heads", []), (False, "auto", ["backbone"])]
)
def test_full_state_rejects_partial_initialization(legacy, mode, skips):
    with pytest.raises(ValueError, match="full-state"):
        action128_skip_keys(_model(legacy=legacy).state_dict(), skips, mode=mode, load_training_state=True)


def test_partial_contract_or_implicit_head_skip_is_rejected():
    state = _model().state_dict()
    with pytest.raises(ValueError, match="reset_heads explicitly"):
        action128_skip_keys(state, ["action2llm"])
    del state["net.action_normalization.scale"]
    with pytest.raises(ValueError, match="Incomplete"):
        action128_skip_keys(state, [])


def _extended_spec():
    spec = tiny_action128_spec()
    registry = spec["registry"]
    # New domain sorts BEFORE the old one, shifting its normalization row to 1.
    registry["domains"].insert(0, "agibot_g2__parallel_grippers/joint128/test")
    registry["embodiments"].insert(0, "agibot_g2__parallel_grippers")
    registry["domain_embodiments"] = [0, 1]
    registry["domain_projection_ids"].insert(0, 22)
    spec["transforms"].insert(0, copy.deepcopy(spec["transforms"][0]))
    spec["transforms"][0]["normalization_mask"][0] = True
    spec["transforms"][0]["scale"][0] = 3.25
    spec["source_digests"].insert(0, "new-source")
    return spec


def _training_model(spec):
    model = _model()
    model.net.action_normalization = Action128Normalization(spec, "action128_seam")
    model.net_ema = copy.deepcopy(model.net)
    model.net_ema.requires_grad_(False)
    with torch.no_grad():
        model.net_ema.backbone.weight.fill_(9)
    model.config = SimpleNamespace(action128_spec=spec)
    return model


def _joint_loader(spec, *, extra=False, changed=False):
    from cosmos_framework.data.generator.action.unified_action128 import stable_hash

    class Dataset:
        def __init__(self, name):
            self.name = name

        def fingerprint_for_spec(self, old_spec):
            return stable_hash([self.name, old_spec, changed])

    entries = {}
    for name in ["old:0", "new:1"] if extra else ["old:0"]:
        dataset = Dataset(name)
        distributor = SimpleNamespace(_dataset=dataset, dataset_fingerprint=dataset.fingerprint_for_spec(spec))
        entries[name] = {"ratio": 1, "dataloader": SimpleNamespace(dataset=SimpleNamespace(_distributor=distributor))}
    return JointCosmosDataLoader(entries, selection_mode="exact_periodic")


def _training_state(model):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    scheduler.step()
    return optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False)


def _checkpointer(tmp_path, callback, *, source=None, mode="extend"):
    from cosmos_framework.utils.config import CheckpointConfig

    checkpointer = object.__new__(DistributedCheckpointer)
    checkpointer.config_checkpoint = CheckpointConfig(action128_load_mode=mode, load_ema_to_reg=True)
    checkpointer.callbacks = MagicMock()
    checkpointer.callbacks._callbacks = [callback]
    checkpointer.async_mode = AsyncMode.DISABLED
    checkpointer._local_dirname = str(tmp_path)
    checkpointer.save_to_object_store = False
    checkpointer.save_s3_backend_key = None
    checkpointer.load_training_state = True
    checkpointer._get_dcp_save_process_group = lambda: None
    checkpointer._write_latest_checkpoint_file = lambda _: None
    checkpointer._check_checkpoint_exists = lambda _: None
    if source is not None:
        location = CheckpointLoadSource(str(source), None, None, warm_start=True)
        checkpointer.keys_to_resume_during_load = lambda: (set(checkpointer.CHECKPOINT_KEYS), location)
    return checkpointer


@pytest.fixture
def cpu_checkpoint(monkeypatch):
    import cosmos_framework.checkpoint.dcp as module

    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda *args: 0)
    monkeypatch.setattr(module, "get_rand_state_dict", lambda: {})
    monkeypatch.setattr(module, "set_rand_state_dict", lambda _: None)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("resume_mode", ["auto", "extend"])
def test_extend_dcp_preserves_all_slots_optimizer_and_resumes_new_cursors(
    tmp_path, cpu_checkpoint, legacy, resume_mode
):
    old_spec, new_spec = tiny_action128_spec(), _extended_spec()
    source = _training_model(old_spec)
    if legacy:
        # Simulate the older executable's 22 registered heads, still 64 slots.
        old_spec["projection_routing"]["keys"] = {
            key: value for key, value in old_spec["projection_routing"]["keys"].items() if value <= 21
        }
        for net in (source.net, source.net_ema):
            normalizer = net.action_normalization
            normalizer.spec = copy.deepcopy(old_spec)
            normalizer._expected_digest = list(action128_contract_digest(old_spec, "action128_seam"))
            normalizer.reset_contract()
    optimizer, scheduler, scaler = _training_state(source)
    old_loader = _joint_loader(old_spec)
    old_loader.set_start_iteration(100)
    old_callback = JointCosmosDataLoaderStateCallback(old_loader)
    old_callback._inner_callback("old:0")._update_state_from_batch(
        {"sample_worker_id": torch.tensor([0]), "sample_epoch": torch.tensor([1]), "sample_index": torch.tensor([20])}
    )
    writer = _checkpointer(tmp_path / "old/checkpoints", old_callback)
    if legacy:
        writer.callbacks.on_save_checkpoint.side_effect = lambda model, state_dict: state_dict["trainer"].pop(
            "action128_contract"
        )
    writer.save(source, optimizer, scheduler, scaler, 50)
    if legacy:
        (tmp_path / "old/config.yaml").write_text(
            json.dumps(
                {
                    "model": {
                        "config": {
                            "action128_spec": old_spec,
                            "diffusion_expert_config": {"vision_spatial_position_mode": "action128_seam"},
                        }
                    }
                }
            )
        )

    target = _training_model(new_spec)
    target_optimizer, target_scheduler, target_scaler = _training_state(target)
    new_loader = _joint_loader(new_spec, extra=True)
    new_callback = JointCosmosDataLoaderStateCallback(new_loader)
    reader = _checkpointer(tmp_path / "new", new_callback, source=tmp_path / "old/checkpoints/iter_000000050")
    assert reader.load(target, target_optimizer, target_scheduler, target_scaler) == 50
    assert reader.reset_dataloader_state and not reader.loaded_dataloader_state
    assert new_loader._global_id == 0
    assert new_callback._inner["old:0"].state == {}
    for name, value in source.state_dict().items():
        if "action_normalization." not in name:
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    assert target.net.action_normalization.scale.shape == (2, 128)
    assert target.net.action_normalization.scale[0, 0] == 3.25
    assert target.net.action_normalization.scale[1, 0] == 1
    assert target_scheduler.state_dict() == scheduler.state_dict()
    for pid, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(target_optimizer.state_dict()["state"][pid][name], value, rtol=0, atol=0)

    # Checkpoint after continuation: model iteration != new data-stream cursor.
    new_loader.set_start_iteration(4)
    reader.save(target, target_optimizer, target_scheduler, target_scaler, 52)
    again = _training_model(new_spec)
    again_state = _training_state(again)
    again_loader = _joint_loader(new_spec, extra=True)
    again_callback = JointCosmosDataLoaderStateCallback(again_loader)
    resume = _checkpointer(tmp_path / "new", again_callback, source=tmp_path / "new/iter_000000052", mode=resume_mode)
    assert resume.load(again, *again_state) == 52
    assert resume.loaded_dataloader_state and not resume.reset_dataloader_state
    assert resume.loaded_dataloader_global_id == 4
    assert again_loader._global_id == 4


@pytest.mark.parametrize("mutation", ["scale", "source", "head", "remove", "rope", "renumber"])
def test_extend_rejects_changes_to_old_contract(mutation):
    saved = checkpoint_contract(_training_model(tiny_action128_spec()))
    current = checkpoint_contract(_training_model(_extended_spec()))
    spec = current["spec"]
    if mutation == "scale":
        spec["transforms"][1]["scale"][0] = 2
    elif mutation == "source":
        spec["source_digests"][1] = "changed"
    elif mutation == "head":
        spec["registry"]["domain_projection_ids"][1] = 1
    elif mutation == "remove":
        spec["registry"]["domains"][1] = "renamed"
    elif mutation == "rope":
        current["vision_spatial_position_mode"] = "canvas"
    else:
        spec["projection_routing"]["keys"]["eef20/shared"] = 1
    with pytest.raises(ValueError):
        validate_contract_extension(saved, current)


def test_append_head_map_and_unknown_legacy_rejected():
    current = checkpoint_contract(_training_model(tiny_action128_spec()))
    saved = copy.deepcopy(current)
    saved["spec"]["projection_routing"]["keys"] = {
        key: value for key, value in saved["spec"]["projection_routing"]["keys"].items() if value <= 21
    }
    assert validate_contract_extension(saved, current)
    with pytest.raises(ValueError, match="trained Action128"):
        action128_skip_keys({}, [], mode="extend", load_training_state=True)


def test_extend_refuses_old_data_change_or_removal():
    spec = tiny_action128_spec()
    saved = JointCosmosDataLoaderStateCallback(_joint_loader(spec)).state_dict()
    callback = JointCosmosDataLoaderStateCallback(_joint_loader(spec, extra=True, changed=True))
    with pytest.raises(ValueError, match="unchanged"):
        callback.validate_action128_extension(saved, spec)


def test_saved_contract_json_round_trip():
    contract = checkpoint_contract(_training_model(tiny_action128_spec()))
    assert not validate_contract_extension(contract, json.loads(json.dumps(contract)))


@pytest.mark.parametrize("model_only", [False, True])
def test_legacy_source_config_must_match_checkpoint_digest(tmp_path, model_only):
    import yaml

    source = _training_model(tiny_action128_spec())
    checkpoint = tmp_path / "backup" if model_only else tmp_path / "checkpoints/iter_000000010"
    checkpoint.mkdir(parents=True)
    if not model_only:
        dcp.save({"iteration": 10}, checkpoint_id=checkpoint / "trainer")
    config = {
        "model": {
            "config": {
                "action128_spec": source.net.action_normalization.spec,
                "diffusion_expert_config": {"vision_spatial_position_mode": "action128_seam"},
            }
        }
    }
    config_path = (checkpoint if model_only else tmp_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpointer = _checkpointer(tmp_path, None)
    location = CheckpointLoadSource(str(checkpoint), None, None, True)
    contract = read_source_contract(checkpointer, location)
    digests = {"net.action_normalization.contract_digest": source.net.action_normalization.contract_digest}
    validate_checkpoint_digest(contract, digests)
    contract["spec"]["transforms"][0]["scale"][0] = 99
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_digest(contract, digests)
    config_path.unlink()
    with pytest.raises(ValueError, match="original config.yaml"):
        read_source_contract(checkpointer, location)


def test_published_alias_migration_is_model_only_and_preserves_old_heads():
    from cosmos_framework.data.generator.action.projection_routing import PUBLISHED_PROJECTION_FAMILY_IDS

    current = checkpoint_contract(_training_model(tiny_action128_spec()))
    saved = copy.deepcopy(current)
    saved["spec"]["projection_routing"]["version"] = "action128-projection-routing-v1"
    current["spec"]["projection_routing"]["keys"].update(PUBLISHED_PROJECTION_FAMILY_IDS)
    assert validate_contract_extension(saved, current, allow_normalization_migration=True)
    with pytest.raises(ValueError, match="projection version"):
        validate_contract_extension(saved, current)
    current["spec"]["projection_routing"]["keys"]["eef20/shared"] = 1
    with pytest.raises(ValueError):
        validate_contract_extension(saved, current, allow_normalization_migration=True)


@pytest.mark.parametrize("extra", [False, True])
def test_data_only_append_and_exact_resume(extra):
    spec = tiny_action128_spec()
    old = JointCosmosDataLoaderStateCallback(_joint_loader(spec)).state_dict()
    current = JointCosmosDataLoaderStateCallback(_joint_loader(spec, extra=extra))
    assert current.validate_action128_extension(old, spec) is extra


def test_extend_requires_full_training_state(tmp_path, cpu_checkpoint):
    model = _training_model(tiny_action128_spec())
    dcp.save(model.state_dict(), checkpoint_id=tmp_path / "source/model")
    checkpointer = _checkpointer(tmp_path, None)
    checkpointer.keys_to_resume_during_load = lambda: (
        {"model"},
        CheckpointLoadSource(str(tmp_path / "source"), None, None, True),
    )
    with pytest.raises(ValueError, match="full training state"):
        checkpointer.load(model)


def test_toml_checkpoint_accepts_extend():
    from cosmos_framework.configs.toml_config.sft_config import CheckpointConfig

    config = CheckpointConfig(action128_load_mode="extend", load_training_state=True)
    assert config.action128_load_mode == "extend"


def test_warm_start_migrates_normalization_and_loads_only_weights(tmp_path, cpu_checkpoint):
    old_spec = tiny_action128_spec()
    new_spec = _extended_spec()
    # Both removal and replacement of normalization rules are intentional.
    new_spec["transforms"][1]["scale"][0] = 7.0
    new_spec["source_digests"][1] = "recomputed"
    source = _training_model(old_spec)
    source_state = _training_state(source)
    callback = JointCosmosDataLoaderStateCallback(_joint_loader(old_spec))
    writer = _checkpointer(tmp_path / "old", callback)
    writer.save(source, *source_state, 50)

    target = _training_model(new_spec)
    optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    loader = _joint_loader(new_spec, extra=True, changed=True)
    callback = JointCosmosDataLoaderStateCallback(loader)
    reader = _checkpointer(tmp_path / "new", callback, mode="warm_start")
    location = CheckpointLoadSource(str(tmp_path / "old/iter_000000050"), None, None, warm_start=True)
    reader.keys_to_resume_during_load = lambda: ({"model"}, location)
    assert reader.load(target, optimizer, scheduler) == 0
    assert optimizer.state_dict()["state"] == {}
    assert scheduler.state_dict() == scheduler_before
    assert not reader.loaded_dataloader_state
    assert loader._global_id == 0
    for name, value in source.state_dict().items():
        if "action_normalization." not in name:
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    assert target.net.action_normalization.scale[1, 0] == 7.0
    target.net.action_normalization.validate_contract()

    # Subsequent same-run auto-resume must not apply the migration again.
    loader.set_start_iteration(4)
    reader.save(target, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False), 2)
    again = _training_model(new_spec)
    again_loader = _joint_loader(new_spec, extra=True, changed=True)
    resume = _checkpointer(tmp_path / "new", JointCosmosDataLoaderStateCallback(again_loader), mode="warm_start")
    same_run = CheckpointLoadSource(str(tmp_path / "new/iter_000000002"), None, None, warm_start=False)
    resume.keys_to_resume_during_load = lambda: (set(resume.CHECKPOINT_KEYS), same_run)
    again_optimizer = torch.optim.AdamW(again.parameters(), lr=0.01)
    again_scheduler = torch.optim.lr_scheduler.StepLR(again_optimizer, step_size=1)
    assert resume.load(again, again_optimizer, again_scheduler, torch.amp.GradScaler("cpu", enabled=False)) == 2
    assert resume.loaded_dataloader_state and again_loader._global_id == 4


def test_warm_start_contract_keeps_head_and_rope_checks():
    saved = checkpoint_contract(_training_model(tiny_action128_spec()))
    current = checkpoint_contract(_training_model(_extended_spec()))
    current["spec"]["registry"]["domains"][1] = "replacement-domain"
    assert validate_contract_extension(saved, current, allow_normalization_migration=True)
    current["spec"]["projection_routing"]["keys"]["eef20/shared"] = 1
    with pytest.raises(ValueError, match="renumber"):
        validate_contract_extension(saved, current, allow_normalization_migration=True)
    current = copy.deepcopy(saved)
    current["vision_spatial_position_mode"] = "canvas"
    with pytest.raises(ValueError, match="RoPE"):
        validate_contract_extension(saved, current, allow_normalization_migration=True)


def test_warm_start_requires_model_only_and_trained_checkpoint():
    from cosmos_framework.configs.toml_config.sft_config import CheckpointConfig

    assert CheckpointConfig(action128_load_mode="warm_start").load_training_state is False
    with pytest.raises(ValueError, match="model-only"):
        action128_skip_keys(_model().state_dict(), [], mode="warm_start", load_training_state=True)
    with pytest.raises(ValueError, match="skipped keys"):
        action128_skip_keys(_model().state_dict(), ["action2llm"], mode="warm_start")
    with pytest.raises(ValueError, match="trained Action128"):
        action128_skip_keys(_model(legacy=True).state_dict(), [], mode="warm_start")


def test_dcp_roundtrip_restores_shared_prefetch_committed_state(tmp_path, cpu_checkpoint):
    import os
    from unittest.mock import patch

    from cosmos_framework.data.generator.dataflow import CosmosDataLoader, IdentityProcessor, MapDistributor
    from cosmos_framework.data.generator.dataflow.shared_prefetch import CosmosDataStream, SharedPrefetchDataLoader

    def build(workers):
        base = CosmosDataLoader(
            MapDistributor([{"id": torch.tensor(i)} for i in range(40)], name="shared", shuffle=False),
            IdentityProcessor(),
            batch_size=2,
        )
        return SharedPrefetchDataLoader(
            {"shared": {"dataloader": CosmosDataStream(base.dataset, 2), "ratio": 1}},
            num_workers=workers,
            prefetch_factor=3,
        )

    original = build(1)
    callback = JointCosmosDataLoaderStateCallback(original)
    source = _training_model(tiny_action128_spec())
    training_state = _training_state(source)
    try:
        for _ in range(5):
            callback._update_state_from_batch(next(iter(original)))
        writer = _checkpointer(tmp_path / "saved", callback, mode="auto")
        writer.save(source, *training_state, iteration=3)
        expected = [next(iter(original)) for _ in range(7)]
        restored = build(2)
        restored_callback = JointCosmosDataLoaderStateCallback(restored)
        target = _training_model(tiny_action128_spec())
        target_training_state = _training_state(target)
        reader = _checkpointer(
            tmp_path / "resumed", restored_callback, source=tmp_path / "saved/iter_000000003", mode="auto"
        )
        try:
            with patch.dict(os.environ):
                assert reader.load(target, *target_training_state) == 3
                assert reader.loaded_dataloader_global_id == 5
                for ref in expected:
                    actual = next(iter(restored))
                    torch.testing.assert_close(actual["id"], ref["id"], rtol=0, atol=0)
                    torch.testing.assert_close(actual["sample_worker_id"], ref["sample_worker_id"], rtol=0, atol=0)
        finally:
            restored.close()
    finally:
        original.close()
