"""The diagnostic loader preserves CPU input and isolates mutable metadata."""

import pytest
import torch

from tools.action_input_replay import ActionInputReplay, build_action_input_replay, configure_replay


def test_replay_isolates_metadata_and_preserves_cpu_tensor_storage():
    batch = {"video": [[torch.arange(4)]], "sequence_plan": [{"frames": [0, 1]}]}
    iterator = iter(ActionInputReplay(batch))
    first = next(iterator)
    first["sequence_plan"][0]["frames"].append(99)
    first["video"].append("mutated")
    second = next(iterator)
    assert second["sequence_plan"] == batch["sequence_plan"]
    assert len(second["video"]) == 1
    assert second["video"][0][0].data_ptr() == batch["video"][0][0].data_ptr()


def test_replay_refuses_resume():
    loader = ActionInputReplay({})
    loader.set_start_iteration(0)
    with pytest.raises(ValueError, match="does not support training resume"):
        loader.set_start_iteration(1)


def test_fixture_cache_keeps_identical_input_and_starts_no_workers(tmp_path, monkeypatch):
    from cosmos_framework.data.generator import catalog_processors
    from cosmos_framework.utils import misc

    calls = []
    seeds = []

    def capture(**kwargs):
        calls.append(kwargs)
        yield {"sequence_plan": [{}], "video": [torch.arange(4)]}

    monkeypatch.setattr(catalog_processors, "build_catalog_training_dataloader", capture)
    monkeypatch.setattr(misc, "set_random_seed", lambda **kwargs: seeds.append(kwargs))
    kwargs = dict(replay_root=str(tmp_path), max_samples_per_batch=1, seed=42, num_workers=18)
    a = build_action_input_replay(**kwargs)
    fingerprint = (tmp_path / "rank_0.json").read_text()
    b = build_action_input_replay(**kwargs)
    assert len(calls) == 1
    assert calls[0]["num_workers"] == 0
    assert calls[0]["prefetch_factor"] is None
    assert torch.equal(next(iter(a))["video"][0], next(iter(b))["video"][0])
    assert (tmp_path / "rank_0.json").read_text() == fingerprint
    assert seeds == [{"seed": 42, "by_rank": True}] * 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        build_action_input_replay(**{**kwargs, "max_samples_per_batch": 2})


def test_replay_entry_changes_only_loader_and_catalog_cursor(tmp_path):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "dataloader_train": {"_target_": "original.factory", "max_samples_per_batch": 11},
            "trainer": {"callbacks": {"dataloader_state": {"_target_": "cursor"}, "benchmark": {"warmup": 10}}},
        }
    )
    configure_replay(config, tmp_path)
    assert config.dataloader_train._target_ == "tools.action_input_replay.build_action_input_replay"
    assert config.dataloader_train.max_samples_per_batch == 11
    assert config.dataloader_train.replay_root == str(tmp_path)
    assert "dataloader_state" not in config.trainer.callbacks
    assert config.trainer.callbacks.benchmark.warmup == 10
