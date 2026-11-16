from types import SimpleNamespace

import torch

from cosmos_framework.callbacks.training_stats import TrainingStatsCallback


def test_action128_statistics_use_source_domains_not_shared_projection_heads():
    callback = TrainingStatsCallback()
    callback.on_train_start(
        SimpleNamespace(
            config=SimpleNamespace(
                action128_spec={
                    "registry": {"embodiments": ["robot_a", "robot_b"], "domain_embodiments": [0, 1, 0]},
                }
            )
        )
    )
    mixed = {"domain_id": [torch.tensor([0]), torch.tensor([0])], "normalization_domain_id": [0, 1]}
    callback._accumulate_embodiment_counts(mixed)
    assert callback._embodiment_counts == {"robot_a": 1, "robot_b": 1}
    assert callback._get_batch_embodiment(mixed) is None
    same_robot = {"normalization_domain_id": [torch.tensor([0]), torch.tensor([2])]}
    assert callback._get_batch_embodiment(same_robot) == "robot_a"
    assert callback._get_batch_embodiment({}) is None


def test_legacy_statistics_keep_legacy_mapping():
    callback = TrainingStatsCallback()
    callback.on_train_start(SimpleNamespace(config=SimpleNamespace()))
    batch = {"domain_id": [torch.tensor([999])]}
    callback._accumulate_embodiment_counts(batch)
    assert callback._embodiment_counts == {"unknown_999": 1}
    assert callback._get_batch_embodiment(batch) == "unknown_999"


def _callback():
    callback = TrainingStatsCallback(log_freq=1)
    callback.on_train_start(
        SimpleNamespace(
            config=SimpleNamespace(
                action128_spec={"registry": {"embodiments": ["a", "b"], "domain_embodiments": [0, 1]}}
            )
        )
    )
    return callback


def test_sample_weighted_rank_merge_and_empty_rank(monkeypatch):
    callback = _callback()
    for _ in range(3):
        callback.on_training_step_batch_end(None, {"normalization_domain_id": [0, 0, 0]}, {}, torch.tensor(1.0))
    import cosmos_framework.callbacks.training_stats as module

    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 3)

    def gather(destination, local):
        destination[:] = [local, ({"train_stats_loss/a": [9.0, 1.0]}, 0), ({}, 0)]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    metrics = callback._compute_per_embodiment_loss_stats("train")
    assert metrics["train_stats_loss/a"] == 1.8
    assert callback._per_embodiment_loss == {}


def test_ga_counts_all_microbatches_and_mixed_loss_is_not_attributed():
    callback = _callback()
    callback.on_training_step_batch_end(None, {"normalization_domain_id": [0, 0]}, {}, torch.tensor(2.0))
    callback.on_training_step_batch_end(None, {"normalization_domain_id": [0]}, {}, torch.tensor(5.0))
    callback.on_training_step_batch_end(None, {"normalization_domain_id": [0, 1]}, {}, torch.tensor(100.0))
    callback.on_training_step_batch_end(None, {"normalization_domain_id": [0]}, {}, torch.tensor(float("nan")))
    result = callback._compute_per_embodiment_loss_stats("train")
    assert result["train_stats_loss/a"] == 3.0
    assert result["train_stats_loss_coverage/skipped_mixed_samples"] == 2
    assert "train_stats_loss/b" not in result
    assert callback._embodiment_counts == {"a": 5, "b": 1}


def test_registered_callback_emits_wandb_metrics(monkeypatch):
    import cosmos_framework.callbacks.training_stats as module

    callback = _callback()
    records = []
    monkeypatch.setattr(module.wandb, "run", object())
    monkeypatch.setattr(module.wandb, "log", lambda data, **kwargs: records.append(data))
    monkeypatch.setattr(module.distributed, "is_rank0", lambda: True)
    batch = {"normalization_domain_id": [0]}
    output = {"flow_matching_loss_action": torch.tensor(2.0)}
    callback.on_training_step_batch_end(None, batch, output, torch.tensor(3.0))
    callback.on_training_step_end(None, batch, output, torch.tensor(3.0), iteration=1)
    assert records[0]["train_stats_loss/a"] == 3.0
    assert records[0]["train_stats_loss_detail/a_flow_matching_loss_action"] == 2.0
    assert records[0]["train_stats_embodiment/a"] == 100.0
