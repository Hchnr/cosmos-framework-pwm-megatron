from types import SimpleNamespace

import pytest

from cosmos_framework.callbacks.action_training_benchmark import ActionTrainingBenchmark


def test_accumulation_counts_every_microbatch():
    callback = ActionTrainingBenchmark()
    for _ in range(4):
        callback.on_training_step_batch_end(None, {"sequence_plan": [None]}, {"action_token_length": 32}, None)
    assert callback.samples == 4
    assert callback.microsteps == 4
    assert callback.tokens["action_token_length"] == 128


def test_timer_not_reset_between_microsteps(monkeypatch):
    import cosmos_framework.callbacks.action_training_benchmark as module

    calls = []
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(module.time, "perf_counter", lambda: 123.0)
    callback = ActionTrainingBenchmark()
    callback.on_before_dataloading(0)
    callback.on_before_dataloading(0)
    assert callback.started == 123.0
    assert calls == ["sync"]


def test_rejects_context_parallel_duplicate_counting():
    model = SimpleNamespace(config=SimpleNamespace(parallelism=SimpleNamespace(context_parallel_shard_degree=2)))
    with pytest.raises(ValueError, match="CP=1"):
        ActionTrainingBenchmark().on_train_start(model)


def test_data_wait_accumulates_without_extra_cuda_sync(monkeypatch):
    import cosmos_framework.callbacks.action_training_benchmark as module

    times = iter([10.0, 10.0, 10.2, 11.0, 11.3])
    syncs = []
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: syncs.append(True))
    callback = ActionTrainingBenchmark()
    for _ in range(2):
        callback.on_before_dataloading(0)
        callback.on_after_dataloading(0)
    assert callback.data_seconds == pytest.approx(0.5)
    assert syncs == [True]
