"""Measurement boundaries must follow loaded steps and whole GA updates."""

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.training_performance_benchmark import TrainingPerformanceBenchmark


def test_resumed_warmup_counts_complete_updates(tmp_path, monkeypatch):
    clock = iter(range(100))
    monkeypatch.setattr(
        "cosmos_framework.callbacks.training_performance_benchmark.time.perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    benchmark = TrainingPerformanceBenchmark(warmup_steps=1)
    benchmark.config = SimpleNamespace(
        job=SimpleNamespace(path_local=str(tmp_path)),
        trainer=SimpleNamespace(max_iter=23002, run_validation=False, grad_accum_iter=2),
        checkpoint=SimpleNamespace(save_iter=1000000),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            parallelism=SimpleNamespace(
                context_parallel_shard_degree=1,
                cfg_parallel_shard_degree=1,
            )
        )
    )
    benchmark.on_train_start(model, iteration=23000)
    for iteration in (23000, 23001):
        for microstep in range(2):
            benchmark.on_before_dataloading(iteration)
            benchmark.on_after_dataloading(iteration)
            benchmark.on_training_step_batch_end(
                model,
                {"__key__": [f"sample{microstep}"]},
                {"gen_token_length": 17},
                None,
                iteration,
            )
        benchmark.on_training_step_end(model, {}, {}, None, iteration + 1)
    rows = [json.loads(line) for line in benchmark.path.read_text().splitlines()]
    assert [row["measured"] for row in rows] == [False, True]
    assert [row["microsteps"] for row in rows] == [2, 2]
    assert [row["gen_token_length"] for row in rows] == [34, 34]
    assert rows[0]["batch_digest"] == rows[1]["batch_digest"]
    assert rows[0]["seconds"] == 5
    assert rows[0]["data_wait_seconds"] == 2
    assert len(syncs) == 6  # Two before each update, one after; none between microsteps.
    with pytest.raises(FileExistsError):
        benchmark.on_train_start(model, iteration=23000)
