"""Exercise the real loop boundary without GPU model initialization."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import cosmos_framework.trainer as trainer_module
from cosmos_framework.trainer import ImaginaireTrainer


@pytest.mark.parametrize("start,max_iter,accum", [(0, 2, 1), (0, 2, 2), (2, 4, 2), (2, 2, 1), (3, 2, 1)])
@pytest.mark.parametrize("restored_data", [False, True])
@pytest.mark.parametrize("reset_data", [False, True])
def test_training_stops_without_fetching_an_unused_batch(
    monkeypatch, start, max_iter, accum, restored_data, reset_data
):
    trainer = object.__new__(ImaginaireTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            memory_format=torch.contiguous_format,
            grad_scaler_args={"enabled": False},
            grad_accum_iter=accum,
            distributed_parallelism="fsdp",
            max_iter=max_iter,
            run_validation=False,
            save_zero_checkpoint=False,
            timeout_period=0,
            straggler_detection=SimpleNamespace(analyze_dataloading=False),
        ),
        checkpoint=SimpleNamespace(save_iter=1),
        optimizer=None,
        scheduler=None,
    )
    trainer.callbacks = MagicMock()
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.load.return_value = start
    trainer.checkpointer.loaded_dataloader_state = restored_data
    trainer.checkpointer.loaded_dataloader_global_id = 3 if restored_data else None
    trainer.checkpointer.reset_dataloader_state = reset_data
    trainer.training_timer = lambda _: nullcontext()
    trainer.straggler_detector = MagicMock()
    trainer.straggler_detector.profile_section.side_effect = lambda *args, **kwargs: nullcontext()
    trainer._cp_data_window = MagicMock()
    calls = []

    def step(*args, **kwargs):
        calls.append(1)
        return {}, torch.tensor(1.0), len(calls) % accum

    trainer.training_step = step
    model = MagicMock()
    model.to.return_value = model
    model.training = True
    model.parallel_dims = None
    model.init_optimizer_scheduler.return_value = (MagicMock(), MagicMock())
    expected_batches = max(0, max_iter - start) * accum
    resume_events = []
    expect_preserve_rng = restored_data and start > 0
    data_starts = []
    data_closed = []

    class ExactLoader:
        def close(self):
            data_closed.append(True)

        def set_start_iteration(self, iteration):
            data_starts.append(iteration)

        def preserve_rng_on_resume(self):
            resume_events.append("preserve")

        def __iter__(self):
            assert bool(resume_events) == expect_preserve_rng
            assert expected_batches > 0, "Completed resume must not even construct an iterator"
            for _ in range(expected_batches):
                yield {}
            raise AssertionError("Fetched an unused batch after the final optimizer step")

    for name in ("maybe_enable_profiling", "maybe_enable_memory_snapshot", "maybe_enable_nsys_profiling"):
        monkeypatch.setattr(trainer_module, name, lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(trainer_module.misc, "to", lambda batch, **kwargs: batch)
    monkeypatch.setattr(trainer_module.distributed, "barrier", lambda: None)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(trainer_module.signal, "alarm", lambda _: None)
    trainer.train(model, ExactLoader(), None)
    assert len(calls) == expected_batches
    trainer.callbacks.on_train_end.assert_called_once_with(model, iteration=max(start, max_iter))
    trainer.checkpointer.finalize.assert_called_once()
    assert resume_events == (["preserve"] if expect_preserve_rng else [])
    assert data_starts == [3 if restored_data else (0 if reset_data else start * accum)]
    assert data_closed == [True]
