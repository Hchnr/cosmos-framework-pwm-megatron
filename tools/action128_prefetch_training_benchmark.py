# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in real training measurements for the independent prefetch ablation.

Use the normal launcher with SFT_TRAINING_MODULE set to this module and an
isolated OUTPUT_ROOT. Only checkpoint writes are skipped. Model construction,
weight loading, forward/backward and optimizer updates follow production code.
"""

from __future__ import annotations

import json
import math
import os
import runpy
import signal
import time
import types
from pathlib import Path

import torch

from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
from cosmos_framework.utils.callback import Callback, CallBackGroup


def _configure_episode_columns():
    mode = os.environ.get("ACTION128_BENCH_EPISODE_COLUMNS", "projected")
    if mode not in {"all", "projected"}:
        raise ValueError(f"Unknown episode column mode: {mode}")
    if mode == "all":
        import cosmos_framework.data.generator.action.unified_action128 as action128

        parquet = action128.pq

        def read_table(*args, **kwargs):
            if kwargs.get("columns") == list(action128._EPISODE_COLUMNS):
                kwargs.pop("columns")
            return parquet.read_table(*args, **kwargs)

        action128.pq = types.SimpleNamespace(
            **{name: getattr(parquet, name) for name in dir(parquet) if name != "read_table"},
            read_table=read_table,
        )


class _TrainingBenchmark(Callback):
    def __init__(self):
        self.started = None
        self.data_wait = 0.0
        self.microbatches = []

    def on_train_start(self, model, iteration=0):
        self.owner = os.getpid()
        self.rank = int(os.environ.get("RANK", "0"))
        output = Path(os.environ["OUTPUT_ROOT"]) / "training_benchmark"
        output.mkdir(parents=True, exist_ok=True)
        self.stream = (output / f"rank_{self.rank:05d}.jsonl").open("a", buffering=1)
        self.loader = next(
            (callback._outer for callback in self.trainer.callbacks._callbacks if hasattr(callback, "_outer")), None
        )

    def on_before_dataloading(self, iteration=0):
        if self.started is None:
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.start_ns = time.time_ns()
        self.loading_started = time.perf_counter()

    def on_after_dataloading(self, iteration=0):
        self.data_wait += time.perf_counter() - self.loading_started

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        assert output_batch["task_type"] == "action" and model.config.max_action_dim == 128
        action_loss = float(output_batch["flow_matching_loss_action"].detach().item())
        assert math.isfinite(action_loss) and action_loss > 0
        self.microbatches.append(
            {
                "dataset": data_batch["dataset_name"],
                "sample_keys": data_batch["__key__"],
                "samples": int(output_batch["batch_size"]),
                "action_loss": action_loss,
                **{
                    name: int(output_batch.get(name, 0))
                    for name in (
                        "und_token_length",
                        "vision_token_length",
                        "action_token_length",
                        "physical_und_token_length",
                        "physical_gen_token_length",
                    )
                },
            }
        )

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        ended = time.perf_counter()
        end_ns = time.time_ns()
        record = {
            "iteration": iteration,
            "rank": self.rank,
            "seconds": ended - self.started,
            "start_ns": self.start_ns,
            "end_ns": end_ns,
            "data_wait_seconds": self.data_wait,
            "microbatches": self.microbatches,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "prefetch_stats": getattr(self.loader, "prefetch_stats", None),
        }
        self.stream.write(json.dumps(record) + "\n")
        self.started = None
        self.data_wait = 0.0
        self.microbatches = []

    def on_train_end(self, model, iteration=0):
        self.stream.close()
        # All measured updates are complete. Stop speculative legacy suffixes
        # together, rather than paying one join timeout for every source.
        for callback in self.trainer.callbacks._callbacks:
            loader = getattr(callback, "_outer", None)
            if loader is None or hasattr(loader, "close"):
                continue
            iterators = getattr(loader, "_iterators", None) or []
            for iterator in iterators:
                for worker in getattr(iterator, "_workers", []):
                    if worker.is_alive():
                        worker.terminate()
            for iterator in iterators:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if shutdown is not None:
                    shutdown()


_original_group_init = CallBackGroup.__init__


def _group_init(self, config, trainer):
    _original_group_init(self, config, trainer)
    callback = _TrainingBenchmark()
    callback.config = config
    callback.trainer = trainer
    self._callbacks.append(callback)


def _skip_checkpoint_write(self, *args, **kwargs):
    return None


if __name__ == "__main__":
    _configure_episode_columns()
    from cosmos_framework.data.generator.dataflow.shared_prefetch import SharedPrefetchDataLoader

    _original_shared_init = SharedPrefetchDataLoader.__init__
    _pin_mode = os.environ.get("ACTION128_BENCH_PIN_MEMORY_MODE", "background")
    if _pin_mode not in {"background", "inline"}:
        raise ValueError(f"Unknown pin memory mode: {_pin_mode}")

    def _shared_init(self, *args, **kwargs):
        kwargs["pin_memory_in_background"] = _pin_mode == "background"
        _original_shared_init(self, *args, **kwargs)

    SharedPrefetchDataLoader.__init__ = _shared_init
    CallBackGroup.__init__ = _group_init
    DistributedCheckpointer.save = _skip_checkpoint_write
    # Legacy data workers also inherit the trainer's signal handler under fork.
    # Reset it only in the benchmark's CPU children, as shared workers do.
    from torch.utils.data._utils import worker as worker_module

    _original_worker_loop = worker_module._worker_loop

    def _worker_loop(*args, **kwargs):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        return _original_worker_loop(*args, **kwargs)

    worker_module._worker_loop = _worker_loop
    torch.set_num_threads(1)
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
