"""Opt-in synchronized update measurements for controlled training comparisons."""

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class TrainingPerformanceBenchmark(Callback):
    """Append last in the callback list; includes data loading and all GA microsteps.

    Use isolated runs without periodic checkpoints or validation. A final checkpoint
    after the last update is outside the measurements. Warmup is relative to the
    loaded iteration, not the absolute training step. Aggregate the slowest rank
    per update and sum actual tokens across ranks (CP/CFG must be one).
    """

    def __init__(self, warmup_steps: int = 5):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        self.warmup_steps = warmup_steps
        self.started = None
        self._reset_update()

    def _reset_update(self):
        self.data_seconds = 0.0
        self.microsteps = 0
        self.tokens = {}
        self.batch_digest = hashlib.sha256()

    def on_train_start(self, model, iteration=0):
        parallel = model.config.parallelism
        if parallel.context_parallel_shard_degree != 1 or parallel.cfg_parallel_shard_degree != 1:
            raise ValueError("Benchmark token aggregation requires CP=CFG=1")
        self.initial_iteration = iteration
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        output = Path(self.config.job.path_local) / "benchmark"
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f"rank_{self.rank}.jsonl"
        if self.path.exists():
            raise FileExistsError(f"Use a fresh benchmark output directory: {self.path}")
        last = self.config.trainer.max_iter
        save_every = self.config.checkpoint.save_iter
        if last // save_every != iteration // save_every or self.config.trainer.run_validation:
            raise ValueError("Benchmark updates must not contain periodic checkpoint saves or validation")

    def on_before_dataloading(self, iteration=0):
        if self.started is None:
            # Synchronize at optimizer-update boundaries only, never between GA microsteps.
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()
        self.loading_started = time.perf_counter()

    def on_after_dataloading(self, iteration=0):
        self.data_seconds += time.perf_counter() - self.loading_started

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        self.microsteps += 1
        for name in (
            "und_token_length",
            "gen_token_length",
            "vision_token_length",
            "physical_und_token_length",
            "physical_gen_token_length",
        ):
            self.tokens[name] = self.tokens.get(name, 0) + int(output_batch.get(name, 0))
        identity = {
            "dataset": data_batch.get("dataset_name"),
            "keys": data_batch.get("__key__"),
            "split_lens": output_batch.get("split_lens"),
            "vae_latent_shapes": output_batch.get("vae_latent_shapes"),
            "tokens": self.tokens,
        }
        self.batch_digest.update(json.dumps(identity, sort_keys=True, default=str).encode())

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        from torch._dynamo.utils import counters

        row = dict(
            iteration=iteration,
            rank=self.rank,
            measured=iteration > self.initial_iteration + self.warmup_steps,
            seconds=elapsed,
            data_wait_seconds=self.data_seconds,
            microsteps=self.microsteps,
            batch_digest=self.batch_digest.hexdigest(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            compile_unique_graphs=counters["stats"]["unique_graphs"],
            compile_graph_breaks=sum(counters["graph_break"].values()),
            **self.tokens,
        )
        if self.microsteps != self.config.trainer.grad_accum_iter:
            raise RuntimeError("Benchmark did not observe a complete optimizer update")
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.started = None
        self._reset_update()
