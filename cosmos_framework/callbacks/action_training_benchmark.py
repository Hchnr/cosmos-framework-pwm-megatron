"""Opt-in optimizer-update measurements for controlled action training experiments."""

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.utils.callback import Callback


class ActionTrainingBenchmark(Callback):
    """Measure complete updates, including loading and all accumulation microbatches.

    Enable this callback last, after validation/logging callbacks. Measurements
    synchronize only at update boundaries. Each rank writes its own JSONL; rank
    zero also logs its measurements to the existing online W&B run. The summary
    should use the slowest rank's total window time, not the last microstep timer.
    """

    def __init__(self, warmup_steps: int = 10, expected_global_batch_size: int = 32):
        if warmup_steps < 0 or expected_global_batch_size < 1:
            raise ValueError("Invalid benchmark warmup or global batch size")
        self.warmup_steps = warmup_steps
        self.expected_global_batch_size = expected_global_batch_size
        self.started = None
        self.loading_started = None
        self.data_seconds = 0.0
        self.samples = 0
        self.microsteps = 0
        self.tokens = {}

    def on_train_start(self, model, iteration=0):
        parallel = model.config.parallelism
        if parallel.context_parallel_shard_degree != 1:
            raise ValueError("Action benchmark sample counting requires CP=1")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.compile_enabled = model.config.compile.enabled
        self.output = Path(self.config.job.path_local) / "benchmark"
        self.output.mkdir(parents=True, exist_ok=True)
        if self.rank == 0 and wandb.run:
            (self.output / "wandb_url.txt").write_text(wandb.run.url + "\n")

    def on_before_dataloading(self, iteration=0):
        if self.started is None:
            torch.cuda.synchronize()
            if iteration == self.warmup_steps:
                torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()
        self.loading_started = time.perf_counter()

    def on_after_dataloading(self, iteration=0):
        self.data_seconds += time.perf_counter() - self.loading_started

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        self.samples += len(data_batch["sequence_plan"])
        self.microsteps += 1
        for name in (
            "und_token_length",
            "vision_token_length",
            "action_token_length",
            "physical_und_token_length",
            "physical_gen_token_length",
        ):
            self.tokens[name] = self.tokens.get(name, 0) + int(output_batch.get(name, 0))

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        counts = torch.tensor(self.samples, device="cuda", dtype=torch.int64)
        if dist.is_initialized():
            dist.all_reduce(counts)
        global_samples = int(counts.item())
        if global_samples != self.expected_global_batch_size:
            raise ValueError(f"Actual global batch {global_samples} != {self.expected_global_batch_size}")
        row = dict(
            iteration=iteration,
            measured=iteration > self.warmup_steps,
            seconds=elapsed,
            data_wait_seconds=self.data_seconds,
            local_samples=self.samples,
            global_samples=global_samples,
            microsteps=self.microsteps,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            **self.tokens,
        )
        if self.compile_enabled:
            from torch._dynamo.utils import counters

            row["compile_unique_graphs"] = counters["stats"]["unique_graphs"]
            row["compile_graph_breaks"] = sum(counters["graph_break"].values())
        with (self.output / f"rank_{self.rank}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if self.rank == 0 and wandb.run:
            wandb.log({f"benchmark/{k}": v for k, v in row.items()}, step=iteration)
        self.started = None
        self.data_seconds = 0.0
        self.samples = self.microsteps = 0
        self.tokens = {}

    def on_train_end(self, model, iteration=0):
        if self.compile_enabled:
            from torch._dynamo.utils import counters

            report = {str(group): {str(k): v for k, v in values.items()} for group, values in counters.items()}
            (self.output / f"compile_rank_{self.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
