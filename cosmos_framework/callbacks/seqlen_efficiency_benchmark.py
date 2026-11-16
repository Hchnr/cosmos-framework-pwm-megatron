"""Per-rank complete-update timing and pack audit for the fixed-seqlen study."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class SeqlenEfficiencyBenchmark(Callback):
    """Write one JSONL record per optimizer update, including every GA microbatch.

    The timer includes data loading, transfers, forward/backward and optimizer.
    CUDA and rank synchronization happen only at update boundaries. Profiling
    iterations are retained as evidence but excluded from the steady-state wall.
    """

    def __init__(self, expected_samples_per_pack: int, warmup_steps: int = 3, profile_steps: int = 2):
        if expected_samples_per_pack < 1 or warmup_steps < 0 or profile_steps < 0:
            raise ValueError("Invalid seqlen efficiency benchmark settings")
        self.expected_samples_per_pack = expected_samples_per_pack
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.started: float | None = None
        self._reset()

    def _reset(self) -> None:
        self.data_seconds = 0.0
        self.microbatches: list[dict] = []

    def on_train_start(self, model, iteration=0):
        parallel = model.config.parallelism
        if parallel.context_parallel_shard_degree != 1 or parallel.cfg_parallel_shard_degree != 1:
            raise ValueError("Seqlen benchmark requires CP=CFG=1")
        self.initial_iteration = iteration
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.last = self.config.trainer.max_iter
        if self.last // self.config.checkpoint.save_iter != iteration // self.config.checkpoint.save_iter:
            raise ValueError("Measured updates must not contain periodic checkpoint saves")
        output = Path(self.config.job.path_local) / "seqlen_efficiency"
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f"rank_{self.rank}.jsonl"
        if self.path.exists():
            raise FileExistsError(f"Use a fresh study output: {self.path}")

    def on_before_dataloading(self, iteration=0):
        if self.started is None:
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
        packed = data_batch["packed_sequence"]
        lengths = [int(value) for value in packed.sample_lens]
        splits = [int(value) for value in packed.split_lens]
        count = len(data_batch["sequence_plan"])
        if count != self.expected_samples_per_pack or len(lengths) != count or len(splits) != 2 * count:
            raise ValueError(f"Pack count mismatch at update {iteration}: samples={count}, splits={len(splits)}")
        names = (
            "und_token_length",
            "gen_token_length",
            "vision_token_length",
            "action_token_length",
            "physical_und_token_length",
            "physical_gen_token_length",
        )
        tokens = {name: int(output_batch.get(name, 0)) for name in names}
        fixed = self.config.model.config.fixed_sequence_length
        if (tokens["physical_und_token_length"], tokens["physical_gen_token_length"]) != (
            fixed.und_tokens,
            fixed.gen_tokens,
        ):
            raise ValueError(f"Physical decoder length changed at update {iteration}: {tokens}")
        if (sum(splits[0::2]), sum(splits[1::2])) != (
            tokens["und_token_length"],
            tokens["gen_token_length"],
        ):
            raise ValueError(f"Real packed token counts disagree with model output at update {iteration}: {tokens}")
        self.microbatches.append(
            {
                "samples": count,
                "sample_lengths": lengths,
                "split_lens": splits,
                "real_total": sum(lengths),
                "tokens": tokens,
            }
        )

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.started
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"Non-finite training loss at update {iteration}")
        if len(self.microbatches) != self.config.trainer.grad_accum_iter:
            raise RuntimeError("Incomplete gradient accumulation update")
        from torch._dynamo.utils import counters

        row = {
            "rank": self.rank,
            "iteration": iteration,
            "window": (
                "warmup"
                if iteration <= self.initial_iteration + self.warmup_steps
                else "profile"
                if iteration > self.last - self.profile_steps
                else "steady"
            ),
            "seconds": seconds,
            "data_wait_seconds": self.data_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "compile_unique_graphs": counters["stats"]["unique_graphs"],
            "compile_graph_breaks": sum(counters["graph_break"].values()),
            "loss": float(loss.detach()),
            "microbatches": self.microbatches,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if iteration == self.last and dist.is_initialized():
            # Every rank must finish the final profiled update before the
            # profiler gate calls cudaProfilerStop on any one of them.
            dist.barrier()
        self.started = None
        self._reset()
