"""Opt-in performance records without gradient scans or additional CUDA synchronization."""

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class Action128Performance(Callback):
    def on_train_start(self, model, iteration=0):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        output = Path(self.config.job.path_local) / "performance"
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f"rank_{self.rank}.jsonl"

    def on_before_dataloading(self, iteration=0):
        self.start = time.perf_counter()

    def on_after_dataloading(self, iteration=0):
        self.data_seconds = time.perf_counter() - self.start

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        # WandBCallback resets these timers at step_end, even with W&B disabled.
        # Snapshot before the step-end callback chain; retain host timing only.
        timers = self.trainer.training_timer.results
        self.host_phase_s = {key: values[-1] for key, values in timers.items() if values}

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        # The recipe's IterSpeed callback already synchronizes during its first
        # 50 steps. Do not insert a second synchronization or scan gradients.
        elapsed = time.perf_counter() - self.start
        packed = data_batch["packed_sequence"]
        record = {
            "iteration": iteration,
            "rank": self.rank,
            "dataset": data_batch["dataset_name"],
            "sample_keys": data_batch["__key__"],
            "step_wall_s": elapsed,
            "dataload_wall_s": self.data_seconds,
            "host_phase_s": self.host_phase_s,
            "tokens": {
                key: int(output_batch[key])
                for key in ("und_token_length", "vision_token_length", "action_token_length")
            },
            "vision_grid_thw": packed.vision.token_shapes,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
