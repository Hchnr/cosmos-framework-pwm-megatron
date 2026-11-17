"""Observe Python GC pauses without changing the default collection policy."""

import gc
import json
import os
import time
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


class PythonGCTrace(Callback):
    def __init__(self, output, disable_after_iteration=None):
        super().__init__()
        self.output = Path(output)
        self.disable_after_iteration = disable_after_iteration
        self.records = []
        self.active = {}
        self.iteration = None
        self.initial_enabled = None
        self.disabled = False
        self.previous_thresholds = gc.get_threshold()

    def _observe(self, phase, info):
        generation = info["generation"]
        now = time.perf_counter_ns()
        if phase == "start":
            token = torch.cuda.nvtx.range_start(f"attribution.python_gc.generation_{generation}")
            self.active[generation] = (now, token, self.iteration)
        else:
            started = self.active.pop(generation, None)
            if started:
                start, token, iteration = started
                torch.cuda.nvtx.range_end(token)
                self.records.append(
                    {
                        "iteration": iteration,
                        "generation": generation,
                        "start_monotonic_ns": start,
                        "end_monotonic_ns": now,
                        "duration_seconds": (now - start) / 1e9,
                        "collected": info["collected"],
                        "uncollectable": info["uncollectable"],
                    }
                )

    def on_train_start(self, model, iteration=0):
        self.initial_enabled = gc.isenabled()
        gc.callbacks.append(self._observe)

    def on_before_dataloading(self, iteration=0):
        self.iteration = iteration

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration == self.disable_after_iteration:
            # Toggle only automatic collection after workers exist; keep heap
            # contents identical through warmup and retain scheduled manual GC.
            gc.disable()
            self.disabled = True

    def on_train_end(self, model, iteration=0):
        gc.callbacks.remove(self._observe)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.with_name(f"{self.output.stem}_rank_{os.environ['RANK']}.json").write_text(
            json.dumps(
                {
                    "initial_enabled": self.initial_enabled,
                    "final_enabled": gc.isenabled(),
                    "disabled_after_iteration": self.disable_after_iteration,
                    "disabled": self.disabled,
                    "thresholds": self.previous_thresholds,
                    "records": self.records,
                },
                indent=2,
            )
            + "\n"
        )
