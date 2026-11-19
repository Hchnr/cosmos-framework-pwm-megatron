"""Collect unreachable Python cycles at bounded microbatch intervals.

This does not alter live activations, tensors, optimizer state, or CUDA caches.
It is separate from the per-update callback because a model-parallel update can
contain hundreds of pipeline microbatches before that callback runs.
"""

import functools
import gc
import time
from contextlib import contextmanager

import torch


class MicrobatchGarbageCollector:
    def __init__(self, interval):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
            raise ValueError("Microbatch GC interval must be a nonnegative integer")
        self.interval = interval
        self.completed_backwards = 0
        self.collections = []

    def after_backward(self):
        self.completed_backwards += 1
        if not self.interval or self.completed_backwards % self.interval:
            return
        allocated_before = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
        started = time.monotonic()
        objects = gc.collect()
        allocated_after = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
        self.collections.append(
            {
                "completed_backwards": self.completed_backwards,
                "collected_python_objects": objects,
                "reclaimed_allocated_bytes": allocated_before - allocated_after,
                "host_seconds": time.monotonic() - started,
            }
        )


@contextmanager
def collect_microbatch_cycles(interval):
    """Wrap MCore backward for one schedule call and restore it on every exit."""
    collector = MicrobatchGarbageCollector(interval)
    if not interval:
        yield collector
        return
    from megatron.core.pipeline_parallel import schedules

    original = schedules.backward_step
    if getattr(original, "_cosmos_microbatch_gc", False):
        raise RuntimeError("Nested microbatch GC schedule scopes are unsupported")

    @functools.wraps(original)
    def backward(*args, **kwargs):
        result = original(*args, **kwargs)
        collector.after_backward()
        return result

    backward._cosmos_microbatch_gc = True
    schedules.backward_step = backward
    try:
        yield collector
    finally:
        schedules.backward_step = original
