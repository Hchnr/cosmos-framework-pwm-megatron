"""Opt-in host phase/compile/callback evidence for Action128 slow steps.

Use SFT_TRAINING_MODULE=tools.action128_spike_probe with the normal launcher.
No added collectives or CUDA synchronizations. CUDA-event timings remain in
the existing load_balance_trace callback. Phase times include asynchronous
launch/wait time, not isolated GPU kernel duration.
"""

import collections
import contextlib
import faulthandler
import json
import os
import runpy
import signal
import time
from pathlib import Path

import torch

from cosmos_framework.utils import nvtx

_owner = os.getpid()
_rank = int(os.environ.get("RANK", "0"))
_output = Path(os.environ["OUTPUT_ROOT"]) / "spike_probe"
_output.mkdir(parents=True, exist_ok=True)
_stream = (_output / f"rank_{_rank:05d}.jsonl").open("a", buffering=1)
_stack = (_output / f"rank_{_rank:05d}.stacks.log").open("a", buffering=1)
faulthandler.enable(file=_stack)
faulthandler.register(signal.SIGUSR2, file=_stack, all_threads=True)
_phases = collections.defaultdict(float)
_iteration = 0
_microbatch = -1
_data_start = None
_batch_start = None
_step_start = None
_save_start = None
_original_range = nvtx.nvtx_range


def emit(kind, **fields):
    if os.getpid() == _owner:
        _stream.write(
            json.dumps(dict(kind=kind, rank=_rank, iteration=_iteration, timestamp_ns=time.time_ns(), **fields)) + "\n"
        )


@contextlib.contextmanager
def timed_range(name, required=nvtx.NVTXGranularity.PHASE):
    if torch.compiler.is_compiling() or os.getpid() != _owner or required > nvtx.NVTXGranularity.PHASE:
        with _original_range(name, required):
            yield
        return
    start = time.perf_counter()
    try:
        with _original_range(name, required):
            yield
    finally:
        _phases[name] += time.perf_counter() - start


nvtx.nvtx_range = timed_range

# Import after installing the phase wrapper so model/trainer imports see it.
from cosmos_framework.utils.callback import CallBackGroup


def traced_getattr(self, method_name):
    def call(*args, **kwargs):
        global _iteration, _microbatch, _data_start, _batch_start, _step_start, _save_start
        if method_name == "on_before_dataloading":
            iteration = kwargs.get("iteration", args[0] if args else 0)
            if iteration != _iteration or _step_start is None:
                _iteration = iteration
                _step_start = time.perf_counter()
                _microbatch = -1
            _microbatch += 1
            _data_start = time.perf_counter()
            _phases.clear()
            faulthandler.dump_traceback_later(45, file=_stack)
            emit("microbatch_start", microbatch=_microbatch)
        if method_name == "on_training_step_batch_start":
            _batch_start = time.perf_counter()
        if method_name == "on_save_checkpoint_start":
            _save_start = time.perf_counter()
            emit("checkpoint_start", completed_iteration=kwargs.get("iteration"))
        callback_times = {}
        for callback in self._callbacks:
            start = time.perf_counter()
            getattr(callback, method_name)(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > 0.005:
                callback_times[type(callback).__name__] = elapsed
        if callback_times:
            emit("callbacks", hook=method_name, host_s=callback_times)
        if method_name == "on_after_dataloading" and _data_start is not None:
            emit("dataloader", microbatch=_microbatch, host_s=time.perf_counter() - _data_start)
        if method_name == "on_training_step_batch_end":
            from torch._dynamo.utils import counters

            memory = torch.cuda.memory_stats()

            emit(
                "microbatch_end",
                microbatch=_microbatch,
                host_s=time.perf_counter() - _batch_start,
                phases=dict(_phases),
                unique_graphs=counters["stats"]["unique_graphs"],
                graph_breaks=dict(counters["graph_break"]),
                max_allocated=torch.cuda.max_memory_allocated(),
                max_reserved=torch.cuda.max_memory_reserved(),
                allocator_retries=memory.get("num_alloc_retries", 0),
                allocator_syncs=memory.get("num_sync_all_streams", 0),
            )
            faulthandler.cancel_dump_traceback_later()
        if method_name == "on_save_checkpoint_end" and _save_start is not None:
            emit("checkpoint_end", host_s=time.perf_counter() - _save_start)
        if method_name == "on_training_step_end" and _step_start is not None:
            emit(
                "optimizer_step_end",
                completed_iteration=kwargs.get("iteration"),
                wall_s=time.perf_counter() - _step_start,
                microbatches=_microbatch + 1,
            )

    return call


CallBackGroup.__getattr__ = traced_getattr

# The save-start callback runs after waiting for an earlier async save. Measure
# the entire call as well, so a slow previous write cannot hide outside the
# callback timing when comparing synchronous and asynchronous checkpointing.
from cosmos_framework.checkpoint.dcp import DistributedCheckpointer

_original_save = DistributedCheckpointer.save


def timed_save(self, *args, **kwargs):
    start = time.perf_counter()
    try:
        return _original_save(self, *args, **kwargs)
    finally:
        emit("checkpoint_call_end", host_s=time.perf_counter() - start)


DistributedCheckpointer.save = timed_save

if __name__ == "__main__":
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
