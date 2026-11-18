"""Diagnostic allocation timing/counters; opt-in, no tensor lifetime changes."""

import functools
import json
import math
import os
import time
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


def use_default_copy_in_stream():
    """Diagnostic: share compute's cache pool; keep separate NCCL/RS streams."""
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext

    original = FSDPCommContext.lazy_init
    changed = []

    @functools.wraps(original)
    def initialize(context, device):
        original(context, device)
        previous = context.all_gather_copy_in_stream.cuda_stream
        context.all_gather_copy_in_stream = torch.cuda.default_stream(device)
        changed.append(dict(previous_stream=previous, new_stream=context.all_gather_copy_in_stream.cuda_stream))

    FSDPCommContext.lazy_init = initialize
    return changed


def memory_counters():
    stats = torch.cuda.memory_stats()
    names = (
        "allocated_bytes.all.current",
        "reserved_bytes.all.current",
        "active_bytes.all.current",
        "inactive_split_bytes.all.current",
        "num_alloc_retries",
        "num_ooms",
        "num_sync_all_streams",
        "num_device_alloc",
        "num_device_free",
    )
    return {name: stats.get(name) for name in names}


def stream_pools():
    """Summarize cached blocks by stream without retaining tensor references."""
    pools = {}
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] != torch.cuda.current_device():
            continue
        row = pools.setdefault(str(segment["stream"]), dict(reserved=0, active=0, inactive=0, largest_inactive=0))
        row["reserved"] += segment["total_size"]
        row["active"] += segment["active_size"]
        for block in segment["blocks"]:
            if block["state"] == "inactive":
                row["inactive"] += block["size"]
                row["largest_inactive"] = max(row["largest_inactive"], block["size"])
    return pools


class AllocatorTrace(Callback):
    def __init__(self, output, start_iteration):
        super().__init__()
        self.output = Path(output)
        self.start_iteration = start_iteration
        self.iteration = None
        self.allocations = []
        self.steps = []
        self.patches = []
        self.installed = False

    def _wrap(self, cls, method, size_argument, dtype_argument):
        original = getattr(cls, method)

        @functools.wraps(original)
        def traced(instance, *args, **kwargs):
            size = kwargs.get(size_argument)
            if size is None:
                size = args[0 if size_argument == "size" and len(args) == 1 else 1]
            dtype = kwargs.get(dtype_argument)
            if dtype is None:
                dtype = args[2]
            elements = math.prod(size) if isinstance(size, (tuple, list, torch.Size)) else int(size)
            requested = elements * dtype.itemsize
            before = memory_counters()
            stream = torch.cuda.current_stream().cuda_stream
            token = torch.cuda.nvtx.range_start(f"attribution.allocate.{cls.__name__}.{requested}")
            start = time.perf_counter_ns()
            try:
                return original(instance, *args, **kwargs)
            finally:
                end = time.perf_counter_ns()
                torch.cuda.nvtx.range_end(token)
                after = memory_counters()
                self.allocations.append(
                    dict(
                        iteration=self.iteration,
                        allocator=cls.__name__,
                        requested_bytes=requested,
                        stream=stream,
                        start_monotonic_ns=start,
                        end_monotonic_ns=end,
                        duration_seconds=(end - start) / 1e9,
                        before=before,
                        after=after,
                    )
                )

        setattr(cls, method, traced)
        self.patches.append((cls, method, original))

    def on_before_dataloading(self, iteration=0):
        self.iteration = iteration
        if iteration < self.start_iteration:
            return
        if not self.installed:
            from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllocMixin

            self._wrap(DefaultAllocMixin, "allocate", "size", "dtype")
            from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
                StorageResizeBasedBucketAllocator,
            )

            self._wrap(StorageResizeBasedBucketAllocator, "allocate", "size", "dtype")
            self.installed = True
        self.steps.append(
            dict(event="before_dataload", iteration=iteration, counters=memory_counters(), pools=stream_pools())
        )

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if self.installed:
            self.steps.append(
                dict(event="update_end", iteration=iteration, counters=memory_counters(), pools=stream_pools())
            )

    def on_train_end(self, model, iteration=0):
        for cls, method, original in self.patches:
            setattr(cls, method, original)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.with_name(f"{self.output.stem}_rank_{os.environ['RANK']}.json").write_text(
            json.dumps(
                dict(
                    allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                    steps=self.steps,
                    allocations=self.allocations,
                ),
                indent=2,
            )
            + "\n"
        )
