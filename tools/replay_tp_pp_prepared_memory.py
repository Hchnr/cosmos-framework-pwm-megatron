# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replay the complete original prepared update without a resident VAE.

Reuses production import, compile/full AC, MCore schedule, loss, and optimizer.
Does not advance logical loaders/RNG or publish a resumable checkpoint. This
isolates decoder memory behavior and cannot accept full training continuity.
"""

import argparse
import datetime
import functools
import gc
import json
import os
import time
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn


class ReplayComplete(Exception):
    pass


class NoPreparation(nn.Module):
    def __init__(self, config, input_adapter, process_group=None):
        super().__init__()
        self.config = config


def main():
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import schedules

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.training_backends.megatron_tp_pp import train
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver
    from cosmos_framework.training_backends.megatron_tp_pp.parallel_preparation import PreparedSpool

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-cycles-every", type=int, default=0)
    args = parser.parse_args()
    if args.collect_cycles_every < 0:
        parser.error("Cycle collection interval must be nonnegative")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    directory = args.output / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["OUTPUT_ROOT"] = str(args.output.resolve() / "runtime")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = os.environ["OUTPUT_ROOT"]
    os.environ["COSMOS_TP_PP_PARALLEL_PREPARATION"] = "0"
    os.environ["COSMOS_TP_PP_VERIFY_RAW_REPLAY"] = "0"
    config = load_experiment_from_toml(
        str(args.toml),
        extra_overrides=[
            "job.name=prepared_memory_replay",
            "trainer.callbacks.compile_tokenizer.enabled=false",
        ],
    )
    if config.model.config.rectified_flow_training_config.sample_level_loss_averaging:
        raise ValueError("This decoder-only probe requires the recipe's rank-level loss averaging")
    config.validate()
    spool = PreparedSpool(args.spool)
    expected = config.model.config.parallelism.megatron.global_packed_microbatches_per_update
    if len(list(args.spool.glob("update*.pt"))) != expected:
        raise ValueError("The probe requires a complete original-capacity prepared update")
    initial_source = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in args.spool.glob("update*.pt")}
    forwards, backwards = 0, 0
    in_flight_slots = deque()
    active_forward_slot = active_backward_slot = None
    trace = (directory / "memory.jsonl").open("x")

    def event(kind, **fields):
        trace.write(
            json.dumps(
                {
                    "event": kind,
                    "forwards": forwards,
                    "backwards": backwards,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    **fields,
                }
            )
            + "\n"
        )
        trace.flush()

    original_forward = PipelineStage.forward

    @functools.wraps(original_forward)
    def forward(stage, batch):
        nonlocal forwards, active_forward_slot
        forwards += 1
        slot = tuple(batch["slot_id"])
        active_forward_slot = slot
        packed = batch["packed_sequence"]
        counts = {
            "text": packed.text_ids.numel(),
            "vision": packed.vision.sequence_indexes.numel() if packed.vision is not None else 0,
            "action": packed.action.sequence_indexes.numel() if packed.action is not None else 0,
        }
        event("before_forward", slot=slot, samples=len(packed.sample_lens), tokens=counts)
        value = original_forward(stage, batch)
        in_flight_slots.append(slot)
        active_forward_slot = None
        event("after_forward", slot=slot)
        return value

    original_backward = schedules.backward_step

    @functools.wraps(original_backward)
    def backward(*positional, **keywords):
        nonlocal backwards, active_backward_slot
        if not in_flight_slots:
            raise RuntimeError("Scheduled backward has no recorded completed forward")
        # This probe uses the recipe's non-interleaved 1F1B schedule. Backward
        # consumes completed forwards in FIFO order, including its warmup.
        active_backward_slot = in_flight_slots[0]
        event("before_backward", slot=active_backward_slot)
        value = original_backward(*positional, **keywords)
        backwards += 1
        event("after_backward", slot=active_backward_slot)
        in_flight_slots.popleft()
        active_backward_slot = None
        if args.collect_cycles_every and backwards % args.collect_cycles_every == 0:
            allocated_before = torch.cuda.memory_allocated()
            started = time.monotonic()
            collected = gc.collect()
            event(
                "cycle_collection",
                collected_python_objects=collected,
                reclaimed_allocated_bytes=allocated_before - torch.cuda.memory_allocated(),
                host_seconds=time.monotonic() - started,
            )
        return value

    def replay_slot(loader, slot, *positional, **keywords):
        batch = spool.load((slot.update, slot.micro_step, slot.logical_rank))
        if tuple(batch["slot_id"]) != (slot.update, slot.micro_step, slot.logical_rank):
            raise ValueError("Prepared input identity changed")
        return batch

    original_step = ModelParallelDriver.step

    @functools.wraps(original_step)
    def step(driver):
        result = original_step(driver)
        torch.cuda.synchronize()
        if not result[0] or forwards != driver.microbatches or backwards != driver.microbatches or in_flight_slots:
            raise ValueError("Prepared update was incomplete or nonfinite")
        if {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in args.spool.glob("update*.pt")} != initial_source:
            raise ValueError("Read-only source spool changed")
        report = {
            "passed": True,
            "scope": "one complete prepared original-capacity decoder/optimizer update, no resident VAE; not logical-loader continuity",
            "rank": rank,
            "iteration": driver.iteration,
            "global_slots": expected,
            "forwards": forwards,
            "backwards": backwards,
            "gradient_norm": float(result[1]),
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "source_spool_unchanged": True,
            "checkpoint_published": False,
            "collect_cycles_every": args.collect_cycles_every,
            "production_capacity_576_slot_acceptance": False,
        }
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        event("optimizer_complete")
        raise ReplayComplete

    train.GenerationPreparation = NoPreparation
    train.prepare_slot = replay_slot
    PipelineStage.forward = forward
    schedules.backward_step = backward
    ModelParallelDriver.step = step
    try:
        train.run(
            config,
            initialized_callback=lambda driver: event(
                "runtime_initialized", iteration=driver.iteration, resident_vae=False
            ),
        )
        raise RuntimeError("Replay ended without its controlled completion marker")
    except ReplayComplete:
        pass
    except Exception as error:
        report = {
            "passed": False,
            "scope": "original-capacity decoder-only memory diagnostic; failed before controlled completion",
            "rank": rank,
            "global_slots": expected,
            "forwards_started": forwards,
            "backwards_completed": backwards,
            "active_forward_slot": active_forward_slot,
            "active_backward_slot": active_backward_slot,
            "in_flight_slots": list(in_flight_slots),
            "allocated_cuda_bytes": torch.cuda.memory_allocated(),
            "reserved_cuda_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "source_spool_unchanged": {
                p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in args.spool.glob("update*.pt")
            }
            == initial_source,
            "checkpoint_published": False,
            "collect_cycles_every": args.collect_cycles_every,
            "production_capacity_576_slot_acceptance": False,
        }
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        event("failure", error_type=type(error).__name__)
        # Another stage can still be blocked in P2P when one rank OOMs. NCCL
        # teardown here can then wait indefinitely instead of letting torchrun
        # observe the failed rank and terminate its peers. Evidence is closed
        # before exiting; no checkpoint is published by this diagnostic.
        trace.close()
        print(json.dumps({"rank": rank, "probe_failure": report}), flush=True)
        os._exit(1)
    finally:
        trace.close()
        ps.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
