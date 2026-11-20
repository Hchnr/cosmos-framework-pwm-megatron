# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Parallel frozen preparation, followed by the unchanged ordered pipeline.

Each model-parallel rank advances a disjoint subset of the group's logical
streams. CPU/Gloo objects are used only at update/checkpoint boundaries; each
prepared microbatch uses the existing explicit tensor schema on local shared
storage. The original owner remains the pipeline input and checkpoint owner.
"""

import datetime
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .planning import completed_sample_totals, preflight_update, prepare_slot
from .transport import decode, encode


def partition_logical_ranks(logical_ranks, workers, index):
    if workers < 1 or not 0 <= index < workers or len(logical_ranks) % workers:
        raise ValueError("Parallel preparation requires equally sized nonempty logical partitions")
    result = tuple(logical_ranks[index::workers])
    if not result:
        raise ValueError("Parallel preparation needs at least one logical stream per worker")
    return result


def merge_logical_states(parts, expected):
    """Reject missing/duplicated streams before publishing an owner checkpoint."""
    merged = {"loader": {}, "rng": {}}
    for part in parts:
        if set(part) != set(merged) or set(part["loader"]) != set(part["rng"]):
            raise ValueError("Preparation loader/RNG ownership differs")
        for field in merged:
            if set(merged[field]) & set(part[field]):
                raise ValueError("Duplicate logical preparation stream")
            merged[field].update(part[field])
    if set(merged["loader"]) != set(expected):
        raise ValueError("Incomplete logical preparation checkpoint")
    return merged


class PreparedSpool:
    """One CPU staging batch at a time; no raw media or CUDA tensors retained."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, slot):
        update, micro, rank = slot
        return self.directory / f"update{update:09d}.micro{micro:03d}.rank{rank:03d}.pt"

    def write(self, batch):
        header, buffers = encode(batch, device="cpu")
        target = self.path(batch["slot_id"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save((header, buffers), handle)
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target.stat().st_size

    def load(self, slot, *, device="cuda"):
        header, buffers = torch.load(self.path(slot), map_location="cpu", weights_only=False)
        value = decode(header, {dtype: tensor.to(device=device) for dtype, tensor in buffers.items()})
        if tuple(value["slot_id"]) != tuple(slot):
            raise ValueError("Prepared spool slot identity mismatch")
        return value


class ParallelPreparation:
    def __init__(self, model_group, logical_ranks, root):
        self.model_group = model_group
        self.members = dist.get_process_group_ranks(model_group)
        self.index = self.members.index(dist.get_rank())
        self.owner = self.index == 0
        self.source = self.members[0]
        self.logical_ranks = tuple(logical_ranks)
        self.assigned = partition_logical_ranks(logical_ranks, len(self.members), self.index)
        # Every rank creates disjoint groups in the same global order.
        membership = torch.tensor(self.members, dtype=torch.int64, device="cuda")
        memberships = [torch.empty_like(membership) for _ in range(dist.get_world_size())]
        dist.all_gather(memberships, membership)
        timeout = datetime.timedelta(seconds=int(os.environ.get("COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC", "7200")))
        for ranks in sorted({tuple(item.cpu().tolist()) for item in memberships}):
            group = dist.new_group(list(ranks), backend="gloo", timeout=timeout)
            if dist.get_rank() in ranks:
                self.cpu_group = group
        for rank in range(dist.get_world_size()):
            group = dist.new_group([rank], backend="nccl", timeout=timeout)
            if rank == dist.get_rank():
                self.singleton = group
        self.root = Path(root) / "prepared_spool"
        self.spool = None
        self.loader = self.rng = None

    def restart_partition(self, restored):
        payload = [
            {"loader": restored["logical_loader"], "rng": restored["logical_training_rng"]} if self.owner else None
        ]
        dist.broadcast_object_list(payload, src=self.source, group=self.cpu_group)
        complete = merge_logical_states(payload, self.logical_ranks)
        return {field: {rank: complete[field][rank] for rank in self.assigned} for field in complete}

    def checkpoint_state(self):
        local = {"loader": self.loader.logical_state_dict(), "rng": self.rng.state_dict()}
        gathered = [None] * len(self.members) if self.owner else None
        dist.gather_object(local, gathered, dst=self.source, group=self.cpu_group)
        return merge_logical_states(gathered, self.logical_ranks) if self.owner else None

    def prepare(self, slots, topology, preparation, *, preflight, trace_prepared, capture_dir, capture_iteration):
        started = time.monotonic()
        local_slots = [slot for slot in slots if slot.logical_rank in self.assigned]
        directory = [None]
        if self.owner:
            self.root.mkdir(parents=True, exist_ok=True)
            directory[0] = tempfile.mkdtemp(prefix=f"iter{slots[0].update:09d}.", dir=self.root)
        dist.broadcast_object_list(directory, src=self.source, group=self.cpu_group)
        self.spool = PreparedSpool(directory[0])
        preflight_started = time.monotonic()
        if preflight:
            self.loader, records, totals = preflight_update(
                self.loader,
                local_slots,
                topology,
                preparation,
                owner=True,
                model_group=self.singleton,
                source_rank=dist.get_rank(),
                dp_group=dist.group.WORLD,
            )
        else:
            records, totals = [{} for _ in local_slots], None
        preflight_seconds = time.monotonic() - preflight_started
        byte_count = 0
        for slot, record in zip(local_slots, records, strict=True):
            batch = prepare_slot(
                self.loader,
                slot,
                record,
                totals,
                topology,
                preparation,
                self.rng,
                hash_raw=preflight,
                trace_prepared=trace_prepared,
                capture_dir=capture_dir if capture_iteration is None or capture_iteration == slot.update else None,
            )
            spool_started = time.monotonic()
            byte_count += self.spool.write(batch)
            record["host_seconds"]["prepared_spool_write"] = time.monotonic() - spool_started
            del batch
        if totals is None:
            totals = completed_sample_totals(
                records,
                topology,
                owner=True,
                model_group=self.singleton,
                source_rank=dist.get_rank(),
                dp_group=dist.group.WORLD,
            )
        else:
            totals = totals["samples"]
        gathered = [None] * len(self.members)
        dist.all_gather_object(gathered, records, group=self.cpu_group)
        ordered = sorted((row for part in gathered for row in part), key=lambda row: tuple(row["slot"]))
        expected = [(slot.update, slot.micro_step, slot.logical_rank) for slot in slots]
        if [tuple(row["slot"]) for row in ordered] != expected:
            raise ValueError("Parallel preparation changed the pipeline slot order or coverage")
        # Also releases transient VAE workspaces before original-capacity 1F1B.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return (
            ordered,
            totals,
            {
                "parallel_prepare_wall_seconds": time.monotonic() - started,
                "local_preflight_seconds": preflight_seconds,
                "local_prepared_spool_bytes": byte_count,
                "local_preparation_slots": len(local_slots),
            },
        )

    def release(self):
        dist.barrier(group=self.cpu_group)
        if self.owner and self.spool is not None:
            if self.spool.directory.parent.resolve() != self.root.resolve():
                raise ValueError("Prepared spool escapes this job's scratch directory")
            shutil.rmtree(self.spool.directory)
        self.spool = None

    def describe(self):
        return json.dumps({"assigned_logical_ranks": self.assigned, "members": self.members})
