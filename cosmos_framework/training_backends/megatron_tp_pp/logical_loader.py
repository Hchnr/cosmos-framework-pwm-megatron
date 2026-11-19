# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Virtualize native logical DP streams over one bounded CPU worker pool."""

import copy
import os
from collections import OrderedDict
from functools import partial

from cosmos_framework.data.generator.dataflow.shared_prefetch import (
    LOADER_META_KEY,
    CosmosDataStream,
    SharedPrefetchDataLoader,
    _StreamRunner,
)


class BoundedLogicalRunner(_StreamRunner):
    """Evict inactive packer generators at the exact native resume frontier."""

    def __init__(self, *args, max_live_streams, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_live_streams = max_live_streams
        self._recent = OrderedDict()
        self._frontiers = {}
        self._environment = {}
        self._media = {}
        self._media_budget = None
        self.maximum_live_streams = 0

    def prepare_media_lookaheads(self, assigned_streams, media_budget=None):
        # Open prefetch only when a stream becomes active, preserving the native
        # media optimization without creating thousands of lookahead caches.
        self._media_budget = media_budget

    def _evict(self, key):
        self.iterators.pop(key).close()
        cache = self._media.pop(key, None)
        if cache is not None:
            cache.close()
        source, worker = key
        rng, epoch, position = self._frontiers.pop(key)
        self.rng[key] = rng
        name = self.streams[source].dataset._distributor._name
        for field, value in (("EPOCH", epoch), ("INDEX", position)):
            variable = f"COSMOS_DL_STATE_{name}_WORKER_{worker}_{field}"
            self._environment.setdefault(variable, os.environ.get(variable))
            os.environ[variable] = str(value)

    def _next_batch(self, source, worker):
        key = (source, worker)
        for stale in list(self._recent):
            if stale not in self.iterators:  # Native timeout replacement closed this iterator.
                self._recent.pop(stale)
                self._frontiers.pop(stale, None)
                cache = self._media.pop(stale, None)
                if cache is not None:
                    cache.close()
        if key not in self.iterators and len(self.iterators) >= self.max_live_streams:
            victim, _ = self._recent.popitem(last=False)
            self._evict(victim)
        if key not in self.iterators:
            stream = self.streams[source]
            dataset = stream.dataset
            prepare = getattr(dataset._distributor, "prepare_media_lookahead", None)
            if prepare is not None:
                self._media[key] = prepare(
                    dataset._dp_rank,
                    dataset._dp_world_size,
                    worker,
                    max(1, stream.num_workers),
                    shared_budget=self._media_budget,
                )
        batch = super()._next_batch(source, worker)
        rng = batch.get("_cosmos_resume_rng")
        if rng is None or "sample_epoch" not in batch:
            raise ValueError("Bounded logical streams require exact native cursor/augmentation RNG metadata")
        self._frontiers[key] = (rng, int(batch["sample_epoch"][0]), int(batch["sample_index"][0]))
        self._recent.pop(key, None)
        self._recent[key] = None
        self.maximum_live_streams = max(self.maximum_live_streams, len(self.iterators))
        return batch

    def close(self):
        try:
            super().close()
        finally:
            for cache in self._media.values():
                if cache is not None:
                    cache.close()
            self._media.clear()
            for variable, value in self._environment.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value
            self._environment.clear()


class LogicalDataLoader(SharedPrefetchDataLoader):
    """Preserve native source selection, shard, packer and worker RNG identities.

    ``states`` contains native loader states for the logical ranks assigned to
    one physical DP owner. The template contains unstarted synchronous stream
    specifications; its catalog/media objects are shared, not reconstructed for
    every logical rank. Only this facade starts physical workers.
    """

    def __init__(self, template, states, *, num_workers=None, prefetch_factor=None, max_live_streams=2):
        if template._started or not states:
            raise ValueError("Logical loading requires an unstarted template and explicit native states")
        self.template = template
        if max_live_streams < 1:
            raise ValueError("max_live_streams must be positive")
        self._runner_factory = partial(BoundedLogicalRunner, max_live_streams=max_live_streams)
        self._resume_environment = {}
        self._resume_positions = {}
        self.logical_ranks = tuple(sorted(states))
        self.native_states = copy.deepcopy(states)
        self.source_count = len(template._names)
        self.logical_world_size = None
        self._fingerprints = template.dataset_fingerprints
        cursors = {state["global_id"] for state in states.values()}
        if len(cursors) != 1:
            raise ValueError("Import logical streams at a completed optimizer-update boundary")
        cursor = cursors.pop()
        streams, seeds, rotations, resume_rng = {}, [], [], {}
        for rank_index, rank in enumerate(self.logical_ranks):
            state = states[rank]
            if state.get("__schedule__") != template.schedule_identity:
                raise ValueError(f"Logical rank {rank}: source schedule mismatch")
            if state.get("__fingerprints__") != self._fingerprints:
                raise ValueError(f"Logical rank {rank}: dataset fingerprint mismatch")
            execution = state.get("__execution__")
            if not execution or execution.get("version") != 1 or execution.get("base_seeds") is None:
                raise ValueError(f"Logical rank {rank}: missing native execution/RNG state")
            if "virtual_shards" in execution:
                raise ValueError("Import native logical ranks before applying physical shard redistribution")
            if len(execution["base_seeds"]) != self.source_count or len(execution["next_workers"]) != self.source_count:
                raise ValueError("Incomplete logical source seed/rotation state")
            for source, (name, original) in enumerate(zip(template._names, template._loaders, strict=True)):
                dataset = copy.copy(original.dataset)
                dataset._distributor = copy.copy(dataset._distributor)
                topology = execution["topology"]
                dataset._dp_rank, dataset._dp_world_size = topology["dp"][source]
                if self.logical_world_size is None:
                    self.logical_world_size = dataset._dp_world_size
                elif self.logical_world_size != dataset._dp_world_size:
                    raise ValueError("Logical source streams disagree on the original DP world size")
                if dataset._dp_rank != rank:
                    raise ValueError("Logical loader state has a different data rank")
                if original.num_workers != topology["logical_workers"][source]:
                    raise ValueError("Logical worker topology changed")
                unique = f"{name}::logical_rank_{rank}"
                dataset._distributor._name = unique
                streams[unique] = {"dataloader": CosmosDataStream(dataset, original.num_workers), "ratio": 1}
                offset = rank_index * self.source_count + source
                for worker, position in state.get(name, {}).items():
                    if (source, worker) not in execution["rng"]:
                        raise ValueError(f"Logical rank {rank}: consumed stream is missing augmentation RNG")
                    prefix = f"COSMOS_DL_STATE_{unique}_WORKER_{worker}_"
                    for field, value in (("EPOCH", position["epoch"]), ("INDEX", position["index"])):
                        key = prefix + field
                        self._resume_positions[key] = str(value)
                for (rng_source, worker), rng in execution["rng"].items():
                    if rng_source == source:
                        resume_rng[offset, worker] = rng
            seeds.extend(execution["base_seeds"])
            rotations.extend(execution["next_workers"])
        timeout = template._timeout_config
        super().__init__(
            streams,
            selection_mode="probabilistic",  # selection uses the original schedule below
            num_workers=template.num_workers if num_workers is None else num_workers,
            prefetch_factor=template.prefetch_factor if prefetch_factor is None else prefetch_factor,
            pin_memory=template.pin_memory,
            pin_memory_in_background=template.pin_memory_in_background,
            receive_in_background=template.receive_in_background,
            worker_timeout=template.worker_timeout,
            microbatch_timeout_sec=timeout[0] if timeout else 0,
            microbatch_timeout_max_skips=timeout[1] if timeout else 0,
            microbatch_skip_log_dir=timeout[2] if timeout else None,
        )
        for rank_index, rank in enumerate(self.logical_ranks):
            expected = states[rank]["__execution__"]["topology"]["active_workers"]
            actual = self._active[rank_index * self.source_count : (rank_index + 1) * self.source_count]
            if actual != expected:
                raise ValueError("Active logical worker assignment changed")
        self._base_seeds = seeds
        self._next_worker = list(rotations)
        self._committed_worker = list(rotations)
        self._resume_rng = resume_rng
        self._committed_rng = dict(resume_rng)
        self._preserve_resume_rng = True
        self.set_start_iteration(cursor * len(self.logical_ranks))

    def dataset_index(self, global_id):
        native_id, rank_index = divmod(global_id, len(self.logical_ranks))
        return rank_index * self.source_count + self.template.dataset_index(native_id)

    def _restore_environment(self):
        for key, value in self._resume_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._resume_environment.clear()

    def _start(self):
        for key, value in self._resume_positions.items():
            self._resume_environment[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            super()._start()
        except BaseException:
            self._restore_environment()
            raise
        if self.num_workers:
            # Forked CPU workers own their copies. Do not leave stale restore
            # cursors in the trainer's environment for a later pool instance.
            self._restore_environment()

    def close(self):
        try:
            super().close()
        finally:
            self._restore_environment()

    def __next__(self):
        batch = super().__next__()
        source = batch[LOADER_META_KEY]["source"]
        rank_index, native_source = divmod(source, self.source_count)
        batch["dataset_name"] = self.template._names[native_source]
        batch["logical_data_rank"] = self.logical_ranks[rank_index]
        return batch

    def commit_batch(self, batch):
        super().commit_batch(batch)
        rank = batch["logical_data_rank"]
        per_source = self.native_states[rank].setdefault(batch["dataset_name"], {})
        for worker, epoch, position in zip(
            batch["sample_worker_id"].tolist(),
            batch["sample_epoch"].tolist(),
            batch["sample_index"].tolist(),
            strict=True,
        ):
            old = per_source.get(worker)
            if old is None or (epoch, position) > (old["epoch"], old["index"]):
                per_source[worker] = {"epoch": epoch, "index": position}

    def fingerprint_batch(self, batch):
        """Canonical transport numbering, without changing the commit cursor.

        Each physical DP owner has a local flattened pool. Its local source and
        delivery IDs differ across topologies even when all samples/RNG match.
        Hash their original logical identities, retaining every other raw field.
        """
        result = dict(batch)
        metadata = dict(batch[LOADER_META_KEY])
        native_id, rank_index = divmod(metadata["global_id"], len(self.logical_ranks))
        rank = batch["logical_data_rank"]
        if self.logical_ranks[rank_index] != rank:
            raise ValueError("Logical fingerprint rank disagrees with the physical pool delivery")
        metadata["global_id"] = native_id * self.logical_world_size + rank
        metadata["source"] = rank * self.source_count + metadata["source"] % self.source_count
        result[LOADER_META_KEY] = metadata
        return result

    def logical_state_dict(self):
        count = len(self.logical_ranks)
        if self._committed_global_id % count:
            raise ValueError("Checkpoint logical streams after completing every rank's micro-step")
        result = copy.deepcopy(self.native_states)
        for rank_index, rank in enumerate(self.logical_ranks):
            state = result[rank]
            state["global_id"] = self._committed_global_id // count
            begin, end = rank_index * self.source_count, (rank_index + 1) * self.source_count
            state["__execution__"].update(
                base_seeds=self._base_seeds[begin:end],
                next_workers=self._committed_worker[begin:end],
                rng={
                    (source - begin, worker): rng
                    for (source, worker), rng in self._committed_rng.items()
                    if begin <= source < end
                },
            )
        return result
