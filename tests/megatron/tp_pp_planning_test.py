"""Streaming and preflight preserve native inputs, RNG and loss denominators."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.test.shared_prefetch_test import _build, _Processor
from cosmos_framework.training_backends.megatron_tp_pp import planning
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.logical_loader import LogicalDataLoader


class StaticPacker:
    def __init__(self, collator):
        self.collator = collator

    def collate(self, samples):
        batch = self.collator.collate(samples)
        batch["packed_sequence"] = PackedSequence(sample_lens=[2] * len(samples))
        return batch


def native(rank):
    loader = _build(workers=0, logical=(0, 0), processor=_Processor(stochastic=True), n=37)
    for stream in loader._loaders:
        stream.dataset._dp_rank, stream.dataset._dp_world_size = rank, 2
        stream.dataset._collator = StaticPacker(stream.dataset._collator)
    loader._base_seeds = [4000 + rank * 10 + index for index in range(2)]
    loader._preserve_resume_rng = True
    return loader


def test_preflight_then_replay_keeps_samples_augmentations_and_weights(monkeypatch):
    states = {}
    for rank in (0, 1):
        reference = native(rank)
        callback = JointCosmosDataLoaderStateCallback(reference)
        callback._update_state_from_batch(next(reference))
        states[rank] = callback.state_dict()
        reference.close()
    loader = LogicalDataLoader(native(0), states, num_workers=0, max_live_streams=1)
    topology = ModelParallelConfig(global_packed_microbatches_per_update=6, logical_data_parallel_size=2)
    slots = logical_slots(topology, update=1200, data_parallel_size=1, data_parallel_rank=0)
    preparation = SimpleNamespace(
        is_image_batch=lambda raw: raw["dataset_name"] == "source_0",
        config=SimpleNamespace(rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=True)),
    )
    preparation.prepare = lambda raw, iteration: {
        "sample_count": len(raw["packed_sequence"].sample_lens),
        "packed_sequence": raw["packed_sequence"],
        "loss_inputs": {"is_image_batch": preparation.is_image_batch(raw)},
    }
    original_zeros, original_empty = torch.zeros, torch.empty
    monkeypatch.setattr(planning.torch, "zeros", lambda *a, **kw: original_zeros(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(planning.torch, "empty", lambda *a, **kw: original_empty(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(planning.dist, "broadcast", lambda *a, **kw: None)
    monkeypatch.setattr(planning.dist, "all_reduce", lambda *a, **kw: None)
    replay, records, totals = planning.preflight_update(
        loader, slots, topology, preparation, owner=True, model_group=None, source_rank=0, dp_group=None
    )
    try:
        assert sum(sum(row) for row in totals) == sum(record["samples"] for record in records)
        for slot, record in zip(slots, records, strict=True):
            result = planning.prepare_slot(
                replay,
                slot,
                record,
                totals,
                topology,
                preparation,
                SimpleNamespace(activate=lambda rank: nullcontext()),
            )
            assert result["weight"] == 1.0  # Both native logical ranks have equal packed sizes.
            assert result["slot_id"] == (1200, slot.micro_step, slot.logical_rank)
        assert all(state["global_id"] == 4 for state in replay.logical_state_dict().values())
    finally:
        replay.close()


def test_raw_fingerprint_ignores_timing_but_detects_changed_pixels():
    raw = {"video": torch.arange(12, dtype=torch.uint8), "_sample_time": 0.1, "_cosmos_loader": {"timing": 1}}
    same = copy.deepcopy(raw)
    same["_sample_time"] = 9.0
    same["_cosmos_loader"]["timing"] = 20
    assert planning.batch_fingerprint(raw) == planning.batch_fingerprint(same)
    same["video"][0] = 255
    assert planning.batch_fingerprint(raw) != planning.batch_fingerprint(same)
    with pytest.raises(TypeError, match="fingerprint"):
        planning.batch_fingerprint({"opaque": object()})


@pytest.mark.parametrize("workers", [0, 2])
def test_single_pass_matches_double_pass_inputs_and_committed_loader_rng(monkeypatch, workers):
    states = {}
    for rank in (0, 1):
        loader = native(rank)
        callback = JointCosmosDataLoaderStateCallback(loader)
        callback._update_state_from_batch(next(loader))
        states[rank] = callback.state_dict()
        loader.close()
    topology = ModelParallelConfig(global_packed_microbatches_per_update=6, logical_data_parallel_size=2)
    slots = logical_slots(topology, update=1200, data_parallel_size=1, data_parallel_rank=0)
    preparation = SimpleNamespace(
        is_image_batch=lambda raw: raw["dataset_name"] == "source_0",
        config=SimpleNamespace(rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=False)),
    )
    preparation.prepare = lambda raw, step: {
        "sample_count": len(raw["packed_sequence"].sample_lens),
        "loss_inputs": {"is_image_batch": preparation.is_image_batch(raw)},
    }
    for name in ("zeros", "empty", "tensor"):
        original = getattr(torch, name)
        monkeypatch.setattr(
            planning.torch, name, lambda *a, _original=original, **kw: _original(*a, **{**kw, "device": "cpu"})
        )
    monkeypatch.setattr(planning.dist, "broadcast", lambda *a, **kw: None)
    monkeypatch.setattr(planning.dist, "all_reduce", lambda *a, **kw: None)
    observations = []
    fingerprints = []
    for mode in ("preflight", "hashed", "unhashed"):
        preflight = mode == "preflight"
        loader = LogicalDataLoader(native(0), copy.deepcopy(states), num_workers=workers, max_live_streams=1)
        if mode == "unhashed":

            def unexpected_hash(raw):
                raise AssertionError("Normal streaming must not scan raw media for a fingerprint")

            loader.fingerprint_batch = unexpected_hash
        updates = []
        update_hashes = []
        try:
            for update in (1200, 1201):
                slots = logical_slots(topology, update=update, data_parallel_size=1, data_parallel_rank=0)
                if preflight:
                    loader, records, totals = planning.preflight_update(
                        loader, slots, topology, preparation, owner=True, model_group=None, source_rank=0, dp_group=None
                    )
                else:
                    records, totals = [{} for _ in slots], None
                for slot, record in zip(slots, records, strict=True):
                    result = planning.prepare_slot(
                        loader,
                        slot,
                        record,
                        totals,
                        topology,
                        preparation,
                        SimpleNamespace(activate=lambda _: nullcontext()),
                        hash_raw=mode != "unhashed",
                    )
                    assert "weight" not in result
                actual_totals = planning.completed_sample_totals(
                    records, topology, owner=True, model_group=None, source_rank=0, dp_group=None
                )
                if totals is not None:
                    assert actual_totals == totals
                updates.append(
                    (
                        [
                            {k: v for k, v in record.items() if k not in ("host_seconds", "sha256")}
                            for record in records
                        ],
                        actual_totals,
                        planning.batch_fingerprint(loader.logical_state_dict()),
                    )
                )
                update_hashes.append([record["sha256"] for record in records])
            observations.append(updates)
            fingerprints.append(update_hashes)
        finally:
            loader.close()
    assert observations[0] == observations[1] == observations[2]
    assert fingerprints[0] == fingerprints[1]
    assert all(value is None for update in fingerprints[2] for value in update)
