# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise actual episode iteration and checkpoint callbacks across DP changes."""

import copy
import itertools
import os
from unittest.mock import patch

import pytest
import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor
from cosmos_framework.data.generator.dataflow import IdentityProcessor, SequentialPackingBatcher, SimpleBatcher
from cosmos_framework.data.generator.dataflow.collators import DefaultBatchCollator
from cosmos_framework.data.generator.dataflow.loader import _DataflowIterableDataset
from cosmos_framework.data.generator.dataflow.reshard import redistribute_episode_states
from cosmos_framework.data.generator.dataflow.shared_prefetch import CosmosDataStream, SharedPrefetchDataLoader


@pytest.fixture(autouse=True)
def _resume_environment():
    # Resume env vars are consumed in forked children, leaving the parent's
    # copy alive just as in the production loader's existing resume tests.
    with patch.dict(os.environ):
        yield


class _Episodes:
    def get_shuffle_blocks(self):
        lengths = [1, 3, 2, 5] * 6
        starts = itertools.accumulate([0, *lengths])
        return list(zip(starts, lengths, strict=False))

    @property
    def num_shuffle_blocks(self):
        return 24

    def get_item(self, index, *, epoch):
        return {"id": torch.tensor(index), "epoch": torch.tensor(epoch)}


def _loader(rank, size, *, physical=0, logical=0, packed=False):
    streams = {}
    for name in ["a", "b"]:
        ds = _DataflowIterableDataset(
            EpisodeBlockDistributor(_Episodes(), seed=42, name=name, dataset_fingerprint=name),
            _RandomProcessor() if packed else IdentityProcessor(),
            SequentialPackingBatcher(max_samples_per_batch=2, lookahead_limit=1) if packed else SimpleBatcher(1),
            DefaultBatchCollator(),
            rank,
            size,
        )
        streams[name] = {"dataloader": CosmosDataStream(ds, logical), "ratio": 1}
    return SharedPrefetchDataLoader(
        streams, num_workers=physical, prefetch_factor=2, seed=42, selection_mode="exact_periodic"
    )


class _RandomProcessor(IdentityProcessor):
    def process(self, item):
        return {
            **item,
            "augmentation": torch.rand(()),
            "video": torch.zeros((3, 1, 32, 32), dtype=torch.uint8),
            "text_token_ids": torch.arange(2),
        }


def _consume(loader, callback, count):
    rows = []
    for _ in range(count):
        batch = next(loader)
        callback._update_state_from_batch(batch)
        worker = int(batch["sample_worker_id"][0])
        rows.append((batch["dataset_name"], worker, int(batch["id"][0]), int(batch["epoch"][0])))
    return rows


@pytest.mark.parametrize("physical", [0, 2])
@pytest.mark.parametrize("logical", [0, 2])
def test_shrink_keeps_all_original_shard_sequences_and_survives_another_resume(physical, logical):
    states, expected, seen = [], {}, set()
    for rank in range(3):
        loader = _loader(rank, 3, logical=logical)
        callback = JointCosmosDataLoaderStateCallback(loader)
        # Equal batch cursor but unequal source-local lengths/epochs.
        for name, _, index, epoch in _consume(loader, callback, 54):
            seen.add((name, index, epoch))
        states.append(copy.deepcopy(callback.state_dict()))
        for name, worker, index, epoch in _consume(loader, callback, 60):
            address = rank * max(1, logical) + worker
            expected.setdefault((name, address), []).append((index, epoch))
        loader.close()
    migrated = redistribute_episode_states(states, 2)
    observed = {}
    for rank, state in enumerate(migrated):
        loader = _loader(rank, 2, physical=physical)
        callback = JointCosmosDataLoaderStateCallback(loader)
        callback.load_state_dict(state)
        names = loader._names
        addresses = state["__execution__"]["virtual_shards"]
        first = _consume(loader, callback, 12)
        saved = copy.deepcopy(callback.state_dict())
        # Includes real prefetched-but-uncommitted work when physical > 0.
        loader.close()
        resumed = _loader(rank, 2, physical=physical)
        cb = JointCosmosDataLoaderStateCallback(resumed)
        cb.load_state_dict(saved)
        following = _consume(resumed, cb, 48)
        resumed.close()
        for name, worker, index, epoch in [*first, *following]:
            assert (name, index, epoch) not in seen
            address = addresses[names.index(name)][worker][0]
            observed.setdefault((name, address), []).append((index, epoch))
    assert set(observed) == set(expected)
    for key, rows in observed.items():
        assert rows == expected[key][: len(rows)]
    # A second DP reduction must retain the same original address set/cursors.
    twice = redistribute_episode_states(migrated, 1)
    assert len(twice[0]["__execution__"]["virtual_shards"][0]) == 3 * max(1, logical)


def test_reject_missing_rank_or_changed_fingerprint():
    states = []
    for rank in range(3):
        loader = _loader(rank, 3)
        cb = JointCosmosDataLoaderStateCallback(loader)
        _consume(loader, cb, 2)
        states.append(copy.deepcopy(cb.state_dict()))
        loader.close()
    with pytest.raises(ValueError, match="all rank files"):
        redistribute_episode_states(states[:2], 2)
    states[-1]["__fingerprints__"]["a"] = "changed"
    with pytest.raises(ValueError, match="fingerprints"):
        redistribute_episode_states(states, 2)


def test_packed_shrink_preserves_per_shard_batches_and_augmentation_rng():
    states, expected = [], {}

    def consume(loader, callback, count, addresses, output):
        for _ in range(count):
            batch = next(loader)
            callback._update_state_from_batch(batch)
            source = loader._names.index(batch["dataset_name"])
            worker = int(batch["sample_worker_id"][0])
            key = (source, addresses[source][worker][0])
            output.setdefault(key, []).append(
                (batch["id"].tolist(), batch["epoch"].tolist(), batch["augmentation"].tolist())
            )

    for rank in range(3):
        loader = _loader(rank, 3, physical=2, packed=True)
        callback = JointCosmosDataLoaderStateCallback(loader)
        addresses = [[[rank, 3]]] * 2
        consume(loader, callback, 6, addresses, {})
        states.append(copy.deepcopy(callback.state_dict()))
        consume(loader, callback, 20, addresses, expected)
        loader.close()
    actual = {}
    for rank, state in enumerate(redistribute_episode_states(states, 2)):
        loader = _loader(rank, 2, physical=2, packed=True)
        callback = JointCosmosDataLoaderStateCallback(loader)
        callback.load_state_dict(state)
        consume(loader, callback, 12, state["__execution__"]["virtual_shards"], actual)
        loader.close()
    assert actual.keys() == expected.keys()
    for key, batches in actual.items():
        assert batches == expected[key][: len(batches)]


def test_prepared_checkpoint_survives_source_retention_and_refuses_overwrite(tmp_path):
    import pickle
    import shutil

    from tools.reshard_action128_checkpoint import prepare

    source, output = tmp_path / "source", tmp_path / "prepared"
    (source / "dataloader").mkdir(parents=True)
    for component in ["model", "optim", "scheduler", "trainer"]:
        (source / component).mkdir()
        (source / component / ".metadata").write_bytes(b"immutable metadata fixture")
    for rank in range(3):
        loader = _loader(rank, 3)
        cb = JointCosmosDataLoaderStateCallback(loader)
        _consume(loader, cb, 2)
        (source / "dataloader" / f"rank_{rank}.pkl").write_bytes(pickle.dumps(cb.state_dict()))
        loader.close()
    (source / ".save_complete").touch()
    report = prepare(source, 2)
    assert report["original_shards_per_source"] == [3, 3]
    assert not output.exists()
    prepare(source, 2, output)
    assert (source / "model/.metadata").stat().st_ino == (output / "model/.metadata").stat().st_ino
    with pytest.raises(ValueError, match="new directory"):
        prepare(source, 2, output)
    shutil.rmtree(source)
    assert (output / ".save_complete").is_file()
    assert (output / "model/.metadata").read_bytes() == b"immutable metadata fixture"
