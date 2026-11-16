# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint->restart resume parity for CosmosDataLoader(MapDistributor) using
CosmosDataLoaderStateCallback. Single process, num_workers=0."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import (
    CosmosDataLoaderStateCallback,
    JointCosmosDataLoaderStateCallback,
    attach_cosmos_dataloader_state_callback,
)
from cosmos_framework.checkpoint.dcp import _DataloaderWrapper
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
    PoolPackingBatcher,
    SequentialPackingBatcher,
)
from cosmos_framework.data.generator.dataflow.loader import JointCosmosDataLoader
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY


class _IdDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"id": torch.tensor(idx)}


def _build(seed=0):
    return CosmosDataLoader(
        distributor=MapDistributor(_IdDS(20), shuffle=False, seed=seed),
        processor=IdentityProcessor(),
        batch_size=1,
        num_workers=0,
    )


@pytest.mark.parametrize("joint", [False, True])
def test_resume_preserves_cpu_training_rng_and_next_batches(joint):
    """A restarted iterator must not shift the diffusion timestep RNG stream."""

    def build():
        if not joint:
            loader = _build()
            return loader, CosmosDataLoaderStateCallback()
        loaders = {
            name: {
                "dataloader": CosmosDataLoader(
                    MapDistributor(_IdDS(100), shuffle=False, name=name), IdentityProcessor(), batch_size=1
                ),
                "ratio": 1,
            }
            for name in ("rng_a", "rng_b", "rng_c")
        }
        loader = JointCosmosDataLoader(loaders, seed=42, selection_mode="exact_periodic")
        return loader, JointCosmosDataLoaderStateCallback(loader)

    with patch.dict(os.environ), torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        original, callback = build()
        iterator = iter(original)

        def updates(iterator, callback, count):
            outputs = []
            for _ in range(count * 2):  # GA2, plus CPU randomness as in the timestep sampler
                batch = next(iterator)
                callback._update_state_from_batch(batch)
                outputs.append((batch["id"].item(), batch.get("dataset_name"), torch.rand(7)))
            return outputs

        updates(iterator, callback, 5)
        saved_state = callback.state_dict()
        saved_rng = torch.get_rng_state()
        expected = updates(iterator, callback, 5)
        expected_rng = torch.get_rng_state()
        restored, restored_callback = build()
        restored_callback.load_state_dict(saved_state)
        torch.set_rng_state(saved_rng)
        restored.preserve_rng_on_resume()
        actual = updates(iter(restored), restored_callback, 5)
        for (index, name, sigma), (ref_index, ref_name, ref_sigma) in zip(actual, expected):
            assert (index, name) == (ref_index, ref_name)
            torch.testing.assert_close(sigma, ref_sigma, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        assert restored_callback.state_dict() == callback.state_dict()


def test_fresh_loader_keeps_existing_iterator_seed_behavior():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        torch.empty((), dtype=torch.int64).random_()  # PyTorch's existing base_seed draw
        expected = torch.get_rng_state()
        torch.manual_seed(42)
        next(iter(_build()))
        torch.testing.assert_close(torch.get_rng_state(), expected, rtol=0, atol=0)


def test_resume_continues_without_dup_or_skip():
    cb = CosmosDataLoaderStateCallback()
    loader = _build()
    it = iter(loader)
    seen_ids = []
    for _ in range(5):
        b = next(it)
        cb._update_state_from_batch(b)
        seen_ids.append(b["id"].item())
    assert seen_ids == [0, 1, 2, 3, 4]

    state = cb.state_dict()
    assert state[0]["index"] == 4
    cb2 = CosmosDataLoaderStateCallback()
    cb2.load_state_dict(state)

    loader2 = _build()
    it2 = iter(loader2)  # one iterator: env-var fast-forward happens once, then continues
    resumed = [next(it2)["id"].item() for _ in range(3)]
    assert resumed == [5, 6, 7]


def test_instantiated_loader_attachment_skips_inactive_legacy_callback():
    inactive = SimpleNamespace(checkpoint_component="dataloader", has_checkpoint_state=lambda: False)
    callbacks = SimpleNamespace(_callbacks=[inactive])
    config = object()
    trainer = SimpleNamespace(callbacks=callbacks, config=config)

    assert attach_cosmos_dataloader_state_callback(trainer, _build())
    wrapper = _DataloaderWrapper(callbacks)

    assert wrapper.has_state()
    assert wrapper._callback.config is config
    assert wrapper._callback.trainer is trainer


class _VideoProcessor(IdentityProcessor):
    def process(self, item):
        item_id = int(item["id"])
        return {
            "id": torch.tensor(item_id),
            "text_token_ids": torch.arange(1),
            "video": torch.zeros(3, 1, 32, 32),
        }


class _DropOneVideoProcessor(_VideoProcessor):
    def process(self, item):
        if int(item["id"]) == 1:
            return {DROP_SAMPLE_KEY: True}
        return super().process(item)


def _build_sequential_packing_loader(n: int, *, drop_one: bool = False):
    return CosmosDataLoader(
        distributor=MapDistributor(_IdDS(n), shuffle=False),
        processor=_DropOneVideoProcessor() if drop_one else _VideoProcessor(),
        batcher=SequentialPackingBatcher(max_samples_per_batch=2),
        num_workers=0,
    )


def test_sequential_packing_resume_cursor_allows_dropped_position():
    loader = _build_sequential_packing_loader(4, drop_one=True)

    batch = next(iter(loader))

    assert batch["id"].tolist() == [0, 2]
    assert batch["sample_epoch"].tolist() == [0, 0]
    assert batch["sample_index"].tolist() == [2, 2]


def test_sequential_packing_uses_lexicographic_cursor_across_epoch():
    loader = _build_sequential_packing_loader(3)
    stream = iter(loader)

    first = next(stream)
    second = next(stream)

    assert first["id"].tolist() == [0, 1]
    assert second["id"].tolist() == [2, 0]
    assert second["sample_epoch"].tolist() == [1, 1]
    assert second["sample_index"].tolist() == [0, 0]


class _AlternatingModalityDS(torch.utils.data.Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, idx):
        sample = {
            "id": torch.tensor(idx),
            "input_ids": torch.tensor([idx]),
        }
        if idx % 2 == 0:
            sample["pixel_values"] = torch.tensor([idx])
        else:
            sample["pixel_values_videos"] = torch.tensor([idx])
        return sample


def _build_reordering_pool(
    *,
    enable_map_style_resume: bool,
    map_style_resume_mode: str = "exact",
):
    return CosmosDataLoader(
        distributor=MapDistributor(_AlternatingModalityDS(), shuffle=False, seed=0),
        processor=IdentityProcessor(),
        batcher=PoolPackingBatcher(max_tokens=32, pool_size=6, max_batch_size=2),
        num_workers=0,
        enable_map_style_resume=enable_map_style_resume,
        map_style_resume_mode=map_style_resume_mode,
    )


def test_reordering_pool_rejects_unsafe_exact_resume():
    with pytest.raises(ValueError, match="Map-style resume cannot safely stamp"):
        next(iter(_build_reordering_pool(enable_map_style_resume=True)))


def test_reordering_pool_can_explicitly_opt_out_of_exact_resume():
    batch = next(iter(_build_reordering_pool(enable_map_style_resume=False)))
    assert batch["id"].tolist() == [0, 2]
    assert "sample_worker_id" not in batch
    assert "sample_epoch" not in batch
    assert "sample_index" not in batch


def test_reordering_pool_at_most_once_resume_never_replays_delivered_rows():
    first = next(
        iter(
            _build_reordering_pool(
                enable_map_style_resume=True,
                map_style_resume_mode="at_most_once",
            )
        )
    )
    assert first["id"].tolist() == [0, 2]
    callback = CosmosDataLoaderStateCallback()
    callback._update_state_from_batch(first)
    assert callback.state_dict()[0] == {"epoch": 0, "index": 2}

    callback = CosmosDataLoaderStateCallback()
    callback.load_state_dict({0: {"epoch": 0, "index": 2}})
    resumed = next(iter(_build_reordering_pool(enable_map_style_resume=True, map_style_resume_mode="at_most_once")))
    assert resumed["id"].tolist() == [3, 5]
    assert set(first["id"].tolist()).isdisjoint(resumed["id"].tolist())
