"""Virtual DP streams match native samples, augmentations and restart state."""

import copy

import pytest
import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.data.test.shared_prefetch_test import _build, _Processor
from cosmos_framework.training_backends.megatron_tp_pp.logical_loader import LogicalDataLoader
from cosmos_framework.training_backends.megatron_tp_pp.planning import batch_fingerprint


def native(rank):
    loader = _build(workers=0, depth=2, logical=(2, 1), processor=_Processor(stochastic=True), n=37)
    for stream in loader._loaders:
        stream.dataset._dp_rank, stream.dataset._dp_world_size = rank, 4
    loader._base_seeds = [1000 + rank * 10 + source for source in range(2)]
    loader._preserve_resume_rng = True
    return loader


def test_raw_fingerprints_survive_redistributing_logical_ranks_between_dp_owners():
    states = {}
    for rank in range(4):
        reference = native(rank)
        callback = JointCosmosDataLoaderStateCallback(reference)
        callback._update_state_from_batch(next(reference))
        states[rank] = copy.deepcopy(callback.state_dict())
        reference.close()
    expected = {}
    for ranks in ((0, 1, 2, 3), (0, 2), (1, 3)):
        pool = LogicalDataLoader(native(0), {rank: states[rank] for rank in ranks}, num_workers=0)
        try:
            for step in range(2):
                for rank in ranks:
                    batch = next(pool)
                    local_id = batch["_cosmos_loader"]["global_id"]
                    digest = batch_fingerprint(pool.fingerprint_batch(batch))
                    assert batch["_cosmos_loader"]["global_id"] == local_id
                    if len(ranks) == 4:
                        assert digest == batch_fingerprint(batch)
                        expected[step, rank] = digest
                    else:
                        assert digest == expected[step, rank]
                    pool.commit_batch(batch)
            assert all(state["global_id"] == 3 for state in pool.logical_state_dict().values())
        finally:
            pool.close()


@pytest.mark.parametrize("workers,resume_workers", [(0, 2), (2, 1)])
def test_virtual_streams_keep_native_batch_rng_and_restart(workers, resume_workers):
    states, expected = {}, {}
    for rank in (1, 3):
        reference = native(rank)
        callback = JointCosmosDataLoaderStateCallback(reference)
        for _ in range(2):
            callback._update_state_from_batch(next(reference))
        states[rank] = copy.deepcopy(callback.state_dict())
        expected[rank] = []
        for _ in range(6):
            batch = next(reference)
            expected[rank].append(copy.deepcopy(batch))
            callback._update_state_from_batch(batch)
        reference.close()
    template = native(0)
    pool = LogicalDataLoader(template, states, num_workers=workers, prefetch_factor=4, max_live_streams=1)

    def consume(pool, begin):
        for step in range(begin, begin + 3):
            for rank in (1, 3):
                batch = next(pool)
                assert batch["logical_data_rank"] == rank
                ref = expected[rank][step]
                assert batch["dataset_name"] == ref["dataset_name"]
                for key in ("id", "augmentation", "video", "mask", "sample_worker_id", "sample_epoch", "sample_index"):
                    torch.testing.assert_close(batch[key], ref[key], rtol=0, atol=0)
                pool.commit_batch(batch)

    try:
        consume(pool, 0)
        saved = pool.logical_state_dict()
        assert all(state["global_id"] == 5 for state in saved.values())
        assert not template._started and len(pool._processes) == workers
        assert pool.prefetch_stats["max_observed_inflight"] <= 4
        if not workers:
            assert pool._runner.maximum_live_streams == 1
    finally:
        pool.close()
    restarted = LogicalDataLoader(template, saved, num_workers=resume_workers, prefetch_factor=3, max_live_streams=1)
    try:
        consume(restarted, 3)
        assert all(state["global_id"] == 8 for state in restarted.logical_state_dict().values())
    finally:
        restarted.close()
