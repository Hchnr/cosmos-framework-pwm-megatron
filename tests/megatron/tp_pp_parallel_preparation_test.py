"""Prepared spooling and checkpoint partitioning preserve logical identities."""

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.training_backends.megatron_tp_pp.parallel_preparation import (
    PreparedSpool,
    merge_logical_states,
    partition_logical_ranks,
)
from cosmos_framework.training_backends.megatron_tp_pp.planning import batch_fingerprint


@pytest.mark.parametrize("dp", [1, 2, 12])
def test_partition_uses_model_group_order_not_physical_world_rank(dp):
    all_ranks = []
    for data_rank in range(dp):
        logical = tuple(range(data_rank, 192, dp))
        parts = [partition_logical_ranks(logical, 4, index) for index in range(4)]
        assert len({len(part) for part in parts}) == 1
        assert sorted(rank for part in parts for rank in part) == list(logical)
        all_ranks.extend(rank for part in parts for rank in part)
    assert sorted(all_ranks) == list(range(192))


def test_checkpoint_merges_back_to_legacy_owner_layout():
    complete = {field: {rank: {"value": torch.tensor(rank)} for rank in range(4)} for field in ("loader", "rng")}
    parts = [{field: {rank: complete[field][rank]} for field in complete} for rank in range(4)]
    actual = merge_logical_states(parts, range(4))
    assert batch_fingerprint(actual) == batch_fingerprint(complete)
    with pytest.raises(ValueError, match="Duplicate"):
        merge_logical_states(parts + parts[:1], range(4))
    with pytest.raises(ValueError, match="Incomplete"):
        merge_logical_states(parts[:-1], range(4))
    parts[0]["rng"] = {}
    with pytest.raises(ValueError, match="ownership"):
        merge_logical_states(parts, range(4))


def test_prepared_spool_roundtrip_and_immutable_publication(tmp_path):
    spool = PreparedSpool(tmp_path)
    batch = {
        "slot_id": (1200, 2, 3),
        "packed_sequence": PackedSequence(
            sample_lens=[2, 3],
            text_ids=torch.tensor([1, 4, 8]),
            text_indexes=torch.tensor([0, 2, 4]),
        ),
        "loss_inputs": {"timesteps": torch.tensor([0.2, 0.7]), "optional": None, "empty": torch.empty(0)},
        "sample_count": 2,
    }
    assert spool.write(batch) > 0
    replay = spool.load(batch["slot_id"], device="cpu")
    assert batch_fingerprint(replay) == batch_fingerprint(batch)
    with pytest.raises(FileExistsError):
        spool.write(batch)
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(ValueError, match="equally sized"):
        partition_logical_ranks((0, 1, 2), 2, 0)
