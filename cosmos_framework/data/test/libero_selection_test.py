"""Protect explicit episode manifests against leakage and accidental double splitting."""

import pytest

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset


def select(ids, split="full"):
    return LIBEROLeRobotDataset._select_episode_ids([1, 4, 9], split, 0.2, 42, ids)


def test_disjoint_manifests():
    assert select([1, 9]) == {1, 9}
    assert select([4]) == {4}
    assert select([1, 9]).isdisjoint(select([4]))


@pytest.mark.parametrize("ids,split", [([], "full"), ([1, 1], "full"), ([2], "full"), ([True], "full"), ([1], "train")])
def test_invalid_manifest_fails(ids, split):
    with pytest.raises(ValueError):
        select(ids, split)


def test_default_split_unchanged():
    train = select(None, "train")
    val = select(None, "val")
    assert train.isdisjoint(val)
    assert train | val == {1, 4, 9}
