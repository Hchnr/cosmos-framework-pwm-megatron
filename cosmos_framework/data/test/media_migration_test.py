"""A new media root may retain cursors only after semantic ordering checks."""

import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_framework.data.generator.action.media_migration import validate_media_migration
from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset, build_action128_spec
from cosmos_framework.data.test.action128_test import source  # noqa: F401


def _pair(source, tmp_path):
    root, _, _ = source
    new = tmp_path / "publication" / root.name
    shutil.copytree(root, new)
    path = new / "meta/episodes/chunk-000/episodes.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["caption_path"] = str(root / row["caption_path"])
        for view in row["video_views"]:
            view.update(path="recoded.mp4", width=1024, height=1024, fps=15.0)
    pq.write_table(pa.Table.from_pylist(rows), path)
    spec = build_action128_spec([str(root)])
    return UnifiedAction128Dataset(root, action128_spec=spec), UnifiedAction128Dataset(new, action128_spec=spec)


def test_media_relocation_keeps_order_but_uses_new_strict_fingerprint(source, tmp_path):
    before, after = _pair(source, tmp_path)
    report = validate_media_migration(before, after)
    assert report["training_episodes"] == 1
    assert before.fingerprint != after.fingerprint
    assert list(before.get_shuffle_blocks()) == list(after.get_shuffle_blocks())


@pytest.mark.parametrize("field", ["episode_index", "dataset_from_index", "action_domain"])
def test_media_migration_rejects_changed_nonmedia_episode_metadata(source, tmp_path, field):
    before, after = _pair(source, tmp_path)
    rows = after.episodes.to_pylist()
    rows[0][field] = "different" if field == "action_domain" else rows[0][field] + 1
    after.episodes = pa.Table.from_pylist(rows, schema=after.episodes.schema)
    with pytest.raises(ValueError, match=field):
        validate_media_migration(before, after)


def test_media_migration_rejects_changed_counts_and_state_contract(source, tmp_path):
    before, after = _pair(source, tmp_path)
    after.counts = after.counts.copy() + 1
    with pytest.raises(ValueError, match="window counts"):
        validate_media_migration(before, after)
    after.counts = before.counts
    after.state_transforms = {"changed": True}
    with pytest.raises(ValueError, match="state_transforms"):
        validate_media_migration(before, after)


@pytest.mark.parametrize("field", ["role", "source_key"])
def test_media_migration_rejects_changed_camera_identity(source, tmp_path, field):
    before, after = _pair(source, tmp_path)
    rows = after.episodes.to_pylist()
    rows[0]["video_views"][0][field] = "different_camera"
    after.episodes = pa.Table.from_pylist(rows, schema=after.episodes.schema)
    with pytest.raises(ValueError, match=field):
        validate_media_migration(before, after)


def test_camera_audit_does_not_concatenate_huge_media_json(source, tmp_path):
    before, after = _pair(source, tmp_path)
    # Share a 2-MiB Arrow buffer across 1025 chunks: logical media JSON exceeds
    # 2 GiB without allocating that much RAM. Combining full view structs would
    # overflow their int32 string offsets; projecting role/source_key is safe.
    blob = "x" * (2 * 1024**2)
    for dataset in [before, after]:
        rows = dataset.episodes.to_pylist()
        rows[0]["video_views"][0]["native_metadata_json"] = blob
        table = pa.Table.from_pylist(rows)
        dataset.episodes = pa.concat_tables([table] * 1025)
        dataset.counts = np.tile(dataset.counts, 1025)
        dataset.ends = np.cumsum(dataset.counts)
    assert validate_media_migration(before, after)["training_episodes"] == 1025
