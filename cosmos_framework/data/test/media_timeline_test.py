import json
import pickle

import numpy as np
import pyarrow as pa
import pytest

from cosmos_framework.data.generator.action.media_timeline import MediaTimeline, episode_views
from cosmos_framework.data.generator.catalog import CatalogSchemaError


def view(**kwargs):
    return dict(role="main", from_timestamp=0.0, to_timestamp=10.0, fps=15.0, source_fps=30.0, time_scale=0.5, **kwargs)


def test_slow_motion_affine_and_legacy(tmp_path):
    mapping = MediaTimeline()
    v = view()
    query, _, _ = mapping.resolve(tmp_path, {}, v, [0, 1, 2])
    np.testing.assert_allclose(query, [0, 0.5, 1])
    del v["time_scale"]
    query, _, _ = mapping.resolve(tmp_path, {}, v, [0, 1, 2])
    np.testing.assert_allclose(query, [0, 1, 2])


def test_native_metadata_nonnull_override():
    views = episode_views(
        {
            "video_views": [
                {
                    "native_metadata_json": json.dumps({"mcap_topic": "left", "time_scale": 0.5, "fps": 30}),
                    "fps": 15,
                    "time_scale": None,
                }
            ]
        }
    )
    assert views == [{"mcap_topic": "left", "time_scale": 0.5, "fps": 15}]


def test_pts_priority_clock_and_native_index(tmp_path):
    np.savez(tmp_path / "pts.npz", left=np.array([0.0, 0.1, 0.2]), main=np.array([0.0, 0.2, 0.4]))
    rows = pa.table(
        {"timestamp": [0.0, 1.0, 2.0], "camera_pts": [100.0, 100.1, 100.2], "source_media_pts": [100.0, 100.2, 100.4]}
    )
    v = view(pts_column="camera_pts", source_clock_origin_s=100.0, timeline_key="left")
    v.update(from_timestamp=100.0, to_timestamp=101.0)
    mapping = MediaTimeline()
    query, indexes, meta = mapping.resolve(tmp_path, {"source_times_path": "pts.npz"}, v, [0, 1, 2], rows)
    np.testing.assert_allclose(query, [0, 0.1, 0.2])
    assert indexes.tolist() == [0, 1, 2] and meta["mapping_rule"] == "camera_pts"
    del v["pts_column"]
    v["timeline_key"] = "main"
    query, _, meta = mapping.resolve(tmp_path, {"source_times_path": "pts.npz"}, v, [0, 1, 2], rows)
    np.testing.assert_allclose(query, [0, 0.2, 0.4])
    assert meta["mapping_rule"] == "source_media_pts"
    assert not pickle.loads(pickle.dumps(mapping)).arrays


def test_missing_pts_and_gap_fail(tmp_path):
    mapping = MediaTimeline()
    with pytest.raises(CatalogSchemaError, match="missing declared"):
        mapping.resolve(tmp_path, {}, view(pts_column="absent"), [0])
    np.savez(tmp_path / "pts.npz", main=np.array([0.0, 0.1, 0.2, 3.0, 3.1]))
    with pytest.raises(CatalogSchemaError, match="no synchronized"):
        mapping.resolve(tmp_path, {"source_times_path": "pts.npz"}, view(timeline_key="main"), [2.0])
    with pytest.raises(CatalogSchemaError, match="does not cover"):
        mapping.resolve(tmp_path, {}, view(), [20.001])


def test_no_shared_frame_index_for_auxiliary(tmp_path):
    rows = pa.table({"timestamp": [0.0, 1.0], "source_frame_id": [20, 21]})
    v = view()
    mapping = MediaTimeline()
    assert mapping.resolve(tmp_path, {}, v, [0, 1], rows)[1].tolist() == [20, 21]
    v["role"] = "left_wrist"
    assert mapping.resolve(tmp_path, {}, v, [0, 1], rows)[1] is None
