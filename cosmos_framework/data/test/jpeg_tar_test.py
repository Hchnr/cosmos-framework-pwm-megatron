"""Generic pretraining JPEG TAR: random access, publication and reader routing."""

import hashlib
import io
import json
import os
import pickle
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from cosmos_framework.data.generator.jpeg_tar import STORAGE, IndexedJpegTarReader, image_storage_fingerprint
from cosmos_framework.data.generator.media_decode import MediaDecodeError, SupervisedVideoDecoder


def jpeg_source(root, *, fps=25, count=7, cameras=3):
    (root / "meta").mkdir(exist_ok=True, parents=True)
    path = root / "frames.tar"
    with tarfile.open(path, "w") as archive:
        for frame in range(count):
            for camera in range(cameras):
                image = Image.fromarray(np.full((8, 12, 3), [30 + 60 * camera, 30 + 20 * frame, 10], np.uint8))
                stream = io.BytesIO()
                image.save(stream, format="JPEG", quality=95)
                payload = stream.getvalue()
                member = tarfile.TarInfo(f"camera{camera}/frame_{frame:06d}.jpg")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    offsets = np.zeros((count + 2, cameras, 2), np.uint64)
    with tarfile.open(path) as archive:
        for frame in range(count):
            for camera in range(cameras):
                member = archive.getmember(f"camera{camera}/frame_{frame:06d}.jpg")
                offsets[2 + frame, camera] = [member.offset_data, member.size]
    index = root / "meta/jpeg_offsets.npy"
    np.save(index, offsets)
    sha = hashlib.sha256(index.read_bytes()).hexdigest()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["image_storage"] = "mixed_mp4_indexed_jpeg_tar_v1"
    info_path.write_text(json.dumps(info))
    contract = dict(
        schema_version=STORAGE,
        container_kind="mixed_mp4_indexed_jpeg_tar_v1",
        index_path="meta/jpeg_offsets.npy",
        index_dtype="uint64",
        index_shape=list(offsets.shape),
        index_sha256=sha,
    )
    (root / "meta/image_storage.json").write_text(json.dumps(contract))
    (root / "training_validation.json").write_text(json.dumps({"status": "passed", "image_index_sha256": sha}))
    view = dict(
        storage=STORAGE,
        index_path="meta/jpeg_offsets.npy",
        jpeg_frame_start=2,
        jpeg_camera_index=cameras - 1,
        path="frames.tar",
        camera_name="last",
        fps=fps,
        from_timestamp=0.0,
        to_timestamp=count / fps,
        frame_count=count,
        width=12,
        height=8,
    )
    return root, view


@pytest.fixture
def source(tmp_path):
    return jpeg_source(tmp_path)


@pytest.mark.parametrize("fps", [20, 25])
def test_native_frames_nearest_resampling_and_valid_final_cell(tmp_path, fps):
    root, view = jpeg_source(tmp_path, fps=fps)
    reader = IndexedJpegTarReader(root)
    try:
        frames, result = reader.decode(view, timestamps_sec=[0, 0, 1.1 / fps, 1.6 / fps, 6.9 / fps])
        assert frames[0] is frames[1]
        assert result["frame_indexes"] == [0, 0, 1, 2, 6]
        assert result["decoded_frames"] == 4
        assert result["end_clamped_frames"] == 1
        np.testing.assert_allclose(result["timestamps_sec"], np.array([0, 0, 1, 2, 6]) / fps)
        assert frames[-1][0, 0, 1] > frames[0][0, 0, 1]
        assert frames[0][0, 0, 0] > 140
        # A 15 Hz request is valid even though source JPEG images are 20/25 Hz.
        _, resampled = reader.decode(view, timestamps_sec=[0, 1 / 15, 2 / 15])
        assert resampled["frame_indexes"] == np.floor(np.array([0, 1 / 15, 2 / 15]) * fps + 0.5).astype(int).tolist()
    finally:
        reader.close()


@pytest.mark.parametrize("times", [[0.28], [-0.01], [], [float("nan")], [float("inf")]])
def test_reject_outside_media_time_range(source, times):
    root, view = source
    reader = IndexedJpegTarReader(root)
    try:
        with pytest.raises(ValueError):
            reader.decode(view, timestamps_sec=times)
    finally:
        reader.close()


def test_clip_range_and_nonzero_origin(source):
    root, view = source
    reader = IndexedJpegTarReader(root)
    try:
        view = {**view, "from_timestamp": 3.0, "to_timestamp": 3.2}
        _, result = reader.decode(view, timestamps_sec=[3, 3.05])
        assert result["frame_indexes"] == [0, 1]
        np.testing.assert_allclose(result["timestamps_sec"], [3, 3.04])
        with pytest.raises(ValueError, match="half-open"):
            reader.decode(view, timestamps_sec=[3.2])
    finally:
        reader.close()


@pytest.mark.parametrize(
    "override", [dict(jpeg_frame_start=100), dict(jpeg_camera_index=3), dict(frame_count=0), dict(width=12.5)]
)
def test_index_and_integer_bounds(source, override):
    root, view = source
    reader = IndexedJpegTarReader(root)
    try:
        with pytest.raises(ValueError):
            reader.decode({**view, **override}, timestamps_sec=[0])
    finally:
        reader.close()


def test_cache_pickle_pid_and_close(source):
    root, view = source
    os.link(root / "frames.tar", root / "other.tar")
    reader = IndexedJpegTarReader(root, cache_size=1)
    try:
        reader.decode(view, timestamps_sec=[0])
        old_fd = next(iter(reader._fds.values()))
        reader.decode({**view, "path": "other.tar"}, timestamps_sec=[0])
        with pytest.raises(OSError):
            os.fstat(old_fd)
        assert len(reader._fds) == 1
        copied = pickle.loads(pickle.dumps(reader))
        assert not copied._fds and copied._index is None
        reader._pid = -1
        reader.decode(view, timestamps_sec=[0])
        copied.decode(view, timestamps_sec=[0])
        copied.close()
    finally:
        reader.close()
    assert not reader._fds and reader._index is None


def test_helper_real_decode_restart_root_switch_and_limits(source, tmp_path):
    root, view = source
    root2, view2 = jpeg_source(tmp_path / "other_root", fps=20, cameras=1)
    direct = IndexedJpegTarReader(root)
    helper = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        expected, _ = direct.decode(view, timestamps_sec=[0, 0.06])
        actual, metadata = helper.decode_jpeg(root, view, timestamps_sec=[0, 0.06])
        np.testing.assert_array_equal(actual, expected)
        assert metadata["video_backend"] == STORAGE
        helper.decode_jpeg(root2, view2, timestamps_sec=[0, 0.05])
        helper.close()
        again, _ = helper.decode_jpeg(root, view, timestamps_sec=[0])
        np.testing.assert_array_equal(again[0], expected[0])
        with pytest.raises(ValueError, match="pixel/window"):
            helper.decode(
                str(root / view["path"]), jpeg_root=str(root), jpeg_view=view, timestamps_sec=[0], max_frame_pixels=1
            )
    finally:
        helper.close()
        direct.close()
    assert helper._process is None and helper._socket is None


def test_corrupt_jpeg_is_media_error(source):
    root, view = source
    index = np.load(root / "meta/jpeg_offsets.npy")
    offset, size = map(int, index[2, 2])
    with (root / "frames.tar").open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"x" * size)
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        with pytest.raises(MediaDecodeError, match="JPEG"):
            decoder.decode_jpeg(root, view, timestamps_sec=[0])
    finally:
        decoder.close()


def test_mixed_storage_fingerprint_validation_and_header(source):
    root, _ = source
    result = image_storage_fingerprint(root)
    assert result["contract"]["container_kind"] == "mixed_mp4_indexed_jpeg_tar_v1"
    assert result["index_stat"][0] == (root / "meta/jpeg_offsets.npy").stat().st_size
    validation_path = root / "training_validation.json"
    validation = json.loads(validation_path.read_text())
    validation_path.write_text(json.dumps({**validation, "image_index_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="validation SHA"):
        image_storage_fingerprint(root)
    validation_path.write_text(json.dumps(validation))
    contract_path = root / "meta/image_storage.json"
    contract = json.loads(contract_path.read_text())
    contract_path.write_text(json.dumps({**contract, "index_shape": [99, 3, 2]}))
    with pytest.raises(ValueError, match="header"):
        image_storage_fingerprint(root)


class TokenizerStub:
    """Only text tokenization is stubbed; actual metadata/action/video are read."""

    def apply_chat_template(self, conversations, **kwargs):
        return list(conversations[-1]["content"].encode()) or [1]

    def encode(self, text, **kwargs):
        return [999999]


@pytest.mark.parametrize("use_state", [False, True])
def test_unified_pretrain_get_item_jpeg_and_mp4_share_default_path(tmp_path, monkeypatch, use_state):
    from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset
    from cosmos_framework.data.test.action128_test import source as numeric_source
    from cosmos_framework.data.test.media_decode_test import _write_index_video
    from cosmos_framework.data.test.pretrain_state_test import state_source

    numeric = numeric_source.__wrapped__(tmp_path)
    root, _, _ = numeric
    if use_state:
        state_source.__wrapped__(numeric)
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["state_storage"] = {"value": "observation.state", "loader_mask": "observation.state_mask"}
        info_path.write_text(json.dumps(info))
    _, view = jpeg_source(root, fps=2, count=7)
    _write_index_video(root / "video.mp4", fps=2, frames=7)
    meta_path = root / "meta/episodes/chunk-000/episodes.parquet"
    rows = pq.read_table(meta_path).to_pylist()
    for row in rows:
        row["split"] = None
    rows[0]["video_views"][0].update(view)
    rows[0]["video_views"].append(
        dict(
            role="left_wrist",
            source_key="camera2",
            fps=2,
            width=64,
            height=32,
            from_timestamp=0.0,
            to_timestamp=3.5,
            path="video.mp4",
            storage="mp4",
        )
    )
    rows[0]["view_mapping"]["left_wrist"] = "camera2"
    pq.write_table(pa.Table.from_pylist(rows), meta_path)
    # Small canvas only: production layout is independently covered by action128
    # tests; do not replace the real helper/metadata/action/caption code paths.
    monkeypatch.setattr(
        "cosmos_framework.data.generator.action.unified_action128.view_layout",
        lambda ep: (ep["video_views"], [(0, 0, 32, 16), (0, 16, 32, 32)], 32, 32),
    )
    ds = UnifiedAction128Dataset(root, tokenizer_config=TokenizerStub(), use_state=use_state)
    try:
        assert ds.duration == 2 and ds.stride == 2 and ds.use_state is use_state
        assert not hasattr(ds, "task_sampling")
        assert len(ds) == 2
        sample = ds.get_item(0, epoch=0)
        assert sample["video"].shape == (3, 5, 32, 32)
        assert sample["action"].shape == (5 if use_state else 4, 128)
        assert sample["action_loss_mask"].any()
        assert sample["vision_loss_mask"].any()
        assert sample["camera_valid"][:4, :2].all()
        if use_state:
            assert not sample["action_loss_mask"][0].any()
            assert sample["sequence_plan"].robot_state_frame_indexes == [0]
        assert json.loads(sample["ai_caption"])["command"].startswith("The robot carries")
        assert sample["decode_metadata"]["views"][0]["video_backend"] == STORAGE
        assert "pyav" in sample["decode_metadata"]["views"][1]["video_backend"]
        assert "image_storage" in ds._fingerprint_inputs
        ds.get_item(1, epoch=0)  # Tail repeats the last eligible real media time.
    finally:
        ds.decoder.close()
        ds.zarr_tar_decoder.close()
