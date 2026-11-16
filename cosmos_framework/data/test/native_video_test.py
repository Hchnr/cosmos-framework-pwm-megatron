import io
import json
import struct
import tarfile
from fractions import Fraction

import google_crc32c
import numpy as np
import pytest
import torch
import zstandard
from PIL import Image

from cosmos_framework.data.generator.action.native_video import NativeVideoReader
from cosmos_framework.data.generator.media_decode import MediaDecodeError, SupervisedVideoDecoder


def assert_helper_resize(decoder, path, **options):
    from cosmos_framework.data.generator.camera_resize import copy_resized_camera

    raw, before = decoder.decode(str(path), **options)
    resized, after = decoder.decode(str(path), camera_output_size=[11, 17], **options)
    expected = torch.empty((3, len(raw), 11, 17), dtype=torch.uint8)
    copy_resized_camera(raw, expected)
    np.testing.assert_array_equal(np.stack(resized), expected.permute(1, 2, 3, 0).numpy())
    assert after == {**before, "camera_output_size": [11, 17]}
    assert all(f.flags.c_contiguous for f in resized)


@pytest.fixture
def jpeg_zarr(tmp_path):
    root = tmp_path / "video.zarr"
    key = root / "images.front_1"
    (key / "c").mkdir(parents=True)
    (root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
    payloads = []
    for color in (20, 130, 230):
        stream = io.BytesIO()
        Image.fromarray(np.full((16, 24, 3), color, dtype=np.uint8)).save(stream, format="JPEG")
        jpeg = stream.getvalue()
        payloads.append(zstandard.ZstdCompressor().compress(struct.pack("<II", 1, len(jpeg)) + jpeg))
    offsets, offset = [], 0
    for data in payloads:
        offsets.append((offset, len(data)))
        offset += len(data)
    index = np.array(offsets, dtype="<u8").tobytes()
    (key / "c/0").write_bytes(b"".join(payloads) + index + struct.pack("<I", google_crc32c.value(index)))
    (key / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "shape": [3],
                "chunk_grid": {"configuration": {"chunk_shape": [3]}},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": [1],
                            "codecs": [{"name": "vlen-bytes"}, {"name": "zstd"}],
                            "index_codecs": [{"name": "bytes"}, {"name": "crc32c"}],
                            "index_location": "end",
                        },
                    }
                ],
            }
        )
    )
    return root


@pytest.mark.parametrize("archive", [False, True])
def test_zarr_directory_and_archive_in_helper(jpeg_zarr, tmp_path, archive):
    root = jpeg_zarr
    if archive:
        root = tmp_path / "video.zarr.tar"
        with tarfile.open(root, "w") as tar:
            tar.add(jpeg_zarr, arcname="record.zarr")
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        frames, meta = decoder.decode(
            str(root),
            native_view={"container_format": "zarr", "stream_key": "images.front_1"},
            timestamps_sec=[0.0, 0.5, 1.0],
            source_frame_indexes=[0, 1, 2],
        )
        assert [round(float(f.mean())) for f in frames] == [20, 130, 230]
        assert meta["source_frame_indexes"] == [0, 1, 2]
        assert_helper_resize(
            decoder,
            root,
            native_view={"container_format": "zarr", "stream_key": "images.front_1"},
            timestamps_sec=[0.0, 0.5, 1.0],
            source_frame_indexes=[0, 1, 2],
        )
    finally:
        decoder.close()


def test_zarr_bad_index_and_bounds(jpeg_zarr):
    reader = NativeVideoReader()
    view = {"stream_key": "images.front_1"}
    with pytest.raises(MediaDecodeError, match="outside"):
        reader.zarr_frame(str(jpeg_zarr), view, 3, 10000)
    path = jpeg_zarr / "images.front_1/c/0"
    path.write_bytes(path.read_bytes()[:-1] + b"\xff")
    with pytest.raises(MediaDecodeError, match="checksum"):
        reader.zarr_frame(str(jpeg_zarr), view, 0, 10000)
    reader.close()


@pytest.fixture
def mcap_video(tmp_path):
    import av
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from mcap_protobuf.writer import Writer

    schema = descriptor_pb2.FileDescriptorProto(name="test_video.proto", package="fixture", syntax="proto3")
    msg = schema.message_type.add(name="VideoPacket")
    msg.field.add(name="data", number=1, type=12, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    message = message_factory.GetMessageClass(pool.FindMessageTypeByName("fixture.VideoPacket"))
    path = tmp_path / "video.mcap"
    origin = 1_700_000_000_000_000_123
    with open(path, "wb") as stream:
        writer = Writer(stream)
        for topic, value in (("left", 40), ("right", 210)):
            codec = av.CodecContext.create("libx264", "w")
            codec.width, codec.height, codec.pix_fmt = 32, 24, "yuv420p"
            codec.time_base = Fraction(1, 30)
            codec.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "18", "keyint": "30"}
            for i in range(61):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), value, dtype=np.uint8), format="rgb24")
                frame.pts = i
                for packet in codec.encode(frame):
                    writer.write_message(topic, message(data=bytes(packet)), log_time=origin + round(i / 30 * 1e9))
        writer.finish()
    return path, origin


def test_mcap_topic_and_integer_clock_in_helper(mcap_video):
    path, origin = mcap_video
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        for topic, expected in (("left", 40), ("right", 210), ("left", 40)):
            frames, meta = decoder.decode(
                str(path),
                native_view={
                    "container_format": "mcap",
                    "mcap_topic": topic,
                    "time_origin_ns": origin,
                    "source_fps": 30,
                },
                timestamps_sec=[0.5, 1.0, 1.5],
            )
            assert all(abs(float(frame.mean()) - expected) < 3 for frame in frames)
            assert max(meta["timestamp_error_seconds"]) < 1e-8
        assert_helper_resize(
            decoder,
            path,
            native_view={"container_format": "mcap", "mcap_topic": "left", "time_origin_ns": origin, "source_fps": 30},
            timestamps_sec=[0.5, 1.0, 1.5],
        )
    finally:
        decoder.close()


@pytest.mark.parametrize("offset", [333e-9, 1 / 60])
def test_mcap_tail_uses_real_last_frame_within_existing_tolerance(mcap_video, offset):
    path, origin = mcap_video
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    view = {"container_format": "mcap", "mcap_topic": "left", "time_origin_ns": origin, "source_fps": 30}
    try:
        reference, _ = decoder.decode(str(path), native_view=view, timestamps_sec=[2.0])
        frames, meta = decoder.decode(str(path), native_view=view, timestamps_sec=[1.9, 2.0 + offset, 2.0 + offset])
        assert meta["pts"][-2:] == [origin + 2_000_000_000] * 2
        assert meta["actual_timestamps_sec"][-2:] == [2.0, 2.0]
        for frame in frames[-2:]:
            np.testing.assert_array_equal(frame, reference[0])
        assert max(meta["timestamp_error_seconds"]) <= 2 / 30
        with pytest.raises(MediaDecodeError, match="outside MCAP topic timeline"):
            decoder.decode(str(path), native_view=view, timestamps_sec=[2.1])
    finally:
        decoder.close()


def test_mcap_tail_does_not_bypass_packet_or_byte_budgets(mcap_video):
    path, origin = mcap_video
    view = {"container_format": "mcap", "mcap_topic": "left", "time_origin_ns": origin, "source_fps": 30}
    # The 61st packet contains the actual tail frame. A 60-packet budget must
    # fail even though the preceding frame is within the timestamp tolerance.
    limited = SupervisedVideoDecoder(timeout_seconds=30, max_decode_frames_per_seek=60)
    byte_limited = SupervisedVideoDecoder(timeout_seconds=30, max_window_bytes=24 * 32 * 3)
    try:
        with pytest.raises(MediaDecodeError, match="bounded GOP pre-roll"):
            limited.decode(str(path), native_view=view, timestamps_sec=[2.0 + 333e-9])
        with pytest.raises(MediaDecodeError, match="max_window_bytes"):
            byte_limited.decode(str(path), native_view=view, timestamps_sec=[2.0 + 333e-9] * 2)
    finally:
        limited.close()
        byte_limited.close()


def test_mcap_topic_bounds_fail_before_decode_and_ignore_overstated_metadata(tmp_path, monkeypatch):
    from mcap.reader import SeekingReader
    from mcap.writer import Writer

    path = tmp_path / "different_tails.mcap"
    origin = 1_700_000_000_000_000_123
    with path.open("wb") as output:
        writer = Writer(output)
        writer.start()
        for topic, duration in [("short", 1), ("long", 2)]:
            channel = writer.register_channel(topic, "", 0)
            for offset in [0, duration]:
                writer.add_message(channel, origin + offset * 10**9, b"unused", origin + offset * 10**9)
        writer.finish()

    def unexpected_decode(*_, **__):
        raise AssertionError("out-of-range requests must not enter H.264 decoding")

    monkeypatch.setattr(SeekingReader, "iter_decoded_messages", unexpected_decode)
    reader = NativeVideoReader()
    limits = {"max_decode_frames_per_seek": 2048, "max_frame_pixels": 10000, "max_window_bytes": 100000}
    try:
        for topic, request in [("short", 1.5), ("long", 2.5), ("short", 1.6)]:
            view = {
                "mcap_topic": topic,
                "time_origin_ns": origin,
                "source_fps": 30,
                "last_log_time_ns": origin + 99 * 10**9,
            }
            with pytest.raises(MediaDecodeError, match="outside MCAP topic timeline"):
                reader.mcap_frames(str(path), view, [request], limits)
    finally:
        reader.close()


def test_mcap_bounds_invalidate_when_source_is_replaced(tmp_path):
    from mcap.writer import Writer

    path = tmp_path / "replaced.mcap"
    origin = 1_700_000_000_000_000_123

    def write(duration):
        with path.open("wb") as output:
            writer = Writer(output)
            writer.start()
            channel = writer.register_channel("camera", "", 0)
            for offset in range(duration + 1):
                writer.add_message(channel, origin + offset * 10**9, b"unused", origin + offset * 10**9)
            writer.finish()

    reader = NativeVideoReader()
    view = {"mcap_topic": "camera", "time_origin_ns": origin, "source_fps": 30}
    limits = {"max_decode_frames_per_seek": 2048, "max_frame_pixels": 10000, "max_window_bytes": 100000}
    try:
        for duration in [1, 2]:
            write(duration)
            with pytest.raises(MediaDecodeError, match=f"available_ns=\\[{origin}, {origin + duration * 10**9}\\]"):
                reader.mcap_frames(str(path), view, [3.0], limits)
    finally:
        reader.close()
