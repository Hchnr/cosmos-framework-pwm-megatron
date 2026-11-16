# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
import pickle
import socket
import subprocess
import sys
import weakref
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from cosmos_framework.data.generator import media_decode
from cosmos_framework.data.generator.media_decode import (
    MediaDecodeError,
    MediaDecodeTimeout,
    SupervisedVideoDecoder,
)

GOLDEN_VIDEO = Path(
    "/share/project/liyue/data/Cosmos3-DROID-golden-5c11a20/success/videos/"
    "observation.image.wrist_image_left/chunk-000/file-000.mp4"
)


def _write_index_video(path: Path, *, frames: int = 300, fps: int = 60) -> None:
    with av.open(path, "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((32, 64, 3), index % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_cached_decoder_limits_threads_and_preserves_frames(tmp_path: Path) -> None:
    path = tmp_path / "thread-count.mp4"
    _write_index_video(path, frames=12)
    with av.open(str(path), "r") as reference:
        expected = [frame.to_ndarray(format="rgb24") for frame in reference.decode(video=0)]
    cache = media_decode._ContainerLRU(2)
    try:
        container = cache.get(str(path))
        assert all(stream.codec_context.thread_count == 1 for stream in container.streams.video)
        actual = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)
        assert cache.get(str(path)) is container
    finally:
        cache.close()


@pytest.mark.parametrize("threads", [2, 4])
@pytest.mark.parametrize("reuse", [False, True])
def test_frame_threaded_helper_preserves_long_gop_seek_pixels(tmp_path: Path, threads: int, reuse: bool) -> None:
    path = tmp_path / "long-gop.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"g": "90", "keyint_min": "90", "sc_threshold": "0", "bf": "3"}
        rng = np.random.default_rng(42)
        for _ in range(100):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reference = SupervisedVideoDecoder(reuse_seek_groups=reuse)
    threaded = SupervisedVideoDecoder(reuse_seek_groups=reuse, video_decode_threads=threads)
    try:
        # Reuse handles across forward/backward seeks, a GOP boundary and EOF.
        for times in ([3.05, 3.05, 3.17, 4.1], [0.0, 0.1], [5.95, 6.0, 6.1, 6.59]):
            expected, expected_meta = reference.decode(str(path), timestamps_sec=times)
            actual, actual_meta = threaded.decode(str(path), timestamps_sec=times)
            assert actual_meta == expected_meta
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(got, want)
    finally:
        reference.close()
        threaded.close()


@pytest.mark.parametrize("threads", [0, 5, True, 1.5])
def test_decode_thread_budget_rejects_invalid_values(threads) -> None:
    with pytest.raises(ValueError, match="video_decode_threads"):
        SupervisedVideoDecoder(video_decode_threads=threads)


@pytest.mark.parametrize("output_size", [None, [16, 32]])
def test_helper_timing_preserves_pixels_and_sampling_metadata(tmp_path, monkeypatch, output_size):
    path = tmp_path / "timed.mp4"
    _write_index_video(path, frames=20)
    options = {"timestamps_sec": [0.0, 0.11, 0.11, 0.29]}
    if output_size is not None:
        options["camera_output_size"] = output_size
    monkeypatch.delenv("COSMOS_DATALOADER_TIMING", raising=False)
    reference = SupervisedVideoDecoder(reuse_seek_groups=True)
    try:
        expected, expected_metadata = reference.decode(str(path), **options)
    finally:
        reference.close()
    assert "timing_seconds" not in expected_metadata

    monkeypatch.setenv("COSMOS_DATALOADER_TIMING", "1")
    decoder = SupervisedVideoDecoder(reuse_seek_groups=True)
    try:
        # Both a fresh helper and a reused helper retain the sampling contract.
        for _ in range(2):
            actual, metadata = decoder.decode(str(path), **options)
            timing = metadata.pop("timing_seconds")
            assert all(value >= 0 for value in timing.values())
            assert timing["request"] >= timing["decode_with_io"] + timing["resize"]
            assert metadata == expected_metadata
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(got, want)
    finally:
        decoder.close()


@pytest.mark.parametrize("timing", [False, True])
def test_idle_helper_releases_previous_camera_window(monkeypatch, timing):
    if timing:
        monkeypatch.setenv("COSMOS_DATALOADER_TIMING", "1")
    else:
        monkeypatch.delenv("COSMOS_DATALOADER_TIMING", raising=False)
    observed = []

    def decode(*args, **kwargs):
        return (np.zeros((32, 64, 3), dtype=np.uint8),), {}

    def send(sock, response):
        observed.append(weakref.ref(response["value"][0][0]))

    def receive(sock):
        if not observed:
            return {"path": "unused.mp4", "kwargs": {}}
        assert observed[0]() is None, "idle helper retained the last camera window"
        raise EOFError

    monkeypatch.setattr(media_decode, "decode_video_at_timestamps", decode)
    monkeypatch.setattr(media_decode, "_send", send)
    monkeypatch.setattr(media_decode, "_recv", receive)
    receiver, peer = socket.socketpair()
    try:
        media_decode._helper(receiver.detach(), 1)
    finally:
        peer.close()
    assert len(observed) == 1


def _duplicate_high_fd_socket(sock: socket.socket) -> socket.socket:
    fcntl = pytest.importorskip("fcntl")
    resource = pytest.importorskip("resource")
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit <= 1024:
        pytest.skip("a socket FD >= 1024 requires a higher RLIMIT_NOFILE")
    # Duplicate directly into the range select() cannot represent, without
    # opening a thousand files or changing the process resource limit.
    fd = fcntl.fcntl(sock.fileno(), fcntl.F_DUPFD_CLOEXEC, 1024)
    try:
        return socket.socket(fileno=fd)
    except BaseException:
        os.close(fd)
        raise


def test_helper_camera_resize_pixels_identity_and_limits(tmp_path):
    import torch

    from cosmos_framework.data.generator.camera_resize import copy_resized_camera

    path = tmp_path / "resize.mp4"
    _write_index_video(path, frames=16, fps=8)
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        times = [0.0, 0.5, 1.0]
        raw, metadata = decoder.decode(str(path), timestamps_sec=times)
        for height, width in [(32, 64), (17, 23), (70, 39)]:
            actual, resized_meta = decoder.decode(str(path), timestamps_sec=times, camera_output_size=[height, width])
            expected = torch.empty((3, len(raw), height, width), dtype=torch.uint8)
            copy_resized_camera(raw, expected)
            np.testing.assert_array_equal(np.stack(actual), expected.permute(1, 2, 3, 0).numpy())
            assert resized_meta == {**metadata, "camera_output_size": [height, width]}
        for bad in ([0, 1], [True, 8], [8], [1.2, 8], [100000, 100000]):
            with pytest.raises(ValueError, match="camera_output_size"):
                decoder.decode(str(path), timestamps_sec=times, camera_output_size=bad)
    finally:
        decoder.close()


@pytest.fixture
def high_fd_socketpair() -> Iterator[tuple[socket.socket, socket.socket]]:
    with ExitStack() as stack:
        receiver, sender = socket.socketpair()
        stack.enter_context(receiver)
        stack.enter_context(sender)
        high_receiver = stack.enter_context(_duplicate_high_fd_socket(receiver))
        receiver.close()
        assert high_receiver.fileno() >= 1024
        yield high_receiver, sender


@pytest.mark.parametrize("close_sender", [False, True])
def test_recv_high_fd_message(high_fd_socketpair, close_sender: bool) -> None:
    receiver, sender = high_fd_socketpair
    message = {"frames": [1, 2, 3], "metadata": {"fps": 5.0}}
    media_decode._send(sender, message)
    if close_sender:
        # A peer may close after sending a complete message. Read its buffered
        # data even when the readiness notification also reports a hangup.
        sender.close()
    assert media_decode._recv(receiver, timeout=1.0) == message


def test_recv_high_fd_timeout(high_fd_socketpair) -> None:
    receiver, _ = high_fd_socketpair
    with pytest.raises(TimeoutError):
        media_decode._recv(receiver, timeout=0.05)


@pytest.mark.parametrize(
    "message_bytes",
    [0, 3, media_decode._HEADER.size, media_decode._HEADER.size + 3],
    ids=["empty", "partial-header", "missing-body", "partial-body"],
)
def test_recv_high_fd_closed_peer(high_fd_socketpair, message_bytes: int) -> None:
    receiver, sender = high_fd_socketpair
    payload = pickle.dumps({"message": "incomplete"})
    wire = media_decode._HEADER.pack(len(payload)) + payload
    sender.sendall(wire[:message_bytes])
    sender.close()
    with pytest.raises(EOFError):
        media_decode._recv(receiver, timeout=1.0)


def test_recv_timeout_covers_header_and_partial_body(monkeypatch: pytest.MonkeyPatch) -> None:
    elapsed = 0.0
    recv_calls = 0
    receiver, sender = socket.socketpair()

    class TimedSocket:
        def fileno(self) -> int:
            return receiver.fileno()

        def recv(self, size: int) -> bytes:
            nonlocal elapsed, recv_calls
            # Simulate time spent receiving the header, then part of the body.
            # Everything is already buffered, so this does not depend on thread
            # scheduling or wall-clock timing to cross the original deadline.
            chunk = receiver.recv(min(size, 2) if recv_calls == 1 else size)
            elapsed += (0.75, 0.5, 0.0)[min(recv_calls, 2)]
            recv_calls += 1
            return chunk

    with receiver, sender:
        media_decode._send(sender, {"message": "long enough for a partial body"})
        monkeypatch.setattr(media_decode, "time", SimpleNamespace(monotonic=lambda: elapsed))
        with pytest.raises(TimeoutError):
            media_decode._recv(TimedSocket(), timeout=1.0)
    assert recv_calls == 2


def test_high_fd_diagnostic_skips_low_fd(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def unexpected_scan(path: str) -> list[str]:
        pytest.fail("low-FD decoder startup should not enumerate file descriptors")

    monkeypatch.setattr(media_decode, "_HIGH_FD_LOGGED_PID", None)
    monkeypatch.setattr(media_decode.os, "listdir", unexpected_scan)
    with caplog.at_level("WARNING", logger=media_decode.__name__):
        media_decode._log_high_fd(SimpleNamespace(fileno=lambda: 5), helper_pid=456)
    assert not caplog.records


def test_high_fd_diagnostic_logs_once_per_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    worker_pid = 123
    scans = []

    def scan_fds(path: str) -> list[str]:
        scans.append(path)
        return ["0", "1", "2048"]

    monkeypatch.setattr(media_decode, "_HIGH_FD_LOGGED_PID", None)
    monkeypatch.setattr(media_decode, "os", SimpleNamespace(getpid=lambda: worker_pid, listdir=scan_fds))
    with caplog.at_level("WARNING", logger=media_decode.__name__):
        media_decode._log_high_fd(SimpleNamespace(fileno=lambda: 1024), helper_pid=456)
        media_decode._log_high_fd(SimpleNamespace(fileno=lambda: 2048), helper_pid=457)
        # Forked workers inherit module state, but must still get their own
        # first diagnostic even if the parent already logged high FD pressure.
        worker_pid = 124
        media_decode._log_high_fd(SimpleNamespace(fileno=lambda: 1025), helper_pid=458)
    assert scans == ["/proc/self/fd", "/proc/self/fd"]
    assert len(caplog.messages) == 2
    for message, pid, helper_pid, fd in zip(caplog.messages, [123, 124], [456, 458], [1024, 1025], strict=True):
        assert f"worker_pid={pid} helper_pid={helper_pid} socket_fd={fd}" in message
        assert "observed_open_fd_count=3 observed_max_fd=2048" in message


@pytest.mark.parametrize("proc_fd_scan_fails", [False, True], ids=["proc-available", "proc-unavailable"])
def test_supervised_decoder_with_high_fd_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    proc_fd_scan_fails: bool,
) -> None:
    video = tmp_path / "high-fd.mp4"
    _write_index_video(video, frames=5, fps=5)
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    original_socketpair = socket.socketpair

    def high_fd_pair() -> tuple[socket.socket, socket.socket]:
        parent, child = original_socketpair()
        try:
            high_parent = _duplicate_high_fd_socket(parent)
        except BaseException:
            child.close()
            raise
        finally:
            parent.close()
        return high_parent, child

    def inaccessible_proc(path: str) -> list[str]:
        raise PermissionError("proc FD enumeration is unavailable")

    monkeypatch.setattr(media_decode.socket, "socketpair", high_fd_pair)
    monkeypatch.setattr(media_decode, "_HIGH_FD_LOGGED_PID", None)
    if proc_fd_scan_fails:
        monkeypatch.setattr(media_decode.os, "listdir", inaccessible_proc)
    try:
        with caplog.at_level("WARNING", logger=media_decode.__name__):
            frames, metadata = decoder.decode(
                str(video),
                trim_head_frames=0,
                valid_frames=5,
                target_fps=5.0,
            )
        assert decoder._socket is not None and decoder._socket.fileno() >= 1024
        process = decoder._process
        assert process is not None
    finally:
        decoder.close()
    assert len(frames) == 5
    assert all(frame.shape == (32, 64, 3) for frame in frames)
    assert metadata["timestamps_sec"] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert process.poll() is not None
    assert len(caplog.messages) == 1
    if proc_fd_scan_fails:
        assert "observed_open_fd_count=None observed_max_fd=None" in caplog.messages[0]


def test_trim_then_resample_60fps_source_to_24fps(tmp_path: Path) -> None:
    video = tmp_path / "source-60fps.mp4"
    _write_index_video(video)
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        frames, metadata = decoder.decode(
            str(video),
            trim_head_frames=10,
            valid_frames=93,
            target_fps=24.0,
        )
    finally:
        decoder.close()
    assert len(frames) == 93
    assert metadata["trim_head_frames"] == 10
    assert metadata["target_fps"] == 24.0
    assert metadata["timestamps_sec"][0] == 0.0
    timestamp_gaps = np.diff(metadata["timestamps_sec"])
    assert set(np.round(timestamp_gaps * 60).astype(int)).issubset({2, 3})
    assert max(metadata["timestamp_error_seconds"]) <= (1 / 120) + 1e-6


def test_timestamp_decode_returns_requested_and_actual_timestamps() -> None:
    if not GOLDEN_VIDEO.is_file():
        pytest.skip("official DROID golden video is not installed")
    decoder = SupervisedVideoDecoder(timeout_seconds=30)
    try:
        frames, metadata = decoder.decode(str(GOLDEN_VIDEO), timestamps_sec=[0.0, 1.0])
    finally:
        decoder.close()
    assert len(frames) == 2
    assert frames[0].shape == (360, 640, 3)
    assert metadata["requested_timestamps_sec"] == [0.0, 1.0]
    assert metadata["timestamps_sec"] == [0.0, 1.0]
    assert len(metadata["decoded_frame_pts"]) == 2


def test_decode_safety_limit_is_a_structural_error() -> None:
    if not GOLDEN_VIDEO.is_file():
        pytest.skip("official DROID golden video is not installed")
    decoder = SupervisedVideoDecoder(timeout_seconds=30, max_sampled_frames=2)
    try:
        with pytest.raises(ValueError, match="max_sampled_frames=2"):
            decoder.decode(str(GOLDEN_VIDEO), timestamps_sec=[0.0, 1.0, 2.0])
    finally:
        decoder.close()


def test_decoded_window_limit_is_a_droppable_media_error(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _write_index_video(video, frames=5, fps=5)
    decoder = SupervisedVideoDecoder(timeout_seconds=30, max_window_bytes=1)
    try:
        with pytest.raises(MediaDecodeError, match="max_window_bytes=1"):
            decoder.decode(
                str(video),
                trim_head_frames=0,
                valid_frames=5,
                target_fps=5.0,
            )
    finally:
        decoder.close()


def test_source_frame_pixel_limit_is_a_droppable_media_error(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _write_index_video(video, frames=2, fps=2)
    decoder = SupervisedVideoDecoder(timeout_seconds=30, max_frame_pixels=1)
    try:
        with pytest.raises(MediaDecodeError, match="max_frame_pixels=1"):
            decoder.decode(
                str(video),
                trim_head_frames=0,
                valid_frames=2,
                target_fps=2.0,
                output_canvas_width=32,
                output_canvas_height=16,
            )
    finally:
        decoder.close()


def test_clip_decode_resizes_before_enforcing_window_bytes(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _write_index_video(video, frames=5, fps=5)
    decoder = SupervisedVideoDecoder(timeout_seconds=30, max_window_bytes=10_000)
    try:
        frames, metadata = decoder.decode(
            str(video),
            trim_head_frames=0,
            valid_frames=5,
            target_fps=5.0,
            output_canvas_width=32,
            output_canvas_height=16,
        )
    finally:
        decoder.close()

    assert frames[0].shape == (16, 32, 3)
    assert metadata["width"] == 64
    assert metadata["height"] == 32
    assert metadata["decoded_width"] == 32
    assert metadata["decoded_height"] == 16


def test_timeout_kills_helper_and_allows_lazy_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder = SupervisedVideoDecoder(timeout_seconds=0.05)
    processes: list[subprocess.Popen] = []

    def start_hung_helper() -> None:
        parent, child = socket.socketpair()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(child.fileno(),),
            close_fds=True,
            start_new_session=True,
        )
        child.close()
        decoder._socket = parent
        decoder._process = process
        processes.append(process)

    monkeypatch.setattr(decoder, "_start", start_hung_helper)
    with pytest.raises(MediaDecodeTimeout):
        decoder.decode("unused.mp4", timestamps_sec=[0.0])
    assert processes[0].poll() is not None
    assert decoder._process is None
    with pytest.raises(MediaDecodeTimeout):
        decoder.decode("unused.mp4", timestamps_sec=[0.0])
    assert len(processes) == 2
    assert all(process.poll() is not None for process in processes)
    # ``close`` is idempotent even after timeout teardown.
    decoder.close()
    assert not any(process.poll() is None for process in processes)


@pytest.mark.parametrize("limit", [1, 2, 2048])
def test_timestamp_rgb_conversion_only_materializes_selected_frames(tmp_path, monkeypatch, limit):
    """Candidate metadata and seek budgets must survive delayed RGB conversion."""
    from types import SimpleNamespace

    from cosmos_framework.data.generator.media_decode import decode_video_at_timestamps

    converted = []
    seeks = []

    class Frame:
        def __init__(self, pts):
            self.pts = pts

        def to_ndarray(self, *, format):
            assert format == "rgb24"
            converted.append(self.pts)
            return np.full((2, 2, 3), self.pts, dtype=np.uint8)

    stream = SimpleNamespace(time_base=0.25, start_time=8, width=2, height=2, average_rate=4, frames=4, duration=4)

    class Container:
        streams = SimpleNamespace(video=[stream])

        def seek(self, offset, **kwargs):
            seeks.append(offset)

        def decode(self, stream):
            return iter(Frame(pts) for pts in [8, 9, 10, 11])

        def close(self):
            pass

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    path = tmp_path / "fake.mp4"
    path.touch()
    requested = [0, 0.375, 0.375, 2.0]
    if limit < 4:
        with pytest.raises(MediaDecodeError, match="budget"):
            decode_video_at_timestamps(str(path), timestamps_sec=requested, max_decode_frames_per_seek=limit)
        return
    images, metadata = decode_video_at_timestamps(str(path), timestamps_sec=requested, max_decode_frames_per_seek=limit)
    expected = [8, 9, 9, 11]
    assert converted == expected
    assert metadata["decoded_frame_pts"] == expected
    assert seeks == [8, 9, 9, 16]
    assert metadata["decoded_frames_per_seek"] == [min(limit, n) for n in [2, 3, 3, 4]]
    for image, pts in zip(images, expected, strict=True):
        np.testing.assert_array_equal(image, np.full((2, 2, 3), pts, dtype=np.uint8))
    assert not np.shares_memory(images[1], images[2])


@pytest.mark.parametrize("limit", [1, 2, 17, 2048])
@pytest.mark.parametrize("variable_pts", [False, True])
def test_packet_group_decode_matches_independent_seeks(tmp_path, limit, variable_pts):
    """Real H264 B-frames, GOP crossings, nonzero/VFR PTS, ties and EOF."""
    from fractions import Fraction

    from cosmos_framework.data.generator.media_decode import decode_video_at_timestamps

    path = tmp_path / "bframes.mp4"
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "30", "bf": "2"}
        for index in range(100):
            pixels = np.full((32, 64, 3), index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.time_base = Fraction(1, 30)
            frame.pts = 60 + index + (index // 3 if variable_pts else 0)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    query = [0, 1 / 60, 0.2, 0.2, 0.9, 1, 1.01, 1.5, 1.99, 2, 2.1, 3, 9]
    kwargs = {"timestamps_sec": query, "max_decode_frames_per_seek": limit}
    if limit < 30:
        # Both paths must report an exhausted budget, rather than agree on an
        # incorrect old frame far before the target.
        for grouped in (False, True):
            with pytest.raises(MediaDecodeError, match="budget"):
                decode_video_at_timestamps(str(path), reuse_seek_groups=grouped, **kwargs)
        return
    expected, old_meta = decode_video_at_timestamps(str(path), **kwargs)
    actual, new_meta = decode_video_at_timestamps(str(path), reuse_seek_groups=True, **kwargs)
    assert new_meta == old_meta
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("reuse_seek_groups", [False, True])
@pytest.mark.parametrize("limit", [1, 17, 2048])
def test_timestamp_seek_budget_must_reach_target_neighborhood(tmp_path, reuse_seek_groups, limit):
    from cosmos_framework.data.generator.media_decode import decode_video_at_timestamps

    path = tmp_path / "long_gop.mp4"
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "100", "keyint_min": "100", "sc_threshold": "0", "bf": "0", "crf": "0"}
        for index in range(100):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    kwargs = dict(timestamps_sec=[8.0, 8.1], reuse_seek_groups=reuse_seek_groups, max_decode_frames_per_seek=limit)
    if limit < 82:
        with pytest.raises(MediaDecodeError, match="budget"):
            decode_video_at_timestamps(str(path), **kwargs)
    else:
        frames, metadata = decode_video_at_timestamps(str(path), **kwargs)
        assert metadata["timestamps_sec"] == pytest.approx([8.0, 8.1])
        for image, expected in zip(frames, [80, 81], strict=True):
            assert np.max(np.abs(image.astype(int) - expected)) <= 1


@pytest.mark.parametrize("reuse_seek_groups", [False, True])
@pytest.mark.parametrize("targets", [[0.0], [1.5, 2.0]])
def test_timestamp_budget_allows_exact_match_and_real_eof(tmp_path, reuse_seek_groups, targets):
    from cosmos_framework.data.generator.media_decode import decode_video_at_timestamps

    path = tmp_path / "short.mp4"
    _write_index_video(path, frames=4, fps=2)
    _, metadata = decode_video_at_timestamps(
        str(path),
        timestamps_sec=targets,
        reuse_seek_groups=reuse_seek_groups,
        max_decode_frames_per_seek=1 if targets == [0.0] else 4,
    )
    assert metadata["timestamps_sec"] == pytest.approx([min(t, 1.5) for t in targets])


def test_packet_groups_decode_shared_prefix_once(tmp_path, monkeypatch):
    from cosmos_framework.data.generator.media_decode import decode_video_at_timestamps

    path = tmp_path / "source.mp4"
    _write_index_video(path, frames=120, fps=30)
    original_open = av.open
    counts = []

    class CountingContainer:
        def __init__(self, *args, **kwargs):
            self.inner = original_open(*args, **kwargs)
            counts.append(0)
            self.index = len(counts) - 1

        def __getattr__(self, key):
            return getattr(self.inner, key)

        def decode(self, stream):
            for frame in self.inner.decode(stream):
                counts[self.index] += 1
                yield frame

    monkeypatch.setattr(av, "open", CountingContainer)
    query = [index / 30 for index in range(60)]
    expected, old_meta = decode_video_at_timestamps(str(path), timestamps_sec=query)
    actual, new_meta = decode_video_at_timestamps(str(path), timestamps_sec=query, reuse_seek_groups=True)
    assert new_meta == old_meta
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert counts[1] < counts[0] / 2


def test_packet_groups_keep_window_memory_limit(tmp_path):
    from cosmos_framework.data.generator.media_decode import decode_video_at_timestamps

    path = tmp_path / "source.mp4"
    _write_index_video(path, frames=20, fps=30)
    with pytest.raises(MediaDecodeError, match="max_window_bytes=1"):
        decode_video_at_timestamps(str(path), timestamps_sec=[0.0, 0.01], reuse_seek_groups=True, max_window_bytes=1)


@pytest.mark.parametrize("enabled", [False, True])
def test_seek_group_config_reaches_supervised_helper(enabled):
    from cosmos_framework.configs.toml_config.sft_config import DataloaderMediaConfig
    from cosmos_framework.data.generator.catalog_processors import build_video_decoder_from_config

    media = DataloaderMediaConfig(reuse_seek_groups=enabled)
    decoder = build_video_decoder_from_config(media.model_dump())
    assert decoder.decode_options["reuse_seek_groups"] is enabled
    assert DataloaderMediaConfig().reuse_seek_groups is False
