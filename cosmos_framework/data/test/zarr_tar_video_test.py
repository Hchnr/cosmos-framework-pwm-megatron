"""Indexed PsiBot video chunks must preserve Zarr's padded edge semantics."""

import io
import json
import tarfile

import numpy as np
import pytest
import zstandard

from cosmos_framework.data.generator.action.zarr_tar_video import ZarrTarVideoDecoder
from cosmos_framework.data.generator.media_decode import MediaDecodeError


def indexed_video(tmp_path, tail_frames, *, truncate_tail=False):
    chunk_frames, fps = 64, 30
    frames = np.broadcast_to(
        np.arange(chunk_frames + tail_frames, dtype=np.uint8)[:, None, None, None],
        (chunk_frames + tail_frames, 2, 3, 3),
    ).copy()
    path = tmp_path / "video.zarr.tar"
    with tarfile.open(path, "w") as archive:
        for chunk in range(2):
            stored = np.full((chunk_frames, 2, 3, 3), 255, dtype=np.uint8)
            valid = frames[chunk * chunk_frames : (chunk + 1) * chunk_frames]
            stored[: len(valid)] = valid
            if chunk == 1 and truncate_tail:
                stored = stored[:tail_frames]
            data = zstandard.ZstdCompressor().compress(stored.tobytes())
            member = tarfile.TarInfo(f"head_rgb/c/{chunk}/0/0/0")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    with tarfile.open(path) as archive:
        chunks = {str(i): [member.offset_data, member.size] for i, member in enumerate(archive.getmembers())}
    index = dict(
        schema_version="psibot-zarr-tar-head-rgb-index-v1",
        tar_size=path.stat().st_size,
        shape=list(frames.shape),
        chunk_frames=chunk_frames,
        fps=fps,
        chunks=chunks,
    )
    path.with_name(path.name + ".head_rgb_index.json").write_text(json.dumps(index))
    return path, frames, fps


@pytest.mark.parametrize("tail_frames", [1, 24, 52, 64])
def test_padded_final_chunk_returns_only_logical_frames(tmp_path, tail_frames):
    path, expected, fps = indexed_video(tmp_path, tail_frames)
    requested = [0, 63, 64, len(expected) - 1, len(expected) + 100, 0]
    decoder = ZarrTarVideoDecoder()
    try:
        frames, _ = decoder.decode(str(path), timestamps_sec=[frame / fps for frame in requested])
        for actual, frame in zip(frames, requested, strict=True):
            np.testing.assert_array_equal(actual, expected[min(frame, len(expected) - 1)])
    finally:
        decoder.close()


def test_truncated_storage_chunk_is_a_decode_error(tmp_path):
    path, _, fps = indexed_video(tmp_path, 24, truncate_tail=True)
    decoder = ZarrTarVideoDecoder()
    try:
        with pytest.raises(MediaDecodeError, match="Cannot read PsiBot TAR frame 64"):
            decoder.decode(str(path), timestamps_sec=[64 / fps])
    finally:
        decoder.close()
