"""Bounded camera concurrency must preserve ordering, pixels, and failure policy."""

import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset
from cosmos_framework.data.generator.media_decode import MediaDecodeError, SupervisedVideoDecoder
from cosmos_framework.data.test.action128_test import source as source
from cosmos_framework.data.test.pretrain_state_test import state_source as state_source


class CameraDecoder(SupervisedVideoDecoder):
    def __init__(self, state):
        self.state = state
        self.closed = False
        state["decoders"].append(self)

    def independent_decoder(self):
        return CameraDecoder(self.state)

    def decode(self, path, *, timestamps_sec, camera_output_size):
        role = Path(path).stem
        with self.state["lock"]:
            self.state["active"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["active"])
        try:
            if self.state.get("barrier") and role in {"main", "left_wrist"}:
                self.state["barrier"].wait(timeout=10)
            if role == self.state.get("fail"):
                raise MediaDecodeError("broken camera")
            height, width = camera_output_size
            value = {"main": 10, "left_wrist": 20, "right_wrist": 30}[role]
            frames = tuple(np.full((height, width, 3), value, dtype=np.uint8) for _ in timestamps_sec)
            return frames, {"camera_output_size": camera_output_size, "camera": role}
        finally:
            with self.state["lock"]:
                self.state["active"] -= 1

    def close(self):
        assert self.state["active"] == 0, "must join every socket user before reset"
        self.closed = True


def make_dataset(views=3, *, overlap=False, fail=None):
    state = {"lock": threading.Lock(), "active": 0, "peak": 0, "decoders": [], "fail": fail}
    if overlap:
        state["barrier"] = threading.Barrier(2)
    dataset = UnifiedAction128Dataset.__new__(UnifiedAction128Dataset)
    dataset.root = Path("egopro")
    dataset.duration = 0.1
    dataset.decoder = CameraDecoder(state)
    dataset.zarr_tar_decoder = CameraDecoder(state)
    roles = ["main", "left_wrist", "right_wrist"][:views]
    episode = {
        "view_mapping": {role: role for role in roles},
        "video_views": [
            dict(
                role=role,
                source_key=role,
                path=role + ".mp4",
                width=64,
                height=192,
                from_timestamp=0,
                to_timestamp=1,
                fps=10,
            )
            for role in roles
        ],
    }
    return dataset, episode, state


@pytest.mark.parametrize("views", [1, 2, 3])
def test_parallel_camera_outputs_equal_serial(monkeypatch, views):
    monkeypatch.delenv("COSMOS_DATALOADER_TIMING", raising=False)
    monkeypatch.delenv("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", raising=False)
    baseline, episode, _ = make_dataset(views)
    expected = baseline._video(episode, 0, 0.1, 10)
    monkeypatch.setenv("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", "egostandard, egopro")
    candidate, episode, state = make_dataset(views, overlap=views > 1)
    actual = candidate._video(episode, 0, 0.1, 10)
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3], strict=True))
    assert actual[3:] == expected[3:]
    assert state["peak"] == min(views, 2)
    assert len(getattr(candidate, "_parallel_view_decoders", {})) == views - 1
    # A second window reuses the same bounded helper set.
    candidate._video(episode, 0, 0.1, 10)
    assert len(state["decoders"]) == max(views + 1, 2)


def test_parallel_camera_failure_joins_and_resets_all_helpers(monkeypatch):
    monkeypatch.setenv("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", "egopro")
    candidate, episode, state = make_dataset(overlap=True, fail="left_wrist")
    with pytest.raises(MediaDecodeError, match="broken camera") as caught:
        candidate._video(episode, 0, 0.1, 10)
    assert caught.value.context["camera"] == "left_wrist"
    assert state["active"] == 0
    assert all(decoder.closed for decoder in state["decoders"])
    # Failed windows must not poison the next request.
    state["fail"] = None
    candidate._video(episode, 0, 0.1, 10)


def test_unlisted_source_stays_serial(monkeypatch):
    monkeypatch.setenv("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", "egostandard")
    candidate, episode, state = make_dataset()
    candidate._video(episode, 0, 0.1, 10)
    assert state["peak"] == 1
    assert not hasattr(candidate, "_parallel_view_decoders")


def test_independent_decoder_preserves_limits_without_live_state():
    original = SupervisedVideoDecoder(
        timeout_seconds=19,
        video_handle_cache_size=2,
        max_window_bytes=123456,
        reuse_seek_groups=True,
        video_decode_threads=4,
    )
    clone = original.independent_decoder()
    assert clone.decode_options == original.decode_options
    assert clone.timeout_seconds == original.timeout_seconds
    assert clone.video_handle_cache_size == original.video_handle_cache_size
    assert clone.video_decode_threads == original.video_decode_threads
    assert clone._process is clone._socket is None


@pytest.mark.parametrize("window", [0, 1])
def test_parallel_full_state_action_sample_matches_serial(state_source, monkeypatch, window):
    root, _ = state_source
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    episodes = pq.read_table(path).to_pylist()
    for episode in episodes:
        original = episode["video_views"][0]
        episode["video_views"] = [
            {**original, "role": role, "source_key": role, "path": role + ".mp4"} for role in ["main", "left_wrist"]
        ]
        episode["view_mapping"] = {"main": "main", "left_wrist": "left_wrist"}
    pq.write_table(pa.Table.from_pylist(episodes), path)
    monkeypatch.setattr(
        "cosmos_framework.model.generator.reasoner.qwen3_vl.utils.tokenize_generation_caption",
        lambda *args, **kwargs: ([1, 2, 3], [False] * 3),
    )
    monkeypatch.delenv("COSMOS_DATALOADER_TIMING", raising=False)
    monkeypatch.delenv("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", raising=False)
    _, _, state = make_dataset(2)
    decoder = CameraDecoder(state)
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=decoder)
    expected = dataset[window]
    fingerprint = dataset.fingerprint
    rng = torch.get_rng_state()
    monkeypatch.setenv("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", root.name)
    state["barrier"] = threading.Barrier(2)
    actual = dataset[window]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value, key
    assert dataset.fingerprint == fingerprint
    assert torch.equal(torch.get_rng_state(), rng)
    assert state["peak"] == 2
