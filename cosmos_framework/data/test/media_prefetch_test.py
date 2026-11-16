"""Selective speculation must stay cheap, bounded, disposable and ordered."""

import gc
import multiprocessing as mp
import threading
import time
import weakref

import av
import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.media_prefetch import (
    MediaPrefetchPlan,
    SelectiveMediaPrefetch,
    SharedMediaBudget,
)
from cosmos_framework.data.generator.action.unified_action128 import BoundedRows, UnifiedAction128Dataset
from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor
from cosmos_framework.data.generator.media_decode import SupervisedVideoDecoder
from cosmos_framework.data.test.action128_test import source as source
from cosmos_framework.data.test.pretrain_state_test import state_source as state_source


def until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "prefetch made no progress"
        time.sleep(0.005)


def candidates(count=40):
    return ((0, i, i) for i in range(count))


def test_light_only_metadata_horizon_does_not_decode_or_scan_forever():
    planned, decoded = [], []

    def plan(candidate):
        planned.append(candidate)
        return None

    cache = SelectiveMediaPrefetch(
        candidates(), plan, decoded.append, budget_bytes=8, lookahead=8, minimum_work=1, size=len
    )
    try:
        until(lambda: cache.stats["plan_count"] == 8)
        assert len(planned) == 8
        assert not decoded
        cache.advance(0, 4)
        until(lambda: len(planned) == 12)
        assert cache.stats["plan_count"] == 8
        assert cache.stats["reserved_bytes"] == 0
    finally:
        cache.close()


def test_only_heavy_far_windows_execute_and_release_output_budget():
    decoded = []

    def plan(candidate):
        epoch, position, index = candidate
        return MediaPrefetchPlan((epoch, index), (epoch, position), 4, 10 if index in {5, 9, 13} else 0)

    def materialize(plan):
        decoded.append(plan.key[1])
        return b"data"

    cache = SelectiveMediaPrefetch(
        candidates(), plan, materialize, budget_bytes=8, lookahead=16, minimum_work=1, size=len
    )
    try:
        until(lambda: cache.stats["completed"] == 2)
        assert decoded == [5, 9]
        assert cache.stats["peak_reserved_bytes"] == 8
        cache.advance(0, 5)
        assert cache.get((0, 5)) == b"data"
        until(lambda: cache.stats["completed"] == 3)
        assert decoded == [5, 9, 13]
        assert cache.stats["reserved_bytes"] == 8
        cache.advance(0, 14)
        until(lambda: cache.stats["reserved_bytes"] == 0)
    finally:
        cache.close()


def test_background_error_is_a_foreground_retry_not_an_early_skip():
    def plan(candidate):
        return MediaPrefetchPlan((0, 0), (0, 0), 4, 10)

    def broken(_):
        raise OSError("temporarily unavailable")

    cache = SelectiveMediaPrefetch(candidates(1), plan, broken, budget_bytes=4, lookahead=1, minimum_work=1, size=len)
    try:
        until(lambda: cache.stats["failed"] == 1)
        cache.advance(0, 0)
        assert cache.get((0, 0)) is None
        assert cache.stats["reserved_bytes"] == 0
    finally:
        cache.close()


def test_prediction_undershoot_falls_back_without_retaining_oversized_output():
    cache = SelectiveMediaPrefetch(
        candidates(1),
        lambda _: MediaPrefetchPlan((0, 0), (0, 0), 4, 1),
        lambda _: b"too large",
        budget_bytes=4,
        lookahead=1,
        minimum_work=1,
        size=len,
    )
    try:
        until(lambda: cache.stats["oversized"] == 1)
        cache.advance(0, 0)
        assert cache.get((0, 0)) is None
        assert cache.stats["reserved_bytes"] == 0
    finally:
        cache.close()


def test_candidate_preview_preserves_resume_variables_and_virtual_shard_order(monkeypatch):
    class Preview:
        closed = False

        def __init__(self, candidates):
            self.rows = [next(candidates) for _ in range(3)]

        def advance(self, epoch, position):
            pass

        def close(self):
            self.closed = True

    class Dataset:
        def get_shuffle_blocks(self):
            return [(0, 2), (2, 2)]

        def media_lookahead_enabled(self):
            return True

        def make_media_lookahead(self, candidates, **kwargs):
            return Preview(candidates)

        def get_item(self, index, *, epoch, media_prefetch):
            return {"index": index}

    distributor = EpisodeBlockDistributor(Dataset(), seed=42, name="preview")
    distributor._virtual_shards = [(1, 2)]
    prefix = "COSMOS_DL_STATE_preview_WORKER_0_"
    monkeypatch.setenv(prefix + "EPOCH", "3")
    monkeypatch.setenv(prefix + "INDEX", "0")
    cache = distributor.prepare_media_lookahead(0, 1, 0, 1)
    assert __import__("os").environ[prefix + "EPOCH"] == "3"
    assert __import__("os").environ[prefix + "INDEX"] == "0"
    stream = distributor.stream(0, 1, 0, 1)
    actual = [next(stream) for _ in range(3)]
    assert [(r["_dp_epoch"], r["_dp_stream_pos"], r["index"]) for r in actual] == cache.rows
    stream.close()
    assert cache.closed


def test_multiple_producers_share_one_rank_byte_budget():
    budget = SharedMediaBudget(mp.get_context("fork"), 4)

    def plan(candidate):
        return MediaPrefetchPlan((0, 0), (0, 0), 4, 10)

    caches = [
        SelectiveMediaPrefetch(
            candidates(1),
            plan,
            lambda _: b"data",
            budget_bytes=4,
            lookahead=1,
            minimum_work=1,
            size=len,
            shared_budget=budget,
        )
        for _ in range(2)
    ]
    try:
        until(lambda: sum(c.stats["completed"] for c in caches) == 1)
        first = next(c for c in caches if c.stats["completed"])
        first.advance(0, 0)
        assert first.get((0, 0)) == b"data"
        until(lambda: sum(c.stats["completed"] for c in caches) == 2)
        assert budget.stats["peak_reserved_bytes"] == 4
    finally:
        for cache in caches:
            cache.close()
    assert budget.stats["reserved_bytes"] == 0


def _reserve_in_child(budget, result):
    result.put(budget.reserve(4))


def test_budget_is_shared_across_forked_physical_workers():
    context = mp.get_context("fork")
    budget = SharedMediaBudget(context, 4)
    results = context.Queue()
    children = [context.Process(target=_reserve_in_child, args=(budget, results)) for _ in range(2)]
    try:
        for child in children:
            child.start()
        assert sorted(results.get(timeout=5) for _ in children) == [False, True]
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
        assert budget.stats["reserved_bytes"] == budget.stats["peak_reserved_bytes"] == 4
        budget.release(4)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        results.close()


def test_close_cancels_running_speculation_and_releases_ticket():
    started, stopped = threading.Event(), threading.Event()
    budget = SharedMediaBudget(mp.get_context("fork"), 4)

    def materialize(_):
        started.set()
        assert stopped.wait(timeout=5)
        return b"data"

    cache = SelectiveMediaPrefetch(
        candidates(1),
        lambda _: MediaPrefetchPlan((0, 0), (0, 0), 4, 1),
        materialize,
        budget_bytes=4,
        lookahead=1,
        minimum_work=1,
        size=len,
        cancel=stopped.set,
        shared_budget=budget,
    )
    assert started.wait(timeout=5)
    cache.close()
    assert not cache._thread.is_alive()
    assert budget.stats["reserved_bytes"] == 0


def test_producer_does_not_retain_a_consumed_window_during_next_decode():
    second_started, finish = threading.Event(), threading.Event()
    references = []

    def materialize(plan):
        if plan.key[1] == 1:
            second_started.set()
            assert finish.wait(timeout=5)
        tensor = torch.empty(4, dtype=torch.uint8)
        references.append(weakref.ref(tensor))
        return tensor

    cache = SelectiveMediaPrefetch(
        candidates(2),
        lambda c: MediaPrefetchPlan((0, c[2]), (0, c[1]), 4, 1),
        materialize,
        budget_bytes=4,
        lookahead=2,
        minimum_work=1,
        size=lambda t: t.numel(),
        cancel=finish.set,
    )
    try:
        until(lambda: cache.stats["completed"] == 1)
        cache.advance(0, 0)
        value = cache.get((0, 0))
        assert second_started.wait(timeout=5)
        del value
        gc.collect()
        assert references[0]() is None
    finally:
        cache.close()


@pytest.mark.parametrize("budget,depth", [(0, 8), (8, 0)])
def test_invalid_limits_rejected(budget, depth):
    with pytest.raises(ValueError):
        SelectiveMediaPrefetch(
            [], lambda _: None, lambda _: None, budget_bytes=budget, lookahead=depth, minimum_work=0, size=len
        )


def test_prefetched_real_video_preserves_full_state_sample_and_padding(state_source, monkeypatch):
    root, _ = state_source
    with av.open(str(root / "video.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(7):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 30, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    monkeypatch.setattr(
        "cosmos_framework.model.generator.reasoner.qwen3_vl.utils.tokenize_generation_caption",
        lambda *args, **kwargs: ([1, 2, 3], [False] * 3),
    )
    monkeypatch.delenv("COSMOS_DATALOADER_TIMING", raising=False)
    decoder = SupervisedVideoDecoder(timeout_seconds=20, reuse_seek_groups=True, video_decode_threads=4)
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=decoder)
    baseline = [dataset.get_item(i, epoch=0) for i in range(2)]
    monkeypatch.setenv("COSMOS_ACTION128_MEDIA_PREFETCH_SOURCES", root.name)
    monkeypatch.setenv("COSMOS_ACTION128_MEDIA_MIN_PIXEL_WORK", "0")
    monkeypatch.setenv("COSMOS_ACTION128_MEDIA_LOOKAHEAD", "8")
    monkeypatch.setenv("COSMOS_ACTION128_MEDIA_PREFETCH_BYTES", str(32 * 1024**2))
    cache = dataset.make_media_lookahead(candidates(2))
    try:
        until(lambda: cache.stats["completed"] == 2)
        episode = dataset.episodes.slice(0, 1).to_pylist()[0]
        thin = BoundedRows().episode_rows(root / episode["data_path"], episode, video_only=True)
        assert "action" not in thin.column_names
        assert "observation.state" not in thin.column_names
        for index in range(2):
            cache.advance(0, index)
            actual = dataset.get_item(index, epoch=0, media_prefetch=cache)
            for key, expected in baseline[index].items():
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(actual[key], expected), key
                else:
                    assert actual[key] == expected, key
        assert cache.stats["hits"] == 2
        assert cache.stats["oversized"] == 0
    finally:
        cache.close()
        decoder.close()
