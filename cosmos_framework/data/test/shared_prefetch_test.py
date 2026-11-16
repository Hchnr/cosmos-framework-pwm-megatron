# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Global capacity, legacy alignment, multi-process resume and failure tests."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import random
import signal
import struct
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
    PoolPackingBatcher,
    SequentialPackingBatcher,
    SimpleBatcher,
)
from cosmos_framework.data.generator.dataflow.loader import JointCosmosDataLoader
from cosmos_framework.data.generator.dataflow.shared_prefetch import (
    LOADER_META_KEY,
    CosmosDataStream,
    SharedPrefetchDataLoader,
)
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY


class _Rows(torch.utils.data.Dataset):
    def __init__(self, n=17):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"id": torch.tensor(index)}

    def get_shuffle_blocks(self):
        return [(index, 1) for index in range(self.n)]


class _Processor(IdentityProcessor):
    def __init__(self, *, stochastic=False, delay=0, fail=False, crash=False, drop=False):
        self.stochastic = stochastic
        self.delay = delay
        self.fail = fail
        self.crash = crash
        self.drop = drop

    def process(self, item):
        if self.crash:
            os._exit(23)
        if self.fail:
            raise ValueError("fixture decode failed")
        if self.delay:
            time.sleep(self.delay)
        index = int(item["id"])
        if self.drop and index == 1:
            return {DROP_SAMPLE_KEY: True}
        result = {
            "id": item["id"],
            "text_token_ids": torch.arange(2),
            "video": torch.full((3, 1, 32, 32), index, dtype=torch.uint8),
            "mask": torch.tensor([True, False]),
        }
        if self.stochastic:
            result["augmentation"] = torch.tensor([random.random(), np.random.random(), torch.rand(()).item()])
        return result


class _UnserializableCollator:
    def collate(self, samples):
        return {"opaque": threading.Lock()}


def _build(
    *,
    shared=True,
    workers=2,
    depth=2,
    logical=(2, 1),
    processor=None,
    batcher=None,
    collator=None,
    n=17,
    episode=False,
    **kwargs,
):
    streams = {}
    for index, count in enumerate(logical):
        name = f"source_{index}"
        distributor = (
            EpisodeBlockDistributor(_Rows(n), seed=42, name=name, dataset_fingerprint=name)
            if episode
            else MapDistributor(_Rows(n), shuffle=False, name=name)
        )
        loader = CosmosDataLoader(
            distributor=distributor,
            processor=copy.deepcopy(processor or _Processor()),
            batcher=copy.deepcopy(batcher or SequentialPackingBatcher(max_samples_per_batch=2, lookahead_limit=1)),
            collator=copy.deepcopy(collator),
            num_workers=count,
            prefetch_factor=1 if count else None,
            persistent_workers=count > 0,
        )
        streams[name] = {
            "dataloader": CosmosDataStream(loader.dataset, count) if shared else loader,
            "ratio": index + 1,
        }
    if shared:
        return SharedPrefetchDataLoader(streams, seed=42, num_workers=workers, prefetch_factor=depth, **kwargs)
    return JointCosmosDataLoader(streams, seed=42, selection_mode="exact_periodic")


def _close(loader):
    if isinstance(loader, SharedPrefetchDataLoader):
        loader.close()
    elif loader._iterators:
        for iterator in loader._iterators:
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()


_PARTIAL_RESULT_STARTED = None


def _partial_result_worker(requests, results, *args, **kwargs):
    requests.get()
    writer = results._queue._writer
    os.write(writer.fileno(), struct.pack("!i", 100_000) + b"incomplete")
    _PARTIAL_RESULT_STARTED.set()
    time.sleep(60)


def _close_partial_result_loader():
    import cosmos_framework.data.generator.dataflow.shared_prefetch as shared_module

    global _PARTIAL_RESULT_STARTED
    os.setsid()
    _PARTIAL_RESULT_STARTED = mp.get_context("fork").Event()
    shared_module._worker_main = _partial_result_worker
    loader = _build(workers=1, depth=1, receive_in_background=True)
    loader._start()
    loader._fill()
    assert _PARTIAL_RESULT_STARTED.wait(timeout=3)
    time.sleep(0.1)  # Let the IPC thread enter the partial payload read.
    started = time.monotonic()
    loader.close()
    assert time.monotonic() - started < 6
    assert not loader._result_thread.is_alive()
    assert all(not process.is_alive() for process in loader._processes)
    loader.close()


def test_close_does_not_hang_on_partial_ipc_message():
    process = mp.get_context("fork").Process(target=_close_partial_result_loader)
    process.start()
    try:
        process.join(timeout=10)
        assert not process.is_alive(), "loader.close() blocked on an incomplete IPC message"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            # Only this isolated test session and its own worker are killed.
            os.killpg(process.pid, signal.SIGKILL)
            process.join(timeout=3)


def _assert_batch(actual, expected):
    actual = {key: value for key, value in actual.items() if key != LOADER_META_KEY}
    expected = {key: value for key, value in expected.items() if key != LOADER_META_KEY}
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize("receive_seconds", [9.99, 10.0])
def test_completed_receive_wait_is_reported_without_changing_batch_or_credits(monkeypatch, receive_seconds):
    from cosmos_framework.data.generator.dataflow import shared_prefetch

    loader = _build(workers=1, depth=2, logical=(0, 0))
    try:
        iter(loader)  # Start the real producer before replacing the consumer clock.
        ticks = iter([100.0, 100.0 + receive_seconds])
        monkeypatch.setattr(
            shared_prefetch,
            "time",
            SimpleNamespace(perf_counter=lambda: next(ticks), monotonic=time.monotonic),
        )
        with patch.object(shared_prefetch.log, "warning") as warning:
            batch = next(loader)
        meta = batch[LOADER_META_KEY]
        assert meta["global_id"] == 0
        assert loader.prefetch_stats["inflight"] == 2
        assert loader.prefetch_stats["receive_seconds"] == pytest.approx(receive_seconds)
        reports = [call for call in warning.call_args_list if "Shared dataloader batch received" in str(call)]
        assert len(reports) == int(receive_seconds >= 10.0)
        if reports:
            message = reports[0].args[0]
            assert "after 10.00s" in message
            assert "global microbatch 0" in message
            assert f"source_{meta['source']}" in message
            assert reports[0].kwargs["rank0_only"] is False
    finally:
        loader.close()


@pytest.mark.parametrize("workers,depth", [(0, 1), (1, 1), (1, 4), (3, 1), (3, 5)])
@pytest.mark.parametrize("stochastic", [False, True])
def test_physical_workers_and_global_depth_preserve_legacy_samples_and_packs(workers, depth, stochastic):
    torch.manual_seed(900)
    legacy = _build(shared=False, processor=_Processor(stochastic=stochastic, drop=True))
    try:
        expected = [next(stream) for stream in [iter(legacy)] for _ in range(18)]
    finally:
        _close(legacy)
    torch.manual_seed(900)
    shared = _build(workers=workers, depth=depth, processor=_Processor(stochastic=stochastic, drop=True))
    try:
        for ref in expected:
            actual = next(iter(shared))
            _assert_batch(actual, ref)
            assert shared.prefetch_stats["inflight"] <= depth
        assert shared.prefetch_stats["max_observed_inflight"] == (depth if workers else 0)
    finally:
        shared.close()
    assert all(not process.is_alive() for process in shared._processes)


@pytest.mark.parametrize(
    "batcher",
    [
        SimpleBatcher(3),
        SequentialPackingBatcher(max_sequence_length=10, lookahead_limit=1),
        PoolPackingBatcher(max_tokens=100, pool_size=4, max_batch_size=1),
    ],
)
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("collect_timing", [False, True])
def test_resume_reconstructs_lookahead_rng_and_round_robin_without_committing_prefetch(
    batcher, workers, collect_timing, monkeypatch
):
    monkeypatch.setenv("COSMOS_DATALOADER_TIMING", "1" if collect_timing else "0")
    with patch.dict(os.environ):
        original = _build(workers=2, depth=4, processor=_Processor(stochastic=True), batcher=batcher)
        callback = JointCosmosDataLoaderStateCallback(original)
        try:
            iterator = iter(original)
            for _ in range(7):
                callback._update_state_from_batch(next(iterator))
            state = copy.deepcopy(callback.state_dict())
            assert state["global_id"] == 7
            # Simulate trainer holding extra uncommitted data at checkpoint time.
            expected = [next(iterator) for _ in range(9)]
            assert callback.state_dict()["global_id"] == 7
            assert state["__execution__"] == callback.state_dict()["__execution__"]
            training_rng = torch.get_rng_state()
            resumed = _build(workers=workers, depth=1, processor=_Processor(stochastic=True), batcher=batcher)
            resumed_callback = JointCosmosDataLoaderStateCallback(resumed)
            resumed_callback.load_state_dict(state)
            resumed.preserve_rng_on_resume()
            try:
                for ref in expected:
                    batch = next(iter(resumed))
                    _assert_batch(batch, ref)
                    resumed_callback._update_state_from_batch(batch)
                torch.testing.assert_close(torch.get_rng_state(), training_rng, rtol=0, atol=0)
                assert resumed_callback.state_dict()["global_id"] == 16
            finally:
                resumed.close()
        finally:
            original.close()


def test_legacy_checkpoint_restores_next_logical_worker_and_pack_boundary():
    with patch.dict(os.environ):
        legacy = _build(shared=False)
        callback = JointCosmosDataLoaderStateCallback(legacy)
        try:
            iterator = iter(legacy)
            for _ in range(7):
                callback._update_state_from_batch(next(iterator))
            state = callback.state_dict()
            expected = [next(iterator) for _ in range(10)]
        finally:
            _close(legacy)
        shared = _build(workers=1)
        try:
            JointCosmosDataLoaderStateCallback(shared).load_state_dict(state)
            for ref in expected:
                _assert_batch(next(iter(shared)), ref)
        finally:
            shared.close()


def test_global_capacity_includes_completed_results_and_does_not_initialize_unselected_sources():
    shared = _build(workers=3, depth=2, logical=(1,) * 8)
    try:
        first = next(iter(shared))
        time.sleep(0.1)
        assert shared.prefetch_stats["inflight"] == 2
        assert shared._next_schedule_id == 3
        assert first[LOADER_META_KEY]["global_id"] == 0
        assert shared._global_id == 1
    finally:
        shared.close()


def _fake_pin_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)


@pytest.mark.parametrize("fail", [False, True])
def test_result_transfer_defers_diagnostic_signal_and_restores_mask(monkeypatch, fail):
    from cosmos_framework.data.generator.dataflow.shared_prefetch import _ResultQueue

    channel = _ResultQueue(torch.multiprocessing.get_context("fork"))
    received = []
    previous_handler = signal.signal(signal.SIGUSR1, lambda signum, frame: received.append(signum))
    previous_mask = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})
    expected_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def transfer(result):
        signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)
        assert received == [], "diagnostics must not interrupt tensor shared-memory allocation"
        if fail:
            raise ValueError("fixture serialization failure")

    monkeypatch.setattr(channel._queue, "put", transfer)
    try:
        if fail:
            with pytest.raises(ValueError, match="fixture serialization failure"):
                channel.put(None)
        else:
            channel.put(None)
        assert received == [signal.SIGUSR1]
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == expected_mask
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        signal.signal(signal.SIGUSR1, previous_handler)
        channel.close()


def test_cpu_background_receive_drains_idle_workers_without_cuda_or_extra_credits(monkeypatch):
    import gc
    import weakref

    def unexpected_cuda(*args, **kwargs):
        raise AssertionError("CPU result receiving must not touch CUDA")

    for name in ["is_available", "current_device", "set_device"]:
        monkeypatch.setattr(torch.cuda, name, unexpected_cuda)
    loader = _build(workers=2, depth=3, processor=_Processor(stochastic=True))
    try:
        batch = next(iter(loader))
        reference = weakref.ref(batch["video"])
        deadline = time.monotonic() + 5
        while loader._received_results.qsize() != 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert loader._received_results.qsize() == 3
        assert loader.prefetch_stats["background_receive"]
        assert not loader.prefetch_stats["background_pin_memory"]
        assert loader.prefetch_stats["inflight"] == 3
        assert loader._next_schedule_id == 4
        assert loader.committed_global_id == 0
        loader.commit_batch(batch)
        assert loader.committed_global_id == 1
        del batch
        gc.collect()
        assert reference() is None
    finally:
        try:
            loader.close()
        finally:
            monkeypatch.undo()
    assert not loader._result_thread.is_alive()
    assert loader._received_results.empty()
    assert all(not process.is_alive() for process in loader._processes)


def test_cpu_result_receiving_can_be_disabled():
    loader = _build(workers=2, depth=3, receive_in_background=False)
    try:
        next(iter(loader))
        assert loader._result_thread is None
        assert loader._received_results is None
        assert not loader.prefetch_stats["background_receive"]
    finally:
        loader.close()


@pytest.mark.parametrize("background", [False, True])
def test_opt_in_task_timing_preserves_batches_and_accounts_for_the_same_bounded_tasks(monkeypatch, background):
    monkeypatch.setenv("COSMOS_DATALOADER_TIMING", "0")
    torch.manual_seed(150)
    reference = _build(workers=2, depth=3, processor=_Processor(stochastic=True))
    try:
        expected = [next(iter(reference)) for _ in range(6)]
        assert all("timing" not in batch[LOADER_META_KEY] for batch in expected)
        assert reference._enqueued_ns == {}
    finally:
        reference.close()

    monkeypatch.setenv("COSMOS_DATALOADER_TIMING", "1")
    torch.manual_seed(150)
    loader = _build(
        workers=2,
        depth=3,
        worker_assignment=[0, 0, 0],
        receive_in_background=background,
        processor=_Processor(stochastic=True, delay=0.02),
    )
    timings = []
    try:
        for reference_batch in expected:
            batch = next(iter(loader))
            _assert_batch(batch, reference_batch)
            assert "_cosmos_prefetch_timing" not in batch
            timing = batch[LOADER_META_KEY]["timing"]
            stamps = [
                timing[key]
                for key in ["enqueued_ns", "worker_started_ns", "worker_processed_ns", "received_ns", "delivered_ns"]
            ]
            assert stamps == sorted(stamps)
            assert timing["physical_worker"] == 0
            assert timing["prepare_seconds"] >= 0.02
            assert timing["lead_seconds"] == pytest.approx(
                sum(timing[f"{phase}_seconds"] for phase in ["queue", "prepare", "transfer", "ready"])
            )
            assert len(loader._enqueued_ns) == loader.prefetch_stats["inflight"] == 3
            loader.commit_batch(batch)
            timings.append(timing)
        # All logical streams deliberately share one executor. Later tasks
        # include earlier tasks' preparation in their measured queue delay.
        assert max(timing["queue_seconds"] for timing in timings) >= 0.02
    finally:
        loader.close()
    assert loader._enqueued_ns == {}


def test_cpu_background_receive_keeps_explicit_inline_pinning(monkeypatch):
    from cosmos_framework.data.generator.dataflow import shared_prefetch

    _fake_pin_device(monkeypatch)
    calls = []

    def pin(batch):
        assert "_cosmos_resume_rng" not in batch
        calls.append(threading.current_thread().name)
        return batch

    monkeypatch.setattr(shared_prefetch, "pin_memory", pin)
    loader = _build(workers=2, depth=3, pin_memory=True, pin_memory_in_background=False)
    try:
        batch = next(iter(loader))
        assert calls == [threading.current_thread().name]
        assert loader.prefetch_stats["background_receive"]
        assert not loader.prefetch_stats["background_pin_memory"]
        assert batch[LOADER_META_KEY]["resume_rng"] is not None
    finally:
        loader.close()


def test_background_pinning_prepares_ahead_without_extra_credits_or_batch_retention(monkeypatch):
    import gc
    import weakref

    from cosmos_framework.data.generator.dataflow import shared_prefetch

    _fake_pin_device(monkeypatch)
    prepared = threading.Event()
    calls = []

    def pin(batch):
        assert "_cosmos_resume_rng" not in batch
        calls.append(threading.current_thread().name)
        if len(calls) == 4:
            prepared.set()
        return batch

    monkeypatch.setattr(shared_prefetch, "pin_memory", pin)
    loader = _build(workers=2, depth=3, pin_memory=True)
    try:
        batch = next(iter(loader))
        reference = weakref.ref(batch["video"])
        assert batch[LOADER_META_KEY]["resume_rng"] is not None
        assert prepared.wait(5), "pinning must advance while the consumer is idle"
        assert calls == ["cosmos-data-pin"] * 4
        assert loader.prefetch_stats["inflight"] == 3
        assert loader._next_schedule_id == 4
        assert loader.committed_global_id == 0
        loader.commit_batch(batch)
        assert loader.committed_global_id == 1
        del batch
        gc.collect()
        assert reference() is None
    finally:
        loader.close()
    assert not loader._result_thread.is_alive()
    assert loader._received_results.empty()
    assert all(not process.is_alive() for process in loader._processes)


@pytest.mark.parametrize("initialization", [False, True])
def test_pinning_failures_propagate_and_close_thread_and_workers(monkeypatch, initialization):
    from cosmos_framework.data.generator.dataflow import shared_prefetch

    _fake_pin_device(monkeypatch)

    def fail(*args):
        raise ValueError("fixture pinning failure")

    monkeypatch.setattr(
        torch.cuda if initialization else shared_prefetch, "set_device" if initialization else "pin_memory", fail
    )
    loader = _build(workers=2, depth=3, pin_memory=True)
    try:
        with pytest.raises(RuntimeError, match="fixture pinning failure"):
            next(iter(loader))
        assert not loader._result_thread.is_alive()
        assert all(not process.is_alive() for process in loader._processes)
    finally:
        loader.close()


@pytest.mark.parametrize("delivery", ["inline", "cpu", "pin"])
@pytest.mark.parametrize("mode", ["exception", "crash", "timeout"])
def test_worker_errors_are_bounded_and_close_the_pool(mode, delivery, monkeypatch):
    if delivery == "pin":
        from cosmos_framework.data.generator.dataflow import shared_prefetch

        _fake_pin_device(monkeypatch)
        monkeypatch.setattr(shared_prefetch, "pin_memory", lambda batch: batch)
    processor = _Processor(fail=mode == "exception", crash=mode == "crash", delay=2 if mode == "timeout" else 0)
    shared = _build(
        workers=2,
        processor=processor,
        worker_timeout=0.15 if mode == "timeout" else 10,
        pin_memory=delivery == "pin",
        receive_in_background=delivery == "cpu",
    )
    try:
        with pytest.raises((RuntimeError, TimeoutError), match="failed|worker exited|timed out"):
            next(iter(shared))
        assert all(not process.is_alive() for process in shared._processes)
    finally:
        shared.close()
        shared.close()


@pytest.mark.parametrize("delivery", ["inline", "cpu", "pin"])
def test_result_serialization_failure_reaches_consumer(delivery, monkeypatch):
    if delivery == "pin":
        from cosmos_framework.data.generator.dataflow import shared_prefetch

        _fake_pin_device(monkeypatch)
        monkeypatch.setattr(shared_prefetch, "pin_memory", lambda batch: batch)
    loader = _build(
        workers=1,
        logical=(0,),
        collator=_UnserializableCollator(),
        worker_timeout=2,
        pin_memory=delivery == "pin",
        receive_in_background=delivery == "cpu",
    )
    try:
        with pytest.raises(RuntimeError, match="cannot pickle.*lock"):
            next(iter(loader))
        assert all(not process.is_alive() for process in loader._processes)
    finally:
        loader.close()


def test_timeout_identifies_the_active_source_and_dumps_its_worker_stack(capfd):
    shared = _build(workers=1, processor=_Processor(delay=5), worker_timeout=0.5)
    try:
        with pytest.raises(TimeoutError) as caught:
            next(iter(shared))
        message = str(caught.value)
        assert "global microbatch 0" in message
        assert f"dataset={shared._names[shared.dataset_index(0)]!r}" in message
        assert f"physical_worker=0, pid={shared._processes[0].pid}" in message
        assert "active_microbatch=0" in message
        assert "last_submitted_microbatch=-1" in message
        assert "shared_prefetch_test.py" in capfd.readouterr().err
        assert all(not process.is_alive() for process in shared._processes)
    finally:
        shared.close()


def test_slow_worker_delivers_after_a_diagnostic_stack_dump(capfd):
    shared = _build(
        workers=1, logical=(0,), processor=_Processor(delay=1.3), batcher=SimpleBatcher(1), worker_timeout=2
    )
    try:
        batch = next(iter(shared))
        assert int(batch["id"][0]) == 0
        assert "shared_prefetch_test.py" in capfd.readouterr().err
        assert shared._processes[0].is_alive()
    finally:
        shared.close()


def test_empty_shards_are_skipped_without_changing_valid_source_rotation():
    legacy = _build(shared=False, logical=(4, 3), n=2, batcher=SimpleBatcher(1))
    shared = _build(workers=1, logical=(4, 3), n=2, batcher=SimpleBatcher(1))
    try:
        old = iter(legacy)
        for _ in range(15):
            _assert_batch(next(iter(shared)), next(old))
    finally:
        shared.close()
        _close(legacy)


def test_topology_change_is_rejected_but_physical_mapping_can_change():
    original = _build()
    state = JointCosmosDataLoaderStateCallback(original).state_dict()
    changed = _build(logical=(1, 1))
    try:
        with pytest.raises(ValueError, match="topology changed"):
            JointCosmosDataLoaderStateCallback(changed).load_state_dict(state)
    finally:
        original.close()
        changed.close()


def test_explicit_affinity_and_commit_order():
    shared = _build(workers=2, worker_assignment=[1, 1, 0])
    try:
        first, second = next(iter(shared)), next(iter(shared))
        with pytest.raises(ValueError, match="commits must follow"):
            shared.commit_batch(second)
        shared.commit_batch(first)
        shared.commit_batch(second)
        assert shared.committed_global_id == 2
    finally:
        shared.close()


def _with_equivalent_virtual_shards(state):
    execution = state["__execution__"]
    counts = execution["topology"]["logical_workers"]
    execution["virtual_shards"] = [[[worker, max(1, count)] for worker in range(max(1, count))] for count in counts]
    execution["virtual_rng_seeds"] = [
        [[seed, worker] for worker in range(max(1, count))]
        for seed, count in zip(execution["base_seeds"], counts, strict=True)
    ]
    return state


def test_virtual_resume_retains_explicit_physical_mapping_without_changing_batches_or_rng():
    with patch.dict(os.environ):
        original = _build(workers=2, depth=3, episode=True, processor=_Processor(stochastic=True))
        callback = JointCosmosDataLoaderStateCallback(original)
        try:
            for _ in range(7):
                callback._update_state_from_batch(next(original))
            state = _with_equivalent_virtual_shards(copy.deepcopy(callback.state_dict()))
            expected = [next(original) for _ in range(9)]
        finally:
            original.close()
        assignment = [1, 1, 0]
        restored = _build(
            workers=2,
            depth=1,
            episode=True,
            processor=_Processor(stochastic=True),
            worker_assignment=assignment,
        )
        assignment[0] = 0  # Configuration is copied rather than retained by reference.
        try:
            JointCosmosDataLoaderStateCallback(restored).load_state_dict(state)
            assert list(restored._affinity.values()) == [1, 1, 0]
            for reference in expected:
                _assert_batch(next(restored), reference)
        finally:
            restored.close()


def test_virtual_resume_rejects_assignment_for_a_different_active_shard_count():
    original = _build(workers=0, episode=True)
    try:
        next(original)
        state = _with_equivalent_virtual_shards(JointCosmosDataLoaderStateCallback(original).state_dict())
    finally:
        original.close()
    restored = _build(workers=2, logical=(0, 0), episode=True, worker_assignment=[1, 0])
    try:
        with pytest.raises(ValueError, match="worker_assignment length.*virtual-shard"):
            JointCosmosDataLoaderStateCallback(restored).load_state_dict(state)
        assert not restored._started
        assert list(restored._affinity.values()) == [1, 0]
    finally:
        restored.close()


@pytest.mark.parametrize("assignment", [[1, 0], [2, 1, 0], [-1, 1, 0], [True, 1, 0], ["1", 1, 0]])
def test_explicit_mapping_rejects_wrong_length_or_invalid_physical_ids(assignment):
    with pytest.raises(ValueError, match="worker_assignment"):
        _build(workers=2, worker_assignment=assignment)


def test_cp_reused_batch_commits_once_at_window_end_and_rejects_midwindow_save():
    from types import SimpleNamespace

    shared = _build(workers=1)
    callback = JointCosmosDataLoaderStateCallback(shared)
    window = SimpleNamespace(active=True)
    callback.trainer = SimpleNamespace(_cp_data_window=window)
    try:
        batch = next(iter(shared))
        callback.on_training_step_batch_end(None, batch, {}, None)
        assert shared.committed_global_id == 0
        with pytest.raises(ValueError, match="CP data-window boundary"):
            callback.state_dict()
        window.active = False
        callback.on_training_step_batch_end(None, batch, {}, None)
        assert callback.state_dict()["global_id"] == 1
    finally:
        shared.close()


def test_reject_shared_state_in_legacy_backend_instead_of_silent_wrong_resume():
    shared, legacy = _build(), _build(shared=False)
    try:
        state = JointCosmosDataLoaderStateCallback(shared).state_dict()
        with pytest.raises(ValueError, match="requires shared_prefetch"):
            JointCosmosDataLoaderStateCallback(legacy).load_state_dict(state)
    finally:
        shared.close()
        _close(legacy)


def _distributed_rank(rank, path):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    loader = _build(workers=1, depth=2, n=1024)
    try:
        trace = []
        callback = JointCosmosDataLoaderStateCallback(loader)
        for _ in range(12):
            batch = next(iter(loader))
            trace.append((batch["dataset_name"], batch["id"].tolist()))
            callback._update_state_from_batch(batch)
        traces = [None, None]
        dist.all_gather_object(traces, trace)
        assert [name for name, _ in traces[0]] == [name for name, _ in traces[1]]
        for source in ("source_0", "source_1"):
            ids = [{index for name, group in other if name == source for index in group} for other in traces]
            assert ids[0].isdisjoint(ids[1])
    finally:
        loader.close()
        dist.destroy_process_group()


def test_two_dp_ranks_share_schedule_but_have_disjoint_sample_shards(tmp_path):
    torch.multiprocessing.start_processes(
        _distributed_rank, args=(str(tmp_path / "gloo_init"),), nprocs=2, start_method="fork", join=True
    )


def _assert_tree(actual, expected):
    import dataclasses

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        # Timing is expected to depend on execution resources; training payload
        # and packing metadata are compared without tolerance.
        ignored = {key for key in expected if key.startswith(("_sample_time", "_aug_", "_pre_aug", "_worker_"))}
        ignored |= {LOADER_META_KEY}
        assert actual.keys() - ignored == expected.keys() - ignored
        for key in expected.keys() - ignored:
            _assert_tree(actual[key], expected[key])
    elif dataclasses.is_dataclass(expected):
        _assert_tree(dataclasses.asdict(actual), dataclasses.asdict(expected))
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree(left, right)
    elif hasattr(expected, "__dict__"):
        assert type(actual) is type(expected)
        _assert_tree(vars(actual), vars(expected))
    else:
        assert actual == expected


def test_real_droid_catalog_payload_pack_and_checkpoint_match_legacy(monkeypatch):
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from cosmos_framework.data.generator.catalog_processors import build_catalog_joint_dataloader
    from cosmos_framework.data.generator.dataflow.collators import VFMListCollator

    catalog = Path("tests/data/cosmos_framework/data/catalog_samples/action_droid.parquet")
    if not catalog.exists():
        # Canonical repository fixture directory (see catalog_test.SAMPLES).
        from cosmos_framework.data.test.catalog_test import SAMPLES

        catalog = SAMPLES / "action_droid.parquet"
    root = Path(json.loads(pq.read_table(catalog, columns=["data"])["data"][0].as_py())["root"])
    if not (root / "success/meta/info.json").exists():
        pytest.skip("official DROID golden subset not installed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tokenizer_path = Path("/share/project/eai_pwm/models/Cosmos3-Nano")
    if not (tokenizer_path / "tokenizer.json").exists():
        pytest.skip("Cosmos3-Nano tokenizer not installed")
    with patch.dict(os.environ):

        def build(backend, workers):
            return build_catalog_joint_dataloader(
                [{"path": str(catalog), "ratio": ratio} for ratio in (1, 2)],
                components={
                    "action": {
                        "batcher": SequentialPackingBatcher(max_samples_per_batch=2, lookahead_limit=1),
                        "collator": VFMListCollator(),
                    }
                },
                num_workers=workers,
                logical_workers=[1, 1],
                prefetch_factor=2,
                execution_backend=backend,
                tokenizer_config={
                    "_target_": "cosmos_framework.data.generator.processors.qwen3vl_processor.Qwen3VLProcessor",
                    "name": str(tokenizer_path),
                },
            )

        legacy, shared = build("legacy_per_source", 2), build("shared_prefetch", 1)
        old_callback = JointCosmosDataLoaderStateCallback(legacy)
        new_callback = JointCosmosDataLoaderStateCallback(shared)
        try:
            initial_rng = torch.get_rng_state()
            old = iter(legacy)
            # JointCosmosDataLoader is a lazy generator: worker seeds are only
            # drawn on the first next(), not on iter().
            first = next(old)
            torch.set_rng_state(initial_rng)
            iter(shared)
            for index in range(5):
                expected, actual = first if index == 0 else next(old), next(iter(shared))
                _assert_tree(actual, expected)
                old_callback._update_state_from_batch(expected)
                new_callback._update_state_from_batch(actual)
            state = new_callback.state_dict()
            expected = next(old)
            restored = build("shared_prefetch", 2)
            try:
                JointCosmosDataLoaderStateCallback(restored).load_state_dict(state)
                _assert_tree(next(iter(restored)), expected)
            finally:
                restored.close()
        finally:
            _close(legacy)
            shared.close()


class _DropEverything(IdentityProcessor):
    def process(self, item):
        return {DROP_SAMPLE_KEY: True}


def test_inline_all_dropped_stream_fails_instead_of_looping_forever():
    shared = _build(workers=0, logical=(1,), processor=_DropEverything())
    try:
        with pytest.raises(RuntimeError, match="1000 consecutive"):
            next(iter(shared))
    finally:
        shared.close()


def test_idle_synchronous_stream_does_not_retain_the_delivered_media():
    import gc
    import weakref

    from cosmos_framework.data.generator.dataflow.shared_prefetch import _StreamRunner

    loader = _build(workers=0, logical=(1,), batcher=SimpleBatcher(1))
    runner = _StreamRunner(loader._loaders, [42], {})
    try:
        batch = runner.next_batch(0, 0)
        reference = weakref.ref(batch["video"])
        del batch
        gc.collect()
        assert reference() is None
    finally:
        runner.close()
        loader.close()


@pytest.mark.gpus(1)
def test_cuda_training_and_checkpoint_resume_preserve_model_and_rng(tmp_path):
    """Exercise actual H2D/backward and persisted loader state on a GPU."""
    from cosmos_framework.utils.misc import to

    torch.manual_seed(123)
    model = torch.nn.Linear(4, 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loader = _build(workers=2, depth=3, processor=_Processor(stochastic=True), pin_memory=True)
    callback = JointCosmosDataLoaderStateCallback(loader)

    def step(current, state_callback):
        batch = next(iter(current))
        if current is loader:
            assert current._result_thread is not None
            assert batch["video"].is_pinned() and batch["id"].is_pinned()
        batch = to(batch, device="cuda")
        features = torch.cat((batch["id"].float().unsqueeze(1), batch["augmentation"]), dim=1)
        loss = model(features).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state_callback.on_training_step_batch_end(model, batch, {}, loss)
        return batch, loss.detach().clone()

    try:
        for _ in range(3):
            step(loader, callback)
        path = tmp_path / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loader": callback.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            path,
        )
        expected = [step(loader, callback) for _ in range(4)]
        expected_model = copy.deepcopy(model.state_dict())
        state = torch.load(path, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Restore into synchronous execution to also catch accidental CUDA RNG
        # reseeding by inline logical-stream initialization.
        resumed = _build(workers=0, depth=1, processor=_Processor(stochastic=True))
        restored_callback = JointCosmosDataLoaderStateCallback(resumed)
        with patch.dict(os.environ):
            restored_callback.load_state_dict(state["loader"])
            resumed.preserve_rng_on_resume()
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])
            try:
                for ref_batch, ref_loss in expected:
                    batch, loss = step(resumed, restored_callback)
                    _assert_batch(batch, ref_batch)
                    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.cuda.get_rng_state(), state["cuda_rng"], rtol=0, atol=0)
            finally:
                resumed.close()
    finally:
        loader.close()


class _PlanProcessor(_Processor):
    def process(self, item):
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        sample = super().process(item)
        sample["sequence_plan"] = SequencePlan(has_text=True, has_vision=True, supervise_text=False)
        return sample


def test_fixed_token_budget_and_static_packed_masks_align_across_workers():
    from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
    from cosmos_framework.data.generator.sequence_packing.static_layout import StaticPackingConfig

    special = {"eos_token_id": 10, "start_of_generation": 11, "end_of_generation": 12}
    batcher = SequentialPackingBatcher(
        max_sequence_length=18,
        lookahead_limit=1,
        fixed_sequence_length_enabled=True,
        fixed_und_tokens=13,
        fixed_gen_tokens=5,
        special_tokens=special,
    )
    collator = VFMListCollator(static_packing_config=StaticPackingConfig(), special_tokens=special)
    kwargs = {"processor": _PlanProcessor(stochastic=True), "batcher": batcher, "collator": collator}
    torch.manual_seed(888)
    legacy = _build(shared=False, **kwargs)
    try:
        expected = [next(iterator) for iterator in [iter(legacy)] for _ in range(12)]
    finally:
        _close(legacy)
    torch.manual_seed(888)
    shared = _build(workers=1, depth=3, **kwargs)
    try:
        for ref in expected:
            _assert_tree(next(iter(shared)), ref)
    finally:
        shared.close()


def test_shared_workers_reset_inherited_trainer_sigterm_handler():
    import signal
    from pathlib import Path

    previous = signal.getsignal(signal.SIGTERM)
    shared = None
    try:
        # The trainer's handler defers shutdown to a model step; a CPU data
        # worker must instead respond to terminate() immediately.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        shared = _build(workers=1)
        iter(shared)
        process = shared._processes[0]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{process.pid}/status").read_text()
            ignored = int(next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:")), 16)
            if not ignored & (1 << (signal.SIGTERM - 1)):
                break
            time.sleep(0.01)
        else:
            pytest.fail("CPU worker inherited the trainer SIGTERM disposition")
        process.terminate()
        process.join(timeout=1)
        assert process.exitcode == -signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, previous)
        if shared is not None:
            shared.close()


class _SlowCandidateRows(_Rows):
    def __getitem__(self, index):
        if index == 2:
            time.sleep(5)
        return super().__getitem__(index)

    def media_sample_identity(self, index):
        return {"root": "/fixture", "episode_index": 0, "window_index": index}


class _SlowCandidateProcessor(_Processor):
    def process(self, item):
        if int(item["id"]) == 2:
            time.sleep(5)
        return super().process(item)


class _SlowEpisodeRows(_SlowCandidateRows):
    def get_shuffle_blocks(self):
        return [(0, self.n)]

    def get_item(self, index, *, epoch):
        return self[index]


@pytest.mark.parametrize("slow_read", [False, True, "episode"])
def test_timeout_discards_partial_batch_records_samples_and_resumes(tmp_path, slow_read):
    import json

    def build():
        loader = _build(
            workers=1,
            depth=2,
            logical=(0,),
            batcher=SimpleBatcher(3),
            processor=_Processor() if slow_read else _SlowCandidateProcessor(),
            microbatch_timeout_sec=0.3,
            worker_timeout=10,
            microbatch_skip_log_dir=str(tmp_path),
        )
        if slow_read == "episode":
            from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor

            loader._loaders[0].dataset._distributor = EpisodeBlockDistributor(
                _SlowEpisodeRows(),
                name="source_0",
                seed=42,
            )
        elif slow_read:
            loader._loaders[0].dataset._distributor._dataset = _SlowCandidateRows()
        return loader

    original = build()
    try:
        batch = next(original)
        assert batch["id"].tolist() == [3, 4, 5]
        assert batch[LOADER_META_KEY]["global_id"] == 0
        callback = JointCosmosDataLoaderStateCallback(original)
        callback._update_state_from_batch(batch)
        state = copy.deepcopy(callback.state_dict())
        expected = next(original)
        records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
        assert [row["sample_index"] for row in records] == [0, 1, 2]
        assert len({row["skip_id"] for row in records}) == 1
        assert all(row["global_microbatch_id"] == 0 for row in records)
        if slow_read:
            assert [row["window_index"] for row in records] == [0, 1, 2]
        resumed = build()
        try:
            with patch.dict(os.environ):
                resumed_callback = JointCosmosDataLoaderStateCallback(resumed)
                resumed_callback.load_state_dict(state)
                resumed.preserve_rng_on_resume()
                _assert_batch(next(resumed), expected)
        finally:
            resumed.close()
    finally:
        original.close()


def test_timeout_restart_keeps_old_records_and_creates_distinct_files(tmp_path):
    import json

    for _ in range(2):
        with _build(
            workers=1,
            depth=1,
            logical=(0,),
            batcher=SimpleBatcher(3),
            processor=_SlowCandidateProcessor(),
            microbatch_timeout_sec=0.3,
            worker_timeout=10,
            microbatch_skip_log_dir=str(tmp_path),
        ) as loader:
            assert next(loader)["id"].tolist() == [3, 4, 5]
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 2
    records = [json.loads(path.read_text().splitlines()[0]) for path in files]
    assert records[0]["run_id"] != records[1]["run_id"]
    assert all(len(path.read_text().splitlines()) == 3 for path in files)


def test_timeout_replacements_are_bounded_and_logged(tmp_path):
    import json

    with _build(
        workers=1,
        depth=1,
        logical=(0,),
        processor=_Processor(delay=5),
        microbatch_timeout_sec=0.2,
        microbatch_timeout_max_skips=1,
        worker_timeout=10,
        microbatch_skip_log_dir=str(tmp_path),
    ) as loader:
        with pytest.raises(RuntimeError, match="exceeded 1 timeout replacements"):
            next(loader)
    records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert [row["sample_index"] for row in records] == [0, 1]
    assert [row["attempt"] for row in records] == [1, 2]


class _SlowPrefixProcessor(_Processor):
    def process(self, item):
        if int(item["id"]) < 4:
            time.sleep(5)
        return super().process(item)


def test_unlimited_timeout_replacements_keep_schedule_and_advance_cursor(tmp_path):
    import json

    with _build(
        workers=1,
        depth=1,
        logical=(0,),
        batcher=SimpleBatcher(1),
        processor=_SlowPrefixProcessor(),
        microbatch_timeout_sec=0.2,
        microbatch_timeout_max_skips=0,
        worker_timeout=10,
        microbatch_skip_log_dir=str(tmp_path),
    ) as loader:
        batch = next(loader)
        assert batch["id"].tolist() == [4]
        assert batch[LOADER_META_KEY]["global_id"] == 0
        assert next(loader)["id"].tolist() == [5]
    records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert [row["sample_index"] for row in records] == [0, 1, 2, 3]
    assert [row["attempt"] for row in records] == [1, 2, 3, 4]
    assert len({row["skip_id"] for row in records}) == 4


def test_unlimited_replacements_still_obey_consumer_wait_watchdog(tmp_path):
    with _build(
        workers=1,
        depth=1,
        logical=(0,),
        processor=_Processor(delay=5),
        microbatch_timeout_sec=0.1,
        microbatch_timeout_max_skips=0,
        worker_timeout=1.5,
        microbatch_skip_log_dir=str(tmp_path),
    ) as loader:
        with pytest.raises(TimeoutError, match="timed out after 1.5s waiting"):
            next(loader)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": 0},
        {"worker_timeout": 0.5},
        {"batcher": PoolPackingBatcher(max_tokens=100)},
    ],
)
def test_timeout_requires_interruptible_workers_and_sequential_resumable_streams(kwargs):
    with pytest.raises(ValueError, match="Microbatch skipping requires"):
        _build(microbatch_timeout_sec=0.3, **kwargs)


class _SlowLaterProcessor(_Processor):
    def process(self, item):
        if int(item["id"]) == 3:
            time.sleep(5)
        return super().process(item)


def test_timeout_records_buffered_lookahead_without_recording_delivered_samples(tmp_path):
    import json

    # Each sample costs 6 tokens; a 10-token budget retains the next row as
    # lookahead. Timeout while fetching row 3 must also discard buffered row 2.
    with _build(
        workers=1,
        depth=1,
        logical=(0,),
        batcher=SequentialPackingBatcher(max_sequence_length=10, lookahead_limit=1),
        processor=_SlowLaterProcessor(),
        microbatch_timeout_sec=0.3,
        worker_timeout=10,
        microbatch_skip_log_dir=str(tmp_path),
    ) as loader:
        assert next(loader)["id"].tolist() == [0]
        assert next(loader)["id"].tolist() == [1]
        batch = next(loader)
        assert batch["id"].tolist() == [4]
        assert batch[LOADER_META_KEY]["global_id"] == 2
    records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert [row["sample_index"] for row in records] == [2, 3]
