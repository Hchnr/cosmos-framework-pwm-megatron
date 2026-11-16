# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DCP save coordination and asynchronous staging tests."""

import multiprocessing
import os
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

import cosmos_framework.checkpoint.dcp as dcp_module
from cosmos_framework.checkpoint.dcp import AsyncMode, DistributedCheckpointer, SaveDone, Terminate


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(1)
def test_async_staging_waits_for_pending_optimizer_writes() -> None:
    checkpointer = object.__new__(DistributedCheckpointer)
    checkpointer.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
    checkpointer.staging_stream = torch.cuda.Stream()
    checkpointer.mp_queue_send = MagicMock()
    weight = torch.zeros(4096, device="cuda")
    cpu_weight = torch.empty(4096, pin_memory=True)
    checkpointer.cpu_offload_state_dict = {"weight": cpu_weight}
    torch.cuda.synchronize()

    # Reproduce a later save with reused buffers: no first-save allocation or
    # blocking CPU transfer can accidentally synchronize the optimizer stream.
    torch.cuda._sleep(1_000_000_000)
    weight.fill_(7)
    try:
        checkpointer._checkpoint_async_with_pinned_memory("iter_000000002", {"weight": weight})
    finally:
        torch.cuda.synchronize()
    torch.testing.assert_close(cpu_weight, torch.full_like(cpu_weight, 7))
    assert checkpointer.checkpoint_in_progress
    checkpointer.mp_queue_send.put_nowait.assert_called_once()


def _empty_checkpointer() -> DistributedCheckpointer:
    checkpointer = object.__new__(DistributedCheckpointer)
    checkpointer._dcp_save_process_group = None
    checkpointer._dcp_save_process_group_initialized = False
    return checkpointer


def _async_checkpointer(process, *, pending=True):
    cp = _empty_checkpointer()
    cp.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
    cp.save_thread = None
    cp.mp = process
    cp.mp_queue_send = queue.Queue()
    cp.mp_queue_recv = queue.Queue()
    cp.checkpoint_in_progress = pending
    cp.callbacks = MagicMock()
    return cp


@pytest.mark.parametrize("alive", [False, True])
@pytest.mark.parametrize("succeeded", [False, True])
def test_async_finalize_checks_pending_ack_even_after_worker_exit(alive, succeeded):
    process = MagicMock()
    process.is_alive.return_value = alive
    cp = _async_checkpointer(process)
    cp.mp_queue_recv.put(SaveDone(iteration=12, elapsed_time=0.5, succeeded=succeeded))

    if succeeded:
        cp.finalize()
        assert not cp.checkpoint_in_progress
        cp.callbacks.on_save_checkpoint_success.assert_called_once_with(iteration=12, elapsed_time=0.5)
    else:
        with pytest.raises(RuntimeError, match="checkpoint save failed"):
            cp.finalize()
        cp.callbacks.on_save_checkpoint_success.assert_not_called()

    if alive and succeeded:
        assert isinstance(cp.mp_queue_send.get_nowait(), Terminate)
        process.join.assert_called_once()
    elif alive:
        process.terminate.assert_called_once()
        process.join.assert_called_once()


def _failed_save_worker():
    os._exit(17)


def test_async_finalize_rejects_real_worker_exit_without_ack():
    # The child only exits; it never uses inherited CUDA state.
    process = multiprocessing.get_context("fork").Process(target=_failed_save_worker)
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 17
    cp = _async_checkpointer(process)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="checkpoint save failed"):
        cp.finalize()
    assert time.monotonic() - started < 5
    cp.callbacks.on_save_checkpoint_success.assert_not_called()


def test_async_finalize_without_pending_save_stops_worker():
    process = MagicMock()
    process.is_alive.return_value = True
    cp = _async_checkpointer(process, pending=False)
    cp.finalize()
    assert isinstance(cp.mp_queue_send.get_nowait(), Terminate)
    process.join.assert_called_once()
    cp.callbacks.on_save_checkpoint_success.assert_not_called()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("contents", ["absent", "directory", "complete"])
def test_dataloader_resume_enforces_strict_state_only_when_bound(tmp_path, monkeypatch, strict, stateful, contents):
    from cosmos_framework.checkpoint.base import CheckpointLoadSource
    from cosmos_framework.utils.config import CheckpointConfig, ObjectStoreConfig

    cp = _empty_checkpointer()
    cp.config_checkpoint = CheckpointConfig(strict_resume=strict)
    handler = MagicMock()
    handler.checkpoint_component = "dataloader"
    handler.has_checkpoint_state.return_value = stateful
    cp.callbacks = MagicMock()
    cp.callbacks._callbacks = [handler]
    source = CheckpointLoadSource(str(tmp_path), None, ObjectStoreConfig(), warm_start=False)
    cp.keys_to_resume_during_load = lambda: (["dataloader"], source)
    cp._check_checkpoint_exists = lambda source: None
    monkeypatch.setattr(dcp_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(dcp_module.dist, "get_rank", lambda: 0)
    model = SimpleNamespace(config=SimpleNamespace(action128_spec=None))
    data_dir = tmp_path / "dataloader"
    if contents != "absent":
        data_dir.mkdir()
    if contents == "complete":
        dcp_module.easy_io.dump({"global_id": 27, "cursor": 91}, str(data_dir / "rank_0.pkl"), file_format="pkl")

    if strict and stateful and contents != "complete":
        with pytest.raises(FileNotFoundError, match="[Ss]trict.*dataloader"):
            cp.load(model)
        handler.load_state_dict.assert_not_called()
    else:
        assert cp.load(model) == 0
        loaded = stateful and contents == "complete"
        assert cp.loaded_dataloader_state is loaded
        assert cp.loaded_dataloader_global_id == (27 if loaded else None)
        if loaded:
            handler.load_state_dict.assert_called_once_with({"global_id": 27, "cursor": 91})
        else:
            handler.load_state_dict.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_save_process_group_is_lazily_created_with_gloo_and_reused() -> None:
    checkpointer = _empty_checkpointer()
    process_group = MagicMock(spec=dist.ProcessGroup)

    with (
        patch.object(dcp_module.dist, "is_available", return_value=True),
        patch.object(dcp_module.dist, "is_initialized", return_value=True),
        patch.object(dcp_module.dist, "get_backend", return_value=dist.Backend.NCCL),
        patch.object(dcp_module.dist, "new_group", return_value=process_group) as new_group,
    ):
        assert checkpointer._get_dcp_save_process_group() is process_group
        assert checkpointer._get_dcp_save_process_group() is process_group

    new_group.assert_called_once_with(
        backend=dist.Backend.GLOO,
        group_desc="dcp_save_metadata",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_save_reuses_default_gloo_group() -> None:
    checkpointer = _empty_checkpointer()

    with (
        patch.object(dcp_module.dist, "is_available", return_value=True),
        patch.object(dcp_module.dist, "is_initialized", return_value=True),
        patch.object(dcp_module.dist, "get_backend", return_value=dist.Backend.GLOO),
        patch.object(dcp_module.dist, "new_group") as new_group,
    ):
        assert checkpointer._get_dcp_save_process_group() is None
        assert checkpointer._get_dcp_save_process_group() is None

    new_group.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_save_state_dict_worker_passes_gloo_group_to_dcp_save() -> None:
    checkpointer = _empty_checkpointer()
    checkpointer.save_to_object_store = False
    checkpointer._local_dirname = "/tmp/checkpoints"
    checkpointer.get_storage_writer = MagicMock(return_value=MagicMock())
    process_group = MagicMock(spec=dist.ProcessGroup)
    checkpointer._get_dcp_save_process_group = MagicMock(return_value=process_group)

    with (
        patch.object(dcp_module.dcp, "save") as dcp_save,
        patch.object(dcp_module.distributed, "is_rank0", return_value=False),
    ):
        checkpointer.save_state_dict_worker(
            {"model": ({"weight": 1}, "/tmp/checkpoints/iter_000000001/model")},
            "iter_000000001",
        )

    assert dcp_save.call_args.kwargs["process_group"] is process_group


@pytest.mark.L0
@pytest.mark.CPU
def test_finalize_destroys_owned_dcp_save_process_group() -> None:
    checkpointer = _empty_checkpointer()
    process_group = MagicMock(spec=dist.ProcessGroup)
    checkpointer._dcp_save_process_group = process_group
    checkpointer._dcp_save_process_group_initialized = True
    checkpointer.async_mode = AsyncMode.DISABLED
    checkpointer.save_thread = None

    with (
        patch.object(dcp_module.dist, "is_available", return_value=True),
        patch.object(dcp_module.dist, "is_initialized", return_value=True),
        patch.object(dcp_module.dist, "destroy_process_group") as destroy_process_group,
    ):
        checkpointer.finalize()

    destroy_process_group.assert_called_once_with(process_group)
    assert checkpointer._dcp_save_process_group is None
    assert not checkpointer._dcp_save_process_group_initialized


def _retention_checkpointer(tmp_path, *, keep_last_n=3, keep_every_n_steps=None):
    from cosmos_framework.utils.config import CheckpointConfig

    cp = _empty_checkpointer()
    cp.config_checkpoint = CheckpointConfig(
        keep_last_n=keep_last_n, keep_every_n_steps=keep_every_n_steps, save_iter=200
    )
    cp.save_to_object_store = False
    cp._local_dirname = str(tmp_path)
    cp._get_dcp_save_process_group = MagicMock(return_value=None)
    cp.get_storage_writer = MagicMock()
    cp._write_latest_checkpoint_file = lambda name: (tmp_path / "latest_checkpoint.txt").write_text(name)
    return cp


def _complete_checkpoint(root, step):
    path = root / f"iter_{step:09d}"
    for part in ("model", "optim", "scheduler", "trainer"):
        (path / part).mkdir(parents=True)
        (path / part / ".metadata").touch()
    return path


@pytest.mark.parametrize("interval", [0, -2000, True, 2000.0, "2000"])
def test_periodic_retention_rejects_invalid_runtime_intervals(interval):
    from cosmos_framework.utils.config import CheckpointConfig

    with (
        patch.object(dcp_module.AbstractCheckpointer, "__init__", return_value=None),
        pytest.raises(ValueError, match="keep_every_n_steps must be a positive integer"),
    ):
        DistributedCheckpointer(CheckpointConfig(keep_every_n_steps=interval), MagicMock())


@pytest.mark.parametrize("save_iter", [0, 300, 3000])
def test_periodic_retention_requires_save_interval_alignment(save_iter):
    from cosmos_framework.utils.config import CheckpointConfig

    with (
        patch.object(dcp_module.AbstractCheckpointer, "__init__", return_value=None),
        patch.object(DistributedCheckpointer, "save_to_object_store", False, create=True),
        pytest.raises(ValueError, match="keep_every_n_steps must be a multiple of save_iter"),
    ):
        DistributedCheckpointer(CheckpointConfig(keep_every_n_steps=2000, save_iter=save_iter), MagicMock())


def test_periodic_retention_rejects_object_store_saves():
    from cosmos_framework.utils.config import CheckpointConfig

    with (
        patch.object(dcp_module.AbstractCheckpointer, "__init__", return_value=None),
        patch.object(DistributedCheckpointer, "save_to_object_store", True, create=True),
        pytest.raises(ValueError, match="keep_every_n_steps currently supports local/shared filesystem"),
    ):
        DistributedCheckpointer(CheckpointConfig(keep_every_n_steps=2000, save_iter=200), MagicMock())


@pytest.mark.parametrize("completion", ["metadata", "marker"])
def test_periodic_retention_keeps_milestones_plus_newest_three_across_resume(tmp_path, completion):
    cp = _retention_checkpointer(tmp_path, keep_every_n_steps=2000)
    _complete_checkpoint(tmp_path, 0)
    for step in range(200, 6601, 200):
        if step == 4200:
            # Recreate the checkpointer: protection must survive a same-job restart.
            cp = _retention_checkpointer(tmp_path, keep_every_n_steps=2000)
        name = f"iter_{step:09d}"
        if completion == "marker":
            (tmp_path / name).mkdir()
            (tmp_path / name / ".save_complete").touch()
        else:
            _complete_checkpoint(tmp_path, step)
        cp._prune_old_checkpoints(name)
        remaining = {int(path.name[5:]) for path in tmp_path.glob("iter_*")}
        latest_three = set(range(max(0, step - 400), step + 1, 200))
        milestones = set(range(2000, step + 1, 2000))
        assert remaining == latest_three | milestones


def test_periodic_retention_without_keep_last_n_keeps_all(tmp_path):
    cp = _retention_checkpointer(tmp_path, keep_last_n=None, keep_every_n_steps=2000)
    steps = (2000, 2200, 2400, 2600, 2800)
    for step in steps:
        _complete_checkpoint(tmp_path, step)
    cp._prune_old_checkpoints("iter_000002800")
    assert {int(path.name[5:]) for path in tmp_path.glob("iter_*")} == set(steps)


def test_retention_keeps_three_and_stays_inside_job(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    cp = _retention_checkpointer(root)
    for step in (1, 2, 3, 4):
        _complete_checkpoint(root, step)
    future = _complete_checkpoint(root, 9)
    incomplete = root / "iter_000000000"
    incomplete.mkdir()
    outside = _complete_checkpoint(tmp_path, 0)
    (root / "iter_000000005").symlink_to(outside, target_is_directory=True)
    cp._prune_old_checkpoints("iter_000000004")
    assert not (root / "iter_000000001").exists()
    assert all((root / f"iter_{step:09d}").is_dir() for step in (2, 3, 4))
    assert future.exists() and incomplete.exists() and outside.exists()
    assert (root / "iter_000000005").is_symlink()


@pytest.mark.parametrize("failure", [None, "save", "barrier", "latest"])
@pytest.mark.parametrize("keep_every_n_steps", [None, 2000])
def test_prune_only_after_save_barrier_and_latest_succeed(tmp_path, failure, keep_every_n_steps):
    cp = _retention_checkpointer(tmp_path, keep_every_n_steps=keep_every_n_steps)
    for step in (2000, 2200, 2400, 2600, 2800):
        _complete_checkpoint(tmp_path, step)
    events = []

    def stage(name):
        def call(*args, **kwargs):
            events.append(name)
            assert (tmp_path / "iter_000002200").exists()
            if failure == name:
                raise RuntimeError(name)

        return call

    cp._write_latest_checkpoint_file = stage("latest")
    with (
        patch.object(dcp_module.dcp, "save", side_effect=stage("save")),
        patch.object(dcp_module.dist, "is_initialized", return_value=True),
        patch.object(dcp_module.dist, "barrier", side_effect=stage("barrier")),
        patch.object(dcp_module.distributed, "is_rank0", return_value=True),
    ):
        payload = {"model": ({}, str(tmp_path / "iter_000002800/model"))}
        if failure:
            with pytest.raises(RuntimeError, match=failure):
                cp.save_state_dict_worker(payload, "iter_000002800")
            assert (tmp_path / "iter_000002200").exists()
            assert (tmp_path / "iter_000002000").exists()
        else:
            cp.save_state_dict_worker(payload, "iter_000002800")
            assert events == ["save", "barrier", "latest"]
            assert not (tmp_path / "iter_000002200").exists()
            assert (tmp_path / "iter_000002000").exists() is (keep_every_n_steps is not None)
            assert (tmp_path / "iter_000002800/.save_complete").is_file()


@pytest.mark.parametrize("succeeds", [True, False])
@pytest.mark.parametrize("keep_every_n_steps", [None, 2000])
def test_async_background_prunes_after_actual_save(tmp_path, succeeds, keep_every_n_steps):
    cp = _retention_checkpointer(tmp_path, keep_every_n_steps=keep_every_n_steps)
    for step in (2000, 2200, 2400, 2600, 2800):
        _complete_checkpoint(tmp_path, step)
    receiver, sender = MagicMock(), MagicMock()
    receiver.get.side_effect = [
        ({"trainer": ({"iteration": 2800}, str(tmp_path / "iter_000002800/trainer"))}, "iter_000002800"),
        dcp_module.Terminate(),
    ]

    def save(*args, **kwargs):
        assert (tmp_path / "iter_000002200").exists()
        if not succeeds:
            raise RuntimeError("background save failed")

    def completed(result):
        assert result.succeeded is succeeds
        assert (tmp_path / "iter_000002200").exists() is not succeeds
        assert (tmp_path / "iter_000002000").exists() is (not succeeds or keep_every_n_steps is not None)

    sender.put.side_effect = completed
    with (
        patch.dict("os.environ", {"MASTER_PORT": "29000", "LOCAL_RANK": "0"}),
        patch.object(dcp_module.torch.cuda, "set_device"),
        patch.object(dcp_module.dist, "is_initialized", return_value=False),
        patch.object(dcp_module.dist, "init_process_group"),
        patch.object(dcp_module.dist, "destroy_process_group"),
        patch.object(dcp_module.distributed, "is_rank0", return_value=True),
        patch.object(dcp_module, "DistributedCheckpointer", return_value=cp),
        patch.object(dcp_module.dcp, "save", side_effect=save),
    ):
        dcp_module.save_checkpoint_in_background(receiver, sender, cp.config_checkpoint, MagicMock())
    sender.put.assert_called_once()
    assert (tmp_path / "iter_000002200").exists() is not succeeds
    assert (tmp_path / "iter_000002000").exists() is (not succeeds or keep_every_n_steps is not None)
