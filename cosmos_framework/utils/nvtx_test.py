# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from cosmos_framework.utils.callback import NVTXCallback
from cosmos_framework.utils.nvtx import (
    NVTXGranularity,
    configure_nvtx,
    disable_nvtx,
    nvtx_enabled,
    nvtx_mark,
    nvtx_range,
)


@pytest.fixture(autouse=True)
def _reset_nvtx() -> None:
    disable_nvtx()
    yield
    disable_nvtx()


def test_nvtx_granularity_is_cumulative_and_validated() -> None:
    configure_nvtx("module")
    assert nvtx_enabled("phase")
    assert nvtx_enabled(NVTXGranularity.MODULE)
    assert not nvtx_enabled("operator")

    with pytest.raises(ValueError, match="Unknown NVTX granularity"):
        configure_nvtx("kernel")


def test_nvtx_range_and_mark_are_noops_below_required_level() -> None:
    with (
        mock.patch.object(torch.cuda.nvtx, "range_push") as range_push,
        mock.patch.object(torch.cuda.nvtx, "range_pop") as range_pop,
        mock.patch.object(torch.cuda.nvtx, "mark") as mark,
    ):
        configure_nvtx("phase")
        with nvtx_range("forward.network"):
            pass
        with nvtx_range("forward.network.decoder.layer_00", "module"):
            pass
        nvtx_mark("meta.sample")

    range_push.assert_called_once_with("cosmos.forward.network")
    range_pop.assert_called_once_with()
    mark.assert_called_once_with("cosmos.meta.sample")


def _callback_config(*, torch_profiler: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        trainer=SimpleNamespace(
            grad_accum_iter=2,
            profiling=SimpleNamespace(
                enable_profiling=torch_profiler,
                enable_nsys=False,
                profile_freq=10,
                profile_active=3,
            ),
        )
    )


def test_nvtx_callback_labels_each_micro_step_and_its_metadata() -> None:
    callback = NVTXCallback(granularity="phase", detail_target_ranks=[0])
    callback.config = _callback_config()
    data = {"video": [torch.empty(3, 9, 16, 16)]}
    output = {
        "is_image_batch": False,
        "batch_size": 1,
        "und_token_length": 11,
        "gen_token_length": 13,
        "physical_und_token_length": 16,
        "physical_gen_token_length": 16,
        "vae_pixel_shapes": [(9, 16, 16)],
        "split_lens": [24],
    }

    with (
        mock.patch("cosmos_framework.utils.callback.distributed.get_rank", return_value=0),
        mock.patch.object(torch.cuda.nvtx, "range_push") as range_push,
        mock.patch.object(torch.cuda.nvtx, "range_pop") as range_pop,
        mock.patch.object(torch.cuda.nvtx, "mark") as mark,
    ):
        callback.on_train_start(mock.Mock(), iteration=7)
        callback.on_training_step_batch_start(mock.Mock(), data, iteration=7)
        callback.on_before_forward(iteration=7)
        callback.on_after_forward(iteration=7)
        callback.on_before_backward(mock.Mock(), torch.tensor(1.0), iteration=7)
        callback.on_after_backward(mock.Mock(), iteration=7)
        callback.on_training_step_batch_end(mock.Mock(), data, output, torch.tensor(1.0), iteration=7)

        callback.on_training_step_batch_start(mock.Mock(), data, iteration=7)
        callback.on_training_step_batch_end(mock.Mock(), data, output, torch.tensor(1.0), iteration=7)
        callback.on_train_end(mock.Mock(), iteration=8)

    pushed_names = [call.args[0] for call in range_push.call_args_list]
    assert "cosmos.train.rank_000.step_000008.micro_01_of_02" in pushed_names
    assert "cosmos.train.rank_000.step_000008.micro_01_of_02.forward" in pushed_names
    assert "cosmos.train.rank_000.step_000008.micro_01_of_02.backward" in pushed_names
    assert "cosmos.train.rank_000.step_000008.micro_02_of_02" in pushed_names
    assert range_pop.call_count == len(pushed_names)
    marked_names = [call.args[0] for call in mark.call_args_list]
    assert any("meta.input.train.rank_000.step_000008.micro_01_of_02" in name for name in marked_names)
    assert any("vae_shapes=[(9,16,16)]" in name for name in marked_names)


def test_optimizer_range_does_not_pop_nested_trainer_ranges() -> None:
    callback = NVTXCallback(granularity="phase", detail_target_ranks=[0])
    callback.config = _callback_config()
    stack: list[str] = []
    popped_names: list[str] = []

    def push(name: str) -> None:
        stack.append(name)

    def pop() -> None:
        popped_names.append(stack.pop())

    with (
        mock.patch("cosmos_framework.utils.callback.distributed.get_rank", return_value=0),
        mock.patch.object(torch.cuda.nvtx, "range_push", side_effect=push) as range_push,
        mock.patch.object(torch.cuda.nvtx, "range_pop", side_effect=pop) as range_pop,
        mock.patch.object(torch.cuda.nvtx, "range_start", side_effect=[101, 202]) as range_start,
        mock.patch.object(torch.cuda.nvtx, "range_end") as range_end,
    ):
        callback.on_train_start(mock.Mock(), iteration=7)
        with nvtx_range("optimizer.callbacks_before_step"):
            callback.on_before_optimizer_step(mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), iteration=7)
        with nvtx_range("optimizer.update"):
            pass
        with nvtx_range("optimizer.callbacks_before_zero_grad"):
            callback.on_before_zero_grad(mock.Mock(), mock.Mock(), mock.Mock(), iteration=7)
        callback.on_train_end(mock.Mock(), iteration=8)

    assert [call.args[0] for call in range_push.call_args_list] == [
        "cosmos.optimizer.callbacks_before_step",
        "cosmos.optimizer.update",
        "cosmos.optimizer.callbacks_before_zero_grad",
    ]
    assert range_pop.call_count == 3
    assert popped_names == [
        "cosmos.optimizer.callbacks_before_step",
        "cosmos.optimizer.update",
        "cosmos.optimizer.callbacks_before_zero_grad",
    ]
    assert stack == []
    assert [call.args[0] for call in range_start.call_args_list] == [
        "optimizer_step",
        "cosmos.train.rank_000.step_000008.micro_01_of_02.optimizer",
    ]
    assert range_end.call_args_list == [mock.call(202), mock.call(101)]


def test_nvtx_cleanup_closes_an_unfinished_optimizer_range() -> None:
    callback = NVTXCallback(granularity="coarse")
    callback.config = _callback_config()

    with (
        mock.patch("cosmos_framework.utils.callback.distributed.get_rank", return_value=0),
        mock.patch.object(torch.cuda.nvtx, "range_start", return_value=303),
        mock.patch.object(torch.cuda.nvtx, "range_end") as range_end,
    ):
        callback.on_train_start(mock.Mock(), iteration=7)
        callback.on_before_optimizer_step(mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), iteration=7)
        callback.on_train_end(mock.Mock(), iteration=8)

    range_end.assert_called_once_with(303)
    assert callback._optimizer_range_ids == []


def test_operator_ranges_can_be_limited_to_selected_ranks() -> None:
    callback = NVTXCallback(granularity="operator", detail_target_ranks=[0], record_shapes=True)
    callback.config = _callback_config()

    with (
        mock.patch("cosmos_framework.utils.callback.distributed.get_rank", return_value=1),
        mock.patch("torch.autograd.profiler.emit_nvtx") as emit_nvtx,
    ):
        callback.on_train_start(mock.Mock())
        callback.on_train_end(mock.Mock())

    emit_nvtx.assert_not_called()


def test_operator_profiler_is_delayed_until_nsys_active_window() -> None:
    callback = NVTXCallback(granularity="operator", record_shapes=True)
    callback.config = _callback_config()
    callback.config.trainer.profiling.enable_nsys = True

    profiler = mock.MagicMock()
    with (
        mock.patch("cosmos_framework.utils.callback.distributed.get_rank", return_value=0),
        mock.patch("torch.autograd.profiler.emit_nvtx", return_value=profiler) as emit_nvtx,
        mock.patch.object(torch.cuda.nvtx, "range_push"),
        mock.patch.object(torch.cuda.nvtx, "mark"),
    ):
        callback.on_train_start(mock.Mock(), iteration=0)
        emit_nvtx.assert_not_called()
        callback.on_training_step_batch_start(mock.Mock(), {}, iteration=6)
        emit_nvtx.assert_not_called()
        callback._micro_range_open = False
        callback.on_training_step_batch_start(mock.Mock(), {}, iteration=7)
        emit_nvtx.assert_called_once_with(record_shapes=True)
        profiler.__enter__.assert_called_once_with()
        callback.on_train_end(mock.Mock())
        profiler.__exit__.assert_called_once_with(None, None, None)
