# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.callbacks.mfu import MFUCallback


def test_fixed_sequence_throughput_accumulates_every_microbatch() -> None:
    callback = MFUCallback(every_n=10, hit_thres=0, grad_accum_iter=2, barrier_after_run=False)

    callback.on_training_step_batch_start(None, {}, iteration=0)  # type: ignore[arg-type]
    callback.on_training_step_batch_end(
        None,  # type: ignore[arg-type]
        {},
        {
            "task_type": "understanding",
            "und_token_length": 120,
            "gen_token_length": 0,
            "physical_und_token_length": 160,
            "physical_gen_token_length": 0,
            "batch_size": 3,
        },
        torch.tensor(1.0),
        iteration=0,
    )
    callback.on_training_step_batch_end(
        None,  # type: ignore[arg-type]
        {},
        {
            "task_type": "generation",
            "und_token_length": 100,
            "gen_token_length": 200,
            "physical_und_token_length": 160,
            "physical_gen_token_length": 256,
            "batch_size": 2,
        },
        torch.tensor(1.0),
        iteration=0,
    )

    assert callback._window_start_time is not None
    assert callback._real_und_tokens_in_window == 220
    assert callback._real_gen_tokens_in_window == 200
    assert callback._physical_und_tokens_in_window == 320
    assert callback._physical_gen_tokens_in_window == 256
    assert callback._effective_samples_in_window == 5
    assert callback._microbatches_in_window == 2
    assert callback._task_stats_in_window["understanding"]["physical_gen"] == 0
    assert callback._task_stats_in_window["understanding"]["microbatches"] == 1
    assert callback._task_stats_in_window["generation"]["real_gen"] == 200


def test_fixed_sequence_throughput_skips_warmup_microbatches() -> None:
    callback = MFUCallback(every_n=10, hit_thres=2, barrier_after_run=False)

    callback.on_training_step_batch_start(None, {}, iteration=0)  # type: ignore[arg-type]
    callback.on_training_step_batch_end(
        None,  # type: ignore[arg-type]
        {},
        {"und_token_length": 10, "gen_token_length": 20, "batch_size": 1},
        torch.tensor(1.0),
        iteration=0,
    )

    assert callback._window_start_time is None
    assert callback._real_und_tokens_in_window == 0
    assert callback._real_gen_tokens_in_window == 0
    assert callback._effective_samples_in_window == 0
