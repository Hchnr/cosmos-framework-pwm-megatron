# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, SequenceTokenCounts


def _sample(sample_id: int, *, text_tokens: int = 10, frames: int = 1) -> dict:
    return {
        "sample_id": sample_id,
        "text_token_ids": [torch.arange(text_tokens)],
        "video": [torch.zeros(3, frames, 64, 64)],
    }


def _fixed_loader(samples: list[dict], *, und_target: int = 25, gen_target: int = 9) -> PackingDataLoader:
    """Build a packing-loop fixture without spawning DataLoader workers."""

    loader = object.__new__(PackingDataLoader)
    loader.dataloader_list = [SimpleNamespace(dataset_name="fixed-test")]
    loader.dataset_name_list = ["fixed-test"]
    loader.dataloaders = [iter(())]
    loader.buffers = [deque(samples)]
    loader.lookahead_limits = [10]
    loader.global_id = 0
    loader.data_len = len(samples)
    loader.max_sequence_length = und_target + gen_target
    loader.max_samples_per_batch = None
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.patch_spatial = 2
    loader.sound_latent_fps = 0
    loader.audio_sample_rate = 48000
    loader.uniae_chunk_frames = None
    loader.uniae_pad_frames = None
    loader.fixed_sequence_length_enabled = True
    loader.fixed_und_tokens = und_target
    loader.fixed_gen_tokens = gen_target
    loader._drop_sample_log_count = 0
    return loader


def test_fixed_token_counter_matches_unified_mot_und_gen_split() -> None:
    loader = _fixed_loader([])

    # UND = 10 caption + EOS + BOV. GEN = 1x2x2 patchified latent grid.
    assert loader._compute_fixed_sequence_token_counts(_sample(0)) == SequenceTokenCounts(und=12, gen=4)


def test_fixed_two_dimensional_admission_accepts_capacity_exact_fit() -> None:
    loader = _fixed_loader([_sample(0), _sample(1), _sample(2)])

    batches = list(loader)

    # Physical targets reserve one token per stream for the isolated padding
    # segment, so 2 * (UND=12, GEN=4) exactly fills real capacity (24, 8).
    assert [batch["sample_id"] for batch in batches] == [[0, 1], [2]]
    assert batches[0]["_estimated_real_und_tokens"] == 24
    assert batches[0]["_estimated_real_gen_tokens"] == 8
    assert batches[0]["_physical_und_tokens"] == 25
    assert batches[0]["_physical_gen_tokens"] == 9
    assert batches[0]["_num_samples"] == 2
    assert batches[1]["_num_samples"] == 1
    assert [sample_id for batch in batches for sample_id in batch["sample_id"]] == [0, 1, 2]


def test_fixed_admission_checks_und_and_gen_independently() -> None:
    und_limited = list(_fixed_loader([_sample(0), _sample(1)], und_target=24, gen_target=20))
    gen_limited = list(_fixed_loader([_sample(0), _sample(1)], und_target=40, gen_target=8))

    assert [batch["sample_id"] for batch in und_limited] == [[0], [1]]
    assert [batch["sample_id"] for batch in gen_limited] == [[0], [1]]


def test_fixed_oversized_sample_fails_instead_of_being_dropped() -> None:
    loader = _fixed_loader([_sample(7)], und_target=12, gen_target=5)

    with pytest.raises(ValueError, match=r"sample=7.*real_und=12 \(capacity=11\)"):
        next(iter(loader))


def test_fixed_counter_rejects_missing_text_or_media() -> None:
    loader = _fixed_loader([])
    with pytest.raises(ValueError, match="requires text_token_ids"):
        loader._compute_fixed_sequence_token_counts({"video": [torch.zeros(3, 1, 64, 64)]})
    with pytest.raises(ValueError, match="requires images or video"):
        loader._compute_fixed_sequence_token_counts({"text_token_ids": [torch.arange(4)]})
