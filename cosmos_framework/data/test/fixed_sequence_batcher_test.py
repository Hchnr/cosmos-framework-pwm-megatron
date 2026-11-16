# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.dataflow.batchers import (
    SequenceTokenCounts,
    SequentialPackingBatcher,
    SequentialTokenBudgetBatcher,
)
from cosmos_framework.data.generator.sequence_packing.sequence import (
    SequencePlan,
    packed_text_span_length,
)
from cosmos_framework.data.generator.sequence_packing.static_layout import (
    StaticPackingConfig,
    build_static_packed_sequence,
)


def _sample(sample_id: int, *, text_tokens: int = 10, frames: int = 1) -> dict:
    return {
        "sample_id": sample_id,
        "text_token_ids": torch.arange(text_tokens),
        "video": torch.zeros(3, frames, 64, 64),
        "sequence_plan": SequencePlan(has_text=True, has_vision=True),
    }


def _batcher(*, und_target: int = 25, gen_target: int = 9, action_target: int = 0) -> SequentialPackingBatcher:
    return SequentialPackingBatcher(
        max_sequence_length=und_target + gen_target,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        lookahead_limit=1,
        fixed_sequence_length_enabled=True,
        fixed_und_tokens=und_target,
        fixed_gen_tokens=gen_target,
        fixed_action_tokens=action_target,
        special_tokens={"eos_token_id": 1, "start_of_generation": 2, "end_of_generation": 3},
    )


def test_fixed_token_counter_matches_unified_mot_split() -> None:
    assert _batcher().sample_token_counts(_sample(0)) == SequenceTokenCounts(und=12, gen=4)


def test_fixed_token_counter_matches_collator_static_layout() -> None:
    sample = {
        **_sample(0),
        "sequence_plan": SequencePlan(has_text=True, has_vision=True),
    }
    counts = _batcher().sample_token_counts(sample)
    packed = build_static_packed_sequence(
        [sample],
        special_tokens={"eos_token_id": 1, "start_of_generation": 2, "end_of_generation": 3},
        config=StaticPackingConfig(
            spatial_compression_factor=16,
            temporal_compression_factor=4,
            latent_patch_size=2,
        ),
    )

    assert counts.und == packed.text_indexes.numel()
    assert counts.gen == packed.sequence_length - packed.text_indexes.numel()


def test_fixed_admission_checks_und_and_gen_independently() -> None:
    samples = [_sample(0), _sample(1), _sample(2)]
    exact_fit = list(_batcher().batches(iter(samples)))
    und_limited = list(_batcher(und_target=24, gen_target=20).batches(iter(samples[:2])))
    gen_limited = list(_batcher(und_target=40, gen_target=8).batches(iter(samples[:2])))

    assert [[sample["sample_id"] for sample in batch] for batch in exact_fit] == [[0, 1], [2]]
    assert [[sample["sample_id"] for sample in batch] for batch in und_limited] == [[0], [1]]
    assert [[sample["sample_id"] for sample in batch] for batch in gen_limited] == [[0], [1]]


def test_fixed_oversized_sample_fails_instead_of_dropping() -> None:
    with pytest.raises(ValueError, match=r"sample=7.*real_und=12.*capacity=11"):
        next(_batcher(und_target=12, gen_target=5).batches(iter([_sample(7)])))


def test_fixed_mode_requires_matching_total_token_budget() -> None:
    with pytest.raises(ValueError, match="must equal the largest configured fixed profile"):
        SequentialPackingBatcher(
            max_sequence_length=100,
            fixed_sequence_length_enabled=True,
            fixed_und_tokens=40,
            fixed_gen_tokens=40,
            special_tokens={"eos_token_id": 1, "start_of_generation": 2},
        )


@pytest.mark.parametrize(
    ("special_tokens", "plan"),
    [
        ({"eos_token_id": 1}, SequencePlan(has_text=True, supervise_text=True)),
        (
            {"bos_token_id": 0, "eos_token_id": 1},
            SequencePlan(has_text=True, supervise_text=True, text_loss_mask=[False, True, True]),
        ),
        ({"bos_token_id": 0, "eos_token_id": 1}, SequencePlan(has_text=True, supervise_text=False)),
    ],
)
def test_understanding_text_estimator_matches_materialized_pack(
    special_tokens: dict[str, int], plan: SequencePlan
) -> None:
    text_ids = torch.tensor([10, 11, 12])
    expected = packed_text_span_length(
        text_ids.numel(),
        special_tokens,
        has_generation=False,
        supervise_text=plan.supervise_text,
        text_loss_mask=plan.text_loss_mask,
    )
    packed = build_static_packed_sequence(
        [{"text_token_ids": text_ids, "sequence_plan": plan}],
        special_tokens=special_tokens,
        config=StaticPackingConfig(),
    )

    assert packed.sequence_length == expected
    assert packed.text_indexes.numel() == expected


def test_fixed_understanding_batcher_is_ordered_and_accepts_exact_capacity() -> None:
    batcher = SequentialTokenBudgetBatcher(
        max_tokens=10,
        size_fn=lambda sample: sample["tokens"],
        physical_target=11,
    )
    samples = [
        {"sample_id": 0, "tokens": 6},
        {"sample_id": 1, "tokens": 4},
        {"sample_id": 2, "tokens": 7},
        {"sample_id": 3, "tokens": 3},
    ]

    batches = list(batcher.batches(iter(samples)))

    assert [[sample["sample_id"] for sample in batch] for batch in batches] == [[0, 1], [2, 3]]


def test_fixed_understanding_batcher_applies_sample_cap_as_secondary_limit() -> None:
    batcher = SequentialTokenBudgetBatcher(
        max_tokens=10,
        size_fn=lambda sample: sample["tokens"],
        max_samples_per_batch=1,
        physical_target=11,
    )

    batches = list(batcher.batches(iter([{"sample_id": 0, "tokens": 2}, {"sample_id": 1, "tokens": 2}])))

    assert [[sample["sample_id"] for sample in batch] for batch in batches] == [[0], [1]]


def test_fixed_understanding_oversized_sample_fails_with_identity_and_counts() -> None:
    batcher = SequentialTokenBudgetBatcher(
        max_tokens=10,
        size_fn=lambda sample: sample["tokens"],
        physical_target=11,
    )

    with pytest.raises(ValueError, match=r"understanding sample 'row-7'.*real_und=11.*target_und=11.*capacity=10"):
        next(batcher.batches(iter([{"catalog_id": "row-7", "tokens": 11}])))


@pytest.mark.parametrize("wrapped", [False, True])
def test_fixed_wam_admission_uses_resized_geometry_and_fills_four_samples(wrapped):
    # DROID concatenated views arrive at320 pixels wide, then model preprocessing
    # resizes to303. Raw pixel admission used to overcount720 rather than648 VIDEO
    # tokens and emit three samples despite a four-sample decoder budget.
    size = torch.tensor([256, 320, 256, 303])
    sample = {
        **_sample(0),
        "mode": "wam",
        "video": torch.empty(3, 33, 256, 320),
        "image_size": [size] if wrapped else size,
        "action": torch.zeros(32, 64),
        "sequence_plan": SequencePlan(has_text=True, has_vision=True, has_action=True),
    }
    batcher = _batcher(und_target=512, gen_target=2816)
    counts = batcher.sample_token_counts(sample)
    packed = build_static_packed_sequence([sample], special_tokens=batcher.special_tokens, config=StaticPackingConfig())
    assert counts.gen == 648 + 32
    assert counts.gen == packed.sequence_length - packed.text_indexes.numel()
    groups = list(batcher.batches(iter([sample] * 8)))
    assert [len(group) for group in groups] == [4, 4]
    # Legacy count-only packing keeps its estimator and stream ordering intact.
    assert batcher.sample_size(sample) == 10 + 1 + 720 + 2 + 32


@pytest.mark.parametrize(("gen_target", "action_target", "actions"), [(15, 8, 1), (20, 3, 2)])
def test_fixed_partition_admission_preserves_order(gen_target, action_target, actions):
    # Both pairs fit aggregate GEN, but exceed VIDEO or ACTION separately.
    batcher = _batcher(und_target=100, gen_target=gen_target, action_target=action_target)
    samples = [{**_sample(i), "action": torch.zeros(actions, 128)} for i in range(3)]
    groups = list(batcher.batches(iter(samples)))
    assert [[s["sample_id"] for s in group] for group in groups] == [[0], [1], [2]]
    counts = batcher.sample_token_counts(samples[0])
    assert counts.video == 4 and counts.action == actions


@pytest.mark.parametrize(("gen_target", "action_target", "actions"), [(12, 8, 1), (12, 3, 3)])
def test_fixed_partition_accepts_exact_stream_capacity(gen_target, action_target, actions):
    batcher = _batcher(und_target=100, gen_target=gen_target, action_target=action_target)
    sample = {**_sample(0), "action": torch.zeros(actions, 128)}
    assert list(batcher.batches(iter([sample]))) == [[sample]]


@pytest.mark.parametrize(("gen_target", "action_target", "actions"), [(12, 9, 1), (12, 2, 3)])
def test_fixed_partition_oversized_single_sample_fails(gen_target, action_target, actions):
    batcher = _batcher(und_target=100, gen_target=gen_target, action_target=action_target)
    with pytest.raises(ValueError, match="real_video=4.*real_action=.*action_capacity="):
        list(batcher.batches(iter([{**_sample(0), "action": torch.zeros(actions, 128)}])))


@pytest.mark.parametrize("action_target", [-1, 9, 10])
def test_fixed_partition_rejects_invalid_capacity(action_target):
    with pytest.raises(ValueError, match="0 < ACTION < GEN"):
        _batcher(gen_target=9, action_target=action_target)
