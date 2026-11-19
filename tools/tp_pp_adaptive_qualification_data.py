# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic tiny packed batches for TP2/FSDP2 adaptive-padding qualification.

Each sample has text, two VIDEO tokens and optionally two ACTION tokens.  These
synthetic inputs do not replace full-capacity or real-dataset qualification.
"""

from dataclasses import asdict, dataclass

import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence


@dataclass(frozen=True)
class Case:
    name: str
    und_lens: tuple[int, ...]
    action: bool = True
    noisy_vision: bool = True
    noisy_action: bool = True

    def __post_init__(self):
        if not self.name or len(self.und_lens) not in (1, 2) or min(self.und_lens) <= 0 or sum(self.und_lens) > 4:
            raise ValueError("A named case must have one or two samples and 1–4 total UND tokens")
        if not self.action and self.noisy_action:
            raise ValueError("Absent ACTION cannot have noisy ACTION targets")


# Every recorded case is warmed in its entirety, including data-dependent
# output/noisy-target paths and sample counts.  Exact-size and just-over-size
# boundaries are both present; the model's first call has a separate full-UND
# 8-token initialization which does not count as a short-bucket warmup.
CASES = (
    Case("short_one", (1,)),
    Case("short_boundary", (2,)),
    Case("long_just_over", (3,)),
    Case("long_boundary_two_samples", (2, 2)),
    Case("short_two_samples_no_action", (1, 1), action=False, noisy_action=False),
    Case("long_no_noisy_targets", (4,), noisy_vision=False, noisy_action=False),
    Case("long_two_samples_no_action", (1, 2), action=False, noisy_action=False),
)


def warmup_cases():
    """One full-UND initialization plus each exact measured case variant."""
    return (CASES[0], *CASES)


def measured_cases():
    return CASES


def qualification_cases():
    """Return stable JSON-serializable case descriptions with deterministic seeds."""
    return [{**asdict(case), "und_lens": list(case.und_lens), "seed": 7300 + index} for index, case in enumerate(CASES)]


def make_qualification_sequence(case):
    """Build the CPU PackedSequence described by qualification_cases()."""
    return make_sequence(
        Case(
            name=case["name"],
            und_lens=tuple(case["und_lens"]),
            action=case["action"],
            noisy_vision=case["noisy_vision"],
            noisy_action=case["noisy_action"],
        ),
        seed=case["seed"],
    )


def make_sequence(case: Case, *, seed: int) -> PackedSequence:
    """Build an independently seeded CPU sequence; caller may invoke to_cuda()."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples, splits, text_indexes, vision_indexes, action_indexes, positions = [], [], [], [], [], []
    video_tokens, action_tokens = [], []
    offset = 0
    for und in case.und_lens:
        total = und + 2 + (2 if case.action else 0)
        samples.append(total)
        splits.extend((und, total - und))
        text_indexes.extend(range(offset, offset + und))
        vision_indexes.extend(range(offset + und, offset + und + 2))
        if case.action:
            action_indexes.extend(range(offset + und + 2, offset + total))
            action_tokens.append(torch.randn((2, 128), generator=generator))
        video_tokens.append(torch.randn((1, 16, 2, 2, 2), generator=generator))
        positions.append(torch.arange(total, dtype=torch.long))
        offset += total
    count = len(samples)
    vision = ModalityData(
        tokens=video_tokens,
        token_shapes=[(2, 1, 1)] * count,
        sequence_indexes=torch.tensor(vision_indexes, dtype=torch.long),
        mse_loss_indexes=torch.tensor(vision_indexes if case.noisy_vision else [], dtype=torch.long),
        timesteps=torch.full((2 * count if case.noisy_vision else 0,), 0.4),
        noisy_frame_indexes=[
            torch.arange(2) if case.noisy_vision else torch.empty(0, dtype=torch.long) for _ in samples
        ],
        condition_mask=[torch.full((2, 1, 1), 0.0 if case.noisy_vision else 1.0) for _ in samples],
        sample_indexes=list(range(count)),
    )
    action = None
    if case.action:
        action = ModalityData(
            tokens=action_tokens,
            token_shapes=[(2,)] * count,
            sequence_indexes=torch.tensor(action_indexes, dtype=torch.long),
            mse_loss_indexes=torch.tensor(action_indexes if case.noisy_action else [], dtype=torch.long),
            domain_id=[torch.tensor([i], dtype=torch.long) for i in range(count)],
            timesteps=torch.full((2 * count if case.noisy_action else 0,), 0.6),
            noisy_frame_indexes=[
                torch.arange(2) if case.noisy_action else torch.empty(0, dtype=torch.long) for _ in samples
            ],
            condition_mask=[torch.full((2, 1), 0.0 if case.noisy_action else 1.0) for _ in samples],
            sample_indexes=list(range(count)),
        )
    return PackedSequence(
        sample_lens=samples,
        split_lens=splits,
        attn_modes=["causal", "full"] * count,
        sequence_length=offset,
        text_ids=torch.tensor([1 + i % 7 for i in range(len(text_indexes))], dtype=torch.long),
        text_indexes=torch.tensor(text_indexes, dtype=torch.long),
        position_ids=torch.cat(positions).expand(3, -1).clone(),
        vision=vision,
        action=action,
    )
