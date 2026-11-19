# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Topology-independent slot identities, and explicit reference loss weights."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogicalSlot:
    update: int
    micro_step: int
    logical_rank: int


def native_sample_level_loss(config):
    """Use the same condition as native loss before supplying an override."""
    return config.rectified_flow_training_config.sample_level_loss_averaging and config.vision_gen


def logical_slots(config, *, update, data_parallel_size, data_parallel_rank):
    """Round-robin permutation of exactly the reference DP × GA slots."""
    total = config.global_packed_microbatches_per_update
    if total % data_parallel_size or not 0 <= data_parallel_rank < data_parallel_size:
        raise ValueError("Invalid logical batch/DP mapping")
    return tuple(
        LogicalSlot(update, index // config.logical_data_parallel_size, index % config.logical_data_parallel_size)
        for index in range(data_parallel_rank, total, data_parallel_size)
    )


def reference_loss_weight(config, *, local_samples, reference_modality_samples):
    """Compensate schedule 1/M and DP 1/D while retaining old GA denominators.

    ``local_samples`` is the native length of ``preds_vision`` for this slot;
    it is a vision-item count and may differ from the packed sample count.
    ``reference_modality_samples`` is the sum of those counts across original
    logical DP ranks for this slot's original micro step and modality. Native
    decoding uses one dummy prediction when the vision payload is absent.
    The baseline per-rank contribution is n/(R*N) before the update average.
    New schedule/DP average contributes 1/(M*D)=1/(L*R), hence L*n/N.
    Counts may vary or be zero; Action/VIDEO retain the original shared weight.
    """
    if local_samples < 0 or reference_modality_samples < local_samples:
        raise ValueError("Invalid reference modality count")
    return config.logical_data_parallel_size * local_samples / max(1, reference_modality_samples)
