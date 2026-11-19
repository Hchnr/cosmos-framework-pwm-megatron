# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU checks for topology-independent slots, objective weights and payload."""

import pytest
import torch

from cosmos_framework.model.generator.megatron_tp_pp.payload import STREAMS, PayloadLayout
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots, reference_loss_weight
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig, layer_range


@pytest.mark.parametrize("tp,pp,dp", [(1, 1, 4), (2, 1, 2), (1, 2, 2), (2, 2, 1)])
def test_all_576_reference_slots_once(tp, pp, dp):
    config = ModelParallelConfig(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    assert config.data_parallel_size(4) == dp
    assert config.microbatches(4) == 576 // dp
    slots = [
        s for r in range(dp) for s in logical_slots(config, update=1200, data_parallel_size=dp, data_parallel_rank=r)
    ]
    assert len(slots) == len(set(slots)) == 576
    assert {(s.micro_step, s.logical_rank) for s in slots} == {(m, r) for m in range(3) for r in range(192)}


@pytest.mark.parametrize("dp", [1, 2, 4])
def test_nonuniform_sample_counts_keep_original_objective(dp):
    config = ModelParallelConfig(global_packed_microbatches_per_update=12, logical_data_parallel_size=4)
    counts = [[1, 2, 0, 3], [2, 1, 1, 2], [0, 1, 3, 1]]
    modalities = [[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
    means = torch.arange(12, dtype=torch.float64).view(3, 4)
    totals = [
        [sum(c for c, k in zip(cs, ks, strict=True) if k == kind) for kind in (0, 1)]
        for cs, ks in zip(counts, modalities, strict=True)
    ]
    expected = (
        sum(means[m, r] * counts[m][r] / max(totals[m][modalities[m][r]], 1) for m in range(3) for r in range(4)) / 3
    )
    actual = 0
    for rank in range(dp):
        for slot in logical_slots(config, update=0, data_parallel_size=dp, data_parallel_rank=rank):
            m, r = slot.micro_step, slot.logical_rank
            actual += (
                means[m, r]
                * reference_loss_weight(
                    config,
                    local_samples=counts[m][r],
                    reference_modality_samples=totals[m][modalities[m][r]],
                )
                / (12 // dp)
                / dp
            )
    torch.testing.assert_close(actual, expected)


def test_unequal_width_payload_preserves_gradients_without_padding():
    layout = PayloadLayout(((3, 8), (7, 8), (2, 3)))
    streams = {k: torch.randn(s, requires_grad=True) for k, s in zip(STREAMS, layout.shapes, strict=True)}
    tensor = layout.pack(streams)
    assert tensor.numel() == 89  # 86 activation values + three backward usage flags
    unpacked = layout.unpack(tensor)
    sum((i + 1) * t.sum() for i, t in enumerate(unpacked.values())).backward()
    for i, t in enumerate(streams.values()):
        torch.testing.assert_close(t.grad, torch.full_like(t, i + 1))


@pytest.mark.parametrize("layers,pp", [(36, 2), (36, 4), (5, 3)])
def test_stage_ranges_cover_depth_once(layers, pp):
    assert [i for rank in range(pp) for i in layer_range(layers, pp, rank)] == list(range(layers))


def test_invalid_topology_and_loss_counts_fail():
    with pytest.raises(ValueError):
        ModelParallelConfig(sequence_parallel=True)
    with pytest.raises(ValueError):
        ModelParallelConfig().microbatches(6)
    with pytest.raises(ValueError):
        reference_loss_weight(ModelParallelConfig(), local_samples=2, reference_modality_samples=1)
