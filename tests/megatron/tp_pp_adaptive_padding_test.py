# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import (
    action_rope,
    small_decoder_inputs,
    small_layer_forward,
)
from tools.run_tp_pp_adaptive_padding import adaptive_capacities, bucket_for, install, parse_buckets


def test_capacity_selection_preserves_original_limit_and_full_warmup():
    pack = {"_num_causal_tokens": 1807, "_action_indices": range(549)}
    und, action = (2048, 4096, 5120, 8192, 11776), (768, 1536, 2944)
    selected, actual = adaptive_capacities(pack, (11776, 59648, 2944), 5120, und, action)
    assert selected == (2048, 59648, 768)
    assert actual == (1807, 549)
    selected, _ = adaptive_capacities(pack, (11776, 59648, 2944), 0, und, action)
    assert selected == (11776, 59648, 768)
    assert bucket_for(11775, 11776, und) == 11776
    with pytest.raises(ValueError, match="fit"):
        bucket_for(11777, 11776, und)
    for invalid in ("", "0,1", "2,2", "4,2", "a,b"):
        with pytest.raises(ValueError):
            parse_buckets(invalid)


def _valid_mask(pack, query_stream, include_action):
    ids = pack["_small_ids"]
    mask = pack["_small_masks"][(include_action, query_stream)]
    query_indices = torch.nonzero(ids[query_stream] >= 0).flatten()
    key_ids = ids[0] if query_stream == 0 else torch.cat(ids if include_action else ids[:2])
    key_indices = torch.nonzero(key_ids >= 0).flatten()
    return mask.index_select(0, query_indices).index_select(1, key_indices)


@pytest.mark.parametrize("action_active", [False, True])
def test_flash_video_metadata_preserves_real_sample_visibility(action_active):
    # Exercise FA3's sample grouping separately from CUDA kernels.  UND and
    # ACTION bucket sizes vary, but neither may change real-token visibility.
    from cosmos_framework.model.generator.mot.small_action_attention import video_varlen_mask

    query = torch.tensor([0, 0, 1, 1, -1, -1])
    video_keys = query
    for und in (torch.tensor([0, 1, -1, -1]), torch.tensor([0, 1])):
        for action in (torch.tensor([0, 1, -1, -1]), torch.tensor([0, 1])):
            keys = torch.cat((und, video_keys, action)) if action_active else torch.cat((und, video_keys))
            metadata = video_varlen_mask(query, keys, 2, fixed_capacity=True)
            order = metadata.kv_order
            visibility = torch.zeros((len(query), len(keys)), dtype=torch.bool)
            for seq in range(2):
                q0, q1 = metadata.cumulative_q[seq : seq + 2].tolist()
                k0, k1 = metadata.cumulative_kv[seq : seq + 2].tolist()
                visibility[q0:q1, order[k0:k1]] = True
            expected = (query[:, None] >= 0) & (query[:, None] == keys[None, :])
            torch.testing.assert_close(visibility, expected)


@pytest.mark.parametrize(
    "samplespecs",
    [[(2, 3, 2)], [(3, 4, 0)], [(4, 2, 3)], [(2, 0, 2)], [(2, 2, 1), (1, 1, 1)]],
)
def test_tp_and_native_policy_preserve_valid_mask_outputs_and_gradients(monkeypatch, samplespecs):
    from cosmos_framework.model.generator.megatron_tp_pp import io as tp_io
    from cosmos_framework.model.generator.mot import small_action_tower

    torch.manual_seed(912)
    layer, shape = make_layer()
    baseline_layer = copy.deepcopy(layer)
    samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in samplespecs]
    u, v, a = (sum(spec[i] for spec in samplespecs) for i in range(3))
    raw, ids = make_pack(samples, shape, raw=True, outer_lengths=(8, 18))
    positions = tuple(from_all_seq(t, raw) for t in action_rope(ids, raw["causal_seq"].dtype, 16))
    baseline = small_action_tower.compact_pack(
        raw, positions, ids, shape, fixed_lengths=(8, 12, 6), attention_backend="sdpa", und_short_tokens=4
    )

    monkeypatch.setattr(small_action_tower, "compact_pack", small_action_tower.compact_pack)
    monkeypatch.setattr(tp_io, "compact_pack", tp_io.compact_pack)
    observations = []
    install((2, 4, 8), (2, 4, 6), observations)
    assert tp_io.compact_pack is small_action_tower.compact_pack
    candidate = tp_io.compact_pack(
        raw, positions, ids, shape, fixed_lengths=(8, 12, 6), attention_backend="sdpa", und_short_tokens=4
    )
    native = small_action_tower.compact_pack(
        raw, positions, ids, shape, fixed_lengths=(8, 12, 6), attention_backend="sdpa", und_short_tokens=4
    )
    assert candidate["_small_real_lengths"] == baseline["_small_real_lengths"] == native["_small_real_lengths"]
    assert len(observations) == 2 and observations[0] == observations[1]
    for stream, key in enumerate(("causal_seq", "full_only_seq", "action_seq")):
        count = baseline["_small_real_lengths"][stream]
        for packed in (candidate, native):
            torch.testing.assert_close(packed[key][:count], baseline[key][:count], atol=0, rtol=0)
            torch.testing.assert_close(packed["_small_ids"][stream][:count], baseline["_small_ids"][stream][:count])
    for active, query in ((False, 0), (False, 1), (True, 1), (True, 2)):
        torch.testing.assert_close(_valid_mask(candidate, query, active), _valid_mask(baseline, query, active))
        torch.testing.assert_close(_valid_mask(native, query, active), _valid_mask(baseline, query, active))

    actual = small_layer_forward(layer, small_decoder_inputs(candidate))[0]
    expected = small_layer_forward(baseline_layer, small_decoder_inputs(baseline))[0]
    for key, length in zip(("causal_seq", "full_only_seq", "action_seq"), (u, v, a), strict=True):
        torch.testing.assert_close(actual[key][:length], expected[key][:length], atol=3e-6, rtol=3e-5)
    actual_loss = actual["full_only_seq"][:v].square().sum() + actual["action_seq"][:a].square().sum()
    expected_loss = expected["full_only_seq"][:v].square().sum() + expected["action_seq"][:a].square().sum()
    torch.testing.assert_close(actual_loss, expected_loss, atol=3e-5, rtol=3e-5)
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (_, other) in zip(layer.named_parameters(), baseline_layer.named_parameters(), strict=True):
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-5, rtol=3e-5, msg=name)
