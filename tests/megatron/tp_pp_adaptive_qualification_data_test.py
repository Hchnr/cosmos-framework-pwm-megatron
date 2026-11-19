# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import json

import pytest
import torch
from tp_pp_network_io_test import make_native

from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs
from tools.run_tp_pp_adaptive_padding import install
from tools.tp_pp_adaptive_qualification_data import make_qualification_sequence, qualification_cases, warmup_cases


def test_cases_json_roundtrip_warmup_and_boundaries():
    cases = qualification_cases()
    assert json.loads(json.dumps(cases)) == cases
    assert len({case["name"] for case in cases}) == len(cases)
    assert [case.name for case in warmup_cases()[1:]] == [case["name"] for case in cases]
    assert sum(cases[0]["und_lens"]) <= 2
    assert {sum(case["und_lens"]) for case in cases} == {1, 2, 3, 4}
    assert any(len(case["und_lens"]) == 2 for case in cases)
    assert any(not case["action"] for case in cases)
    assert any(not case["noisy_vision"] and not case["noisy_action"] for case in cases)


def _mask(query, keys, *, causal=False):
    out = ((query[:, None] >= 0) & (query[:, None] == keys[None, :])) | (
        (query[:, None] < 0) & (torch.arange(len(keys))[None, :] == 0)
    )
    if causal:
        out &= torch.arange(len(query))[:, None] >= torch.arange(len(keys))[None, :]
    return out


@pytest.mark.parametrize("case", qualification_cases(), ids=lambda case: case["name"])
def test_actual_production_pack_ids_masks_and_native_forward(case, monkeypatch):
    from cosmos_framework.model.generator.megatron_tp_pp import io as tp_io
    from cosmos_framework.model.generator.mot import small_action_tower

    torch.manual_seed(31)
    reference, shape = make_native()
    config = reference.config
    config.fixed_sequence_length_enabled = True
    config.fixed_und_tokens, config.fixed_gen_tokens, config.fixed_action_tokens = 8, 12, 6
    config.action_tower_und_short_tokens = 4
    model = reference.language_model.model
    model.small_action_fixed_lengths = (8, 6, 6)
    model.small_action_und_short_tokens = 4
    model._small_action_warm_full_und = False
    first = FirstStageInputs(config, reference.language_model.config, None, shape=shape, vocab_parallel=False).to_empty(
        device="cpu"
    )
    source = reference.state_dict()
    first.load_state_dict({key: source[key] for key in first.state_dict()}, strict=True)
    first._warm_full_und = False

    monkeypatch.setattr(small_action_tower, "compact_pack", small_action_tower.compact_pack)
    monkeypatch.setattr(tp_io, "compact_pack", tp_io.compact_pack)
    observations = []
    install((2, 4, 8), (2, 4, 6), observations)
    sequence = make_qualification_sequence(case)
    twin = make_qualification_sequence(case)
    assert torch.equal(sequence.vision.tokens[0], twin.vision.tokens[0])
    assert sequence.sequence_length == sum(sequence.sample_lens)
    assert sequence.attn_modes == ["causal", "full"] * len(case["und_lens"])
    batch = {"packed_sequence": copy.deepcopy(sequence)}
    pack = first(batch)
    output = reference(copy.deepcopy(sequence))
    assert all(torch.isfinite(value).all() for key in ("preds_vision", "preds_action") for value in output[key])
    assert len(observations) == 2 and observations[0] == observations[1]
    ids = batch["decoder_metadata"]["_small_ids"]
    expected = (
        [i for i, und in enumerate(case["und_lens"]) for _ in range(und)],
        [i for i in range(len(case["und_lens"])) for _ in range(2)],
        [i for i in range(len(case["und_lens"])) for _ in range(2)] if case["action"] else [],
    )
    for actual, expected_ids in zip(ids, expected, strict=True):
        assert actual.tolist() == expected_ids + [-1] * (len(actual) - len(expected_ids))
    masks = pack["_small_masks"]
    torch.testing.assert_close(masks[(False, 0)], _mask(ids[0], ids[0], causal=True))
    torch.testing.assert_close(masks[(False, 1)], _mask(ids[1], torch.cat(ids[:2])))
    torch.testing.assert_close(masks[(True, 1)], _mask(ids[1], torch.cat(ids)))
    torch.testing.assert_close(masks[(True, 2)], _mask(ids[2], torch.cat(ids)))
    for stream, real in zip((sequence.text_indexes, sequence.vision.sequence_indexes), expected[:2], strict=True):
        assert len(stream) == len(real)
    if case["action"]:
        assert sequence.action is not None and len(sequence.action.sequence_indexes) == len(expected[2])
    else:
        assert sequence.action is None
