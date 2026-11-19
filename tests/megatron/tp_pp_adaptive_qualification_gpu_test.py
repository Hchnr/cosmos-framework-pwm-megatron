# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest
import torch

from cosmos_framework.model.generator.mot.small_action_tower import build_small_masks
from tools.qualify_tp_pp_adaptive_padding_gpu import audit_real_pack, compare_real_packs, validate_coverage


def rows_and_events():
    cases = [{"name": "short"}, {"name": "long"}]
    rows = []
    for mode in ("fsdp_full", "fsdp_adaptive", "tp_adaptive"):
        rows.append({"phase": "warmup_full", "mode": mode, "case": "short", "capacity": [8, 6, 2]})
        for phase in ("warmup", "update"):
            for name, cap in (("short", 2), ("long", 4)):
                rows.append({"phase": phase, "mode": mode, "case": name, "capacity": [cap, 6, 2]})
    return rows, [{"phase": "warmup_full", "mode": "fsdp_full", "case": "short", "compile_id": "0"}], cases


def test_every_backend_case_and_bucket_must_be_warmed_before_updates():
    rows, events, cases = rows_and_events()
    validate_coverage(rows, events, cases)
    broken = [
        row
        for row in rows
        if not (row["phase"] == "warmup" and row["case"] == "short" and row["mode"] == "tp_adaptive")
    ]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_coverage(broken, events, cases)
    broken = copy.deepcopy(rows)
    next(row for row in broken if row["phase"] == "update" and row["case"] == "short" and row["mode"] == "tp_adaptive")[
        "capacity"
    ][0] = 4
    with pytest.raises(ValueError, match="capacity differ"):
        validate_coverage(broken, events, cases)
    with pytest.raises(ValueError, match="compile variant"):
        validate_coverage(rows, [*events, {"phase": "update"}], cases)


def tiny_pack():
    ids = [torch.tensor([0, 1, -1, -1]), torch.tensor([0, 0, 1, -1]), torch.tensor([0, 1, -1])]
    return {
        "_small_ids": ids,
        "_small_real_lengths": (2, 3, 2),
        "_small_masks": build_small_masks(ids, "sdpa", 2, fixed_capacity=True),
        "causal_seq": torch.arange(8).reshape(4, 2).float(),
        "full_only_seq": torch.arange(8).reshape(4, 2).float(),
        "action_seq": torch.arange(6).reshape(3, 2).float(),
    }


def test_audit_checks_real_sample_boundaries_and_causal_visibility():
    pack = tiny_pack()
    reference = audit_real_pack(pack)
    compare_real_packs(reference, audit_real_pack(copy.deepcopy(pack)))
    # A query in sample 0 must never see the VIDEO key in sample 1.
    pack["_small_masks"][(True, 1)][0, 4 + 2] = True
    with pytest.raises(AssertionError, match="sample/causal mask"):
        audit_real_pack(pack)


def test_real_token_or_sample_id_differences_fail():
    first = audit_real_pack(tiny_pack())
    second = copy.deepcopy(first)
    second["tokens"][1][0, 0] += 1
    with pytest.raises(AssertionError):
        compare_real_packs(first, second)
    second = copy.deepcopy(first)
    second["ids"][0][0] = 1
    with pytest.raises(AssertionError, match="IDs differ"):
        compare_real_packs(first, second)


def test_actual_native_and_tp_input_audits_match_on_all_cases(monkeypatch):
    from tp_pp_network_io_test import make_native

    from cosmos_framework.model.generator.megatron_tp_pp import io as tp_io
    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs
    from cosmos_framework.model.generator.mot import small_action_tower
    from tools.run_tp_pp_adaptive_padding import install
    from tools.tp_pp_adaptive_qualification_data import make_qualification_sequence, qualification_cases

    torch.manual_seed(43)
    reference, shape = make_native()
    cfg = reference.config
    cfg.fixed_sequence_length_enabled = True
    cfg.fixed_und_tokens, cfg.fixed_gen_tokens, cfg.fixed_action_tokens = 8, 10, 4
    cfg.action_tower_und_short_tokens = 4
    reference.language_model.model.small_action_fixed_lengths = (8, 6, 4)
    reference.language_model.model.small_action_und_short_tokens = 4
    reference.language_model.model._small_action_warm_full_und = False
    first = FirstStageInputs(cfg, reference.language_model.config, None, shape=shape, vocab_parallel=False).to_empty(
        device="cpu"
    )
    first.load_state_dict({key: reference.state_dict()[key] for key in first.state_dict()}, strict=True)
    first._warm_full_und = False
    original = small_action_tower.compact_pack
    monkeypatch.setattr(small_action_tower, "compact_pack", original)
    monkeypatch.setattr(tp_io, "compact_pack", tp_io.compact_pack)
    observations = []
    install((2, 4, 8), (1, 2, 4), observations)
    adaptive = small_action_tower.compact_pack
    captures = []
    mode = {"adaptive": False}

    def capture(*args, **kwargs):
        pack = (adaptive if mode["adaptive"] else original)(*args, **kwargs)
        captures.append(audit_real_pack(pack))
        return pack

    small_action_tower.compact_pack = capture
    tp_io.compact_pack = capture
    for case in qualification_cases():
        sequence = make_qualification_sequence(case)
        mode["adaptive"] = False
        full = reference(copy.deepcopy(sequence))
        mode["adaptive"] = True
        narrow = reference(copy.deepcopy(sequence))
        first({"packed_sequence": copy.deepcopy(sequence)})
        compare_real_packs(captures[-3], captures[-2])
        compare_real_packs(captures[-3], captures[-1])
        for key in ("preds_vision", "preds_action"):
            for expected, actual in zip(full[key], narrow[key], strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
