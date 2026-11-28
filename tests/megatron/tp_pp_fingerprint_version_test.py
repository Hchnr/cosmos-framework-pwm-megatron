# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from tools.verify_tp_pp_native_data_replay import fingerprint_evidence_flags


def test_legacy_recorded_match_never_claims_data_or_noise_acceptance():
    slots = [{"slot": [0, step, rank]} for rank in range(192) for step in range(3)]
    legacy = fingerprint_evidence_flags(1, prepared=True, logical_dp=192, slots=slots)
    assert legacy["recorded_fingerprints_match"] is True
    assert legacy["legacy_fingerprint_ambiguous"] is True
    assert legacy["passed"] is False
    assert legacy["noise_equality_independently_checked"] is False
    assert legacy["production_576_slot_raw_data_accepted"] is False
    current = fingerprint_evidence_flags(2, prepared=True, logical_dp=192, slots=slots)
    assert current["passed"] is True
    assert current["noise_equality_independently_checked"] is True
    assert current["production_576_slot_raw_data_accepted"] is True


def test_production_slot_coverage_rejects_partial_and_duplicate():
    slots = [{"slot": [0, step, rank]} for rank in range(192) for step in range(3)]
    duplicate = slots[:-1] + slots[:1]
    partial_with_extra_steps = [{"slot": [0, step, rank]} for rank in range(96) for step in range(6)]
    for invalid in (duplicate, partial_with_extra_steps):
        result = fingerprint_evidence_flags(2, prepared=True, logical_dp=192, slots=invalid)
        assert result["production_576_slot_raw_data_accepted"] is False
