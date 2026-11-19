# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qualify_tp_pp_und_selective_ac import configure_native_buckets, validate


def valid_inputs():
    observation = SimpleNamespace(data={"stage_count": 1, "selected_blocks": 2, "saved_mm_by_tokens": {"4": 1, "8": 1}})
    exercise = {"missing_action": 1, "missing_noisy_targets": 1, "actual_und_tokens": {"4": 1, "8": 1}}
    smoke = {
        "passed": True,
        "compiled": True,
        "groups": {"tp": [0, 1], "pp": [0], "dp": [0]},
        "relative_gradient_l2": {f"{i}/net.action_block_moe_gen_action.w": 1e-6 for i in range(4)},
    }
    return observation, exercise, smoke


def test_oracle_requires_two_actual_buckets_and_both_policies():
    observation, exercise, smoke = valid_inputs()
    assert validate(observation, exercise, smoke)["gradient_entries"] == 4
    del exercise["actual_und_tokens"]["8"]
    with pytest.raises(ValueError, match="bucket 8 did not actually execute"):
        validate(observation, exercise, smoke)
    observation, exercise, smoke = valid_inputs()
    del observation.data["saved_mm_by_tokens"]["4"]
    with pytest.raises(ValueError, match="bucket 4 did not trigger"):
        validate(observation, exercise, smoke)


@pytest.mark.parametrize("item", ["missing_action", "missing_noisy_targets"])
def test_oracle_rejects_missing_coverage(item):
    observation, exercise, smoke = valid_inputs()
    exercise[item] = 0
    with pytest.raises(ValueError, match="Missing-modality"):
        validate(observation, exercise, smoke)


def test_oracle_rejects_missing_fourth_update_or_unselected_stage():
    observation, exercise, smoke = valid_inputs()
    truncated = copy.deepcopy(smoke)
    del truncated["relative_gradient_l2"]["3/net.action_block_moe_gen_action.w"]
    with pytest.raises(ValueError, match="update 3"):
        validate(observation, exercise, truncated)
    observation.data["selected_blocks"] = 1
    with pytest.raises(ValueError, match="both blocks"):
        validate(observation, exercise, smoke)


def test_qualification_entry_is_executable_without_gpu_or_pythonpath(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/qualify_tp_pp_und_selective_ac.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--shared-input-reduction" in result.stdout


def test_tiny_fixed_buckets_keep_native_and_stage_predictions_equal():
    from tp_pp_network_io_test import make_native, make_sequence

    from cosmos_framework.model.generator.megatron_tp_pp.io import FirstStageInputs, LastStageOutputs
    from cosmos_framework.model.generator.megatron_tp_pp.payload import STREAMS, PayloadLayout

    torch.manual_seed(421)
    native, shape = make_native()
    configure_native_buckets(native)
    first = FirstStageInputs(
        native.config, native.language_model.config, None, shape=shape, vocab_parallel=False
    ).to_empty(device="cpu")
    last = LastStageOutputs(
        native.config, native.language_model.config, None, shape=shape, vocab_parallel=False
    ).to_empty(device="cpu")
    source = native.state_dict()
    first.load_state_dict({name: source[name] for name in first.state_dict()}, strict=True)
    last.load_state_dict({name: source[name] for name in last.state_dict()}, strict=True)
    first._warm_full_und = True
    layers = copy.deepcopy(native.language_model.model.layers)
    for tokens in (8, 4):
        sequence = make_sequence()
        expected = native(copy.deepcopy(sequence))
        batch = {"packed_sequence": sequence}
        pack = first(batch)
        assert tuple(pack[key].shape[0] for key in STREAMS) == (tokens, 6, 2)
        layout = PayloadLayout(tuple(tuple(pack[key].shape) for key in STREAMS), context_width=96)
        pack = {**pack, **layout.unpack(layout.pack(pack), frozen_und=True)}
        context = pack.pop("pipeline_context")
        for layer in layers:
            pack = layer(pack, None, None)[0]
        pack["pipeline_context"] = context
        actual = last(layout.unpack(layout.pack(pack), frozen_und=True), batch)
        for key in ("preds_vision", "preds_action"):
            for candidate, reference in zip(actual[key], expected[key], strict=True):
                torch.testing.assert_close(candidate, reference, atol=2e-6, rtol=2e-5)
