# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from tools.run_tp_pp_selective_ac import selected_layers, selective_config


def test_selective_candidate_copies_full_ac_policy_without_changing_baseline():
    original = ActivationCheckpointingConfig(mode="full", preserve_rng_state=True, determinism_check="default")
    result = selective_config(original, [r"^mm\.default$"])
    assert result.mode == "selective" and result.save_ops_regex == [r"^mm\.default$"]
    assert result.preserve_rng_state == original.preserve_rng_state
    assert result.determinism_check == original.determinism_check
    assert original.mode == "full" and original.save_ops_regex == ["fmha"]
    with pytest.raises(ValueError, match="full-AC baseline"):
        selective_config(result, ["other"])


def test_selective_layers_are_bounded_to_last_blocks_of_each_stage():
    assert selected_layers(("18", "19", "20", "21"), 2) == {"20", "21"}
    with pytest.raises(ValueError, match="fit every pipeline stage"):
        selected_layers(("0", "1"), 3)
