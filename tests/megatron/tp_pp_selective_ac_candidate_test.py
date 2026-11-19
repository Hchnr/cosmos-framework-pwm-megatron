# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from tools.run_tp_pp_selective_ac import install, selected_layers, selective_config


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


def test_selective_stage_uses_current_backend_compile_options(monkeypatch):
    from cosmos_framework.model.generator.mot import parallelize_unified_mot
    from cosmos_framework.training_backends.megatron_tp_pp import model, train

    options = {"reorder_for_compute_comm_overlap": True}
    observed = []
    monkeypatch.setattr(model, "inductor_compile_options", lambda config: options)
    monkeypatch.setattr(parallelize_unified_mot, "_apply_selective_ac", lambda block, config: block)
    monkeypatch.setattr(torch, "compile", lambda block, **kwargs: observed.append(kwargs) or block)
    monkeypatch.setattr(train, "optimize_stage", train.optimize_stage)
    ac = ActivationCheckpointingConfig(mode="full")
    compile_config = SimpleNamespace(enabled=True, compiled_region="language", use_cuda_graphs=False, compile_dynamic=False)
    stage = SimpleNamespace(layers=torch.nn.ModuleDict({"0": torch.nn.Identity()}))
    install([r"^mm\.default$"], 1)
    train.optimize_stage(stage, SimpleNamespace(activation_checkpointing=ac, compile=compile_config))
    assert len(observed) == 1
    assert observed[0]["options"] == options
