# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import pytest

from tools.run_tp_pp_inductor_overlap import (
    OPTION_ENABLED,
    OPTION_PASSES,
    PASSES,
    Observation,
    candidate_options,
    install,
)


def test_options_preserve_existing_settings_and_require_both_overlap_controls():
    original = {"triton.mix_order_reduction_split_size": 32}
    options = candidate_options(
        SimpleNamespace(enabled=True), original, deterministic_algorithms=False, inductor_deterministic=False
    )
    assert original == {"triton.mix_order_reduction_split_size": 32}
    assert options[OPTION_ENABLED] is True
    assert options[OPTION_PASSES] == list(PASSES)
    assert options["triton.mix_order_reduction_split_size"] == 32


def test_installed_inductor_accepts_and_restores_candidate_options():
    from torch._inductor import config

    previous = (config.reorder_for_compute_comm_overlap, list(config.reorder_for_compute_comm_overlap_passes))
    options = candidate_options(
        SimpleNamespace(enabled=True), {}, deterministic_algorithms=False, inductor_deterministic=False
    )
    with config.patch(options):
        assert config.reorder_for_compute_comm_overlap is True
        assert config.reorder_for_compute_comm_overlap_passes == list(PASSES)
    assert (config.reorder_for_compute_comm_overlap, config.reorder_for_compute_comm_overlap_passes) == previous


@pytest.mark.parametrize(
    "enabled,torch_deterministic,inductor_deterministic",
    [(False, False, False), (True, True, False), (True, False, True)],
)
def test_no_silent_fallback(enabled, torch_deterministic, inductor_deterministic):
    with pytest.raises(ValueError):
        candidate_options(
            SimpleNamespace(enabled=enabled),
            {},
            deterministic_algorithms=torch_deterministic,
            inductor_deterministic=inductor_deterministic,
        )


def test_conflicting_policy_is_rejected():
    with pytest.raises(ValueError, match="already set"):
        candidate_options(
            SimpleNamespace(enabled=True),
            {OPTION_PASSES: []},
            deterministic_algorithms=False,
            inductor_deterministic=False,
        )


def test_reports_observed_scheduler_and_collective_without_altering_result(tmp_path):
    observation = Observation(tmp_path / "rank0.json")
    config = SimpleNamespace(
        deterministic=False,
        reorder_for_compute_comm_overlap=True,
        reorder_for_compute_comm_overlap_passes=list(PASSES),
    )
    fake_torch = SimpleNamespace(
        _inductor=SimpleNamespace(config=config), are_deterministic_algorithms_enabled=lambda: False
    )
    model = SimpleNamespace(inductor_compile_options=lambda compile_config: {"existing": 5})
    comms = SimpleNamespace(
        reorder_compute_and_comm_for_overlap=lambda nodes: list(reversed(nodes)),
        is_collective=lambda node: node == "collective",
    )
    install(observation, model_module=model, comms_module=comms, torch_module=fake_torch)
    assert model.inductor_compile_options(SimpleNamespace(enabled=True))["existing"] == 5
    nodes = [SimpleNamespace(node="compute"), SimpleNamespace(node="collective")]
    assert comms.reorder_compute_and_comm_for_overlap(nodes) == nodes[::-1]
    observation.finish()
    report = json.loads(observation.target.read_text())
    assert report["effect_verified"]
    assert report["scheduler_calls"] == report["scheduler_calls_with_collectives"] == 1
    assert report["collective_nodes_observed"] == 1


def test_cache_hit_or_no_collectives_is_reported_as_unverified(tmp_path):
    observation = Observation(tmp_path / "rank0.json")
    observation.data["options_installed"] = True
    observation.finish()
    report = json.loads(observation.target.read_text())
    assert not report["effect_verified"]
    assert "cache hit" in report["failure"]
