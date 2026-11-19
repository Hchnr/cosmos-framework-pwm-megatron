# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import json
from pathlib import Path

import pytest

from tools import run_tp_pp_original_capacity_reference_coordinator as coordinator
from tools.run_tp_pp_untraced_capacity_math import validate_contract


@pytest.fixture
def gates():
    streamed = {
        "streamed_native": {"returncode": 0, "gate": {"passed": True}},
        "streamed_compare": {"returncode": 0, "gate": {"passed": True}},
    }
    case = {"passed": True, "returncode": 0, "gate": {"passed": True}}
    qualification = {"completed": True, "failure": None, "cases": {"checkpointed_chunk": case, "nested_memory": case}}
    return streamed, qualification


def test_all_three_prerequisites_are_mandatory(gates):
    streamed, qualification = gates
    coordinator.require_gates(streamed, qualification)
    for key in ("streamed_native", "streamed_compare"):
        failed = copy.deepcopy(streamed)
        failed[key]["gate"]["passed"] = False
        with pytest.raises(ValueError):
            coordinator.require_gates(failed, qualification)
    for key in ("checkpointed_chunk", "nested_memory"):
        failed = copy.deepcopy(qualification)
        failed["cases"][key]["gate"]["passed"] = False
        with pytest.raises(ValueError):
            coordinator.require_gates(streamed, failed)


@pytest.mark.parametrize("mode,tp", [("native", None), ("tp", 2), ("tp", 4)])
def test_generated_commands_use_original_capacity_and_continuous_trajectory(mode, tp):
    command = coordinator.command(Path("/tmp/work"), Path("/tmp/out.json"), Path("/tmp/reference"), mode, tp)
    marker = command.index("--mode")
    runner_args = command[marker:]
    assert validate_contract(runner_args, None if mode == "native" else Path("/tmp/order"), 4096) == mode
    assert "--original-capacity" in command and "--replay-native-state" not in command
    assert command[command.index("--prepared-dir") + 1] == str(coordinator.INPUTS)
    assert command[command.index("--reference-dir") + 1] == "/tmp/reference"


def test_completed_numerical_tp_failure_is_recorded_and_missing_rank_is_rejected(tmp_path):
    output = tmp_path / "tp4.json"
    report = {
        "updates": 3,
        "original_capacity": True,
        "attention_backend": "flex",
        "objective": "native_rf",
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "continuous_trajectory": True,
        "replay_native_state": False,
        "native_adam_arithmetic": True,
        "deterministic_algorithms": True,
        "activation_cpu_offload": True,
        "production_capacity_576_slot_acceptance": False,
        "gradient_summary": [{}, {}, {}],
        "parameter_gradient_errors": [{"update": i} for i in range(3)],
        "parameter_state_errors": [{"update": i} for i in range(3)],
        "compute_parameter_checks": [{"update": i} for i in range(3)],
        "pointwise_gates_passed": False,
        "passed": False,
        "gate_failures": ["grad"],
    }
    for rank in range(4):
        output.with_name(f"tp4.rank{rank}.json").write_text(json.dumps(report))
    gate = coordinator.tp_gate(output, 4, {"returncode": 1})
    assert gate["complete_observation"] and not gate["passed"]
    output.with_name("tp4.rank3.json").unlink()
    with pytest.raises(FileNotFoundError):
        coordinator.tp_gate(output, 4, {"returncode": 1})
