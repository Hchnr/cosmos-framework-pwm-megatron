# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert command[command.index("--seed") + 1] == str(coordinator.DIAGNOSTIC_SEED)


@pytest.mark.parametrize("seed", [None, True, 41, "42"])
def test_native_reference_gate_rejects_missing_or_wrong_seed(tmp_path, seed):
    (tmp_path / "manifest.json").write_text(json.dumps({"passed": True, "seed": seed}))
    with pytest.raises(ValueError, match="Incomplete or incorrect"):
        coordinator.reference_gate(tmp_path, {"returncode": 0})


def test_snapshot_copies_isolated_source_without_modifying_base(tmp_path, monkeypatch):
    source = tmp_path / "source"
    root = tmp_path / "root"
    for base in ("tools", "tests", "cosmos_framework"):
        (source / base).mkdir(parents=True)
        (source / base / "module.py").write_text(f"{base} = True\n")
    (source / "tools/compare_tp_pp_runtime_checkpoints.py").write_text(
        'def metric(relative_l2, aa):\n    return {\n        "relative_l2": relative_l2,\n    }\n'
    )
    (source / "tests/megatron").mkdir()
    (source / "tests/megatron/tp_pp_full_source_math_cuda_compare_streamed_reference_v2.py").write_text(
        'def configuration(toml, *, original_capacity=False, attention_backend="sdpa"):\n'
        "    pass\n"
        "def native(args, names):\n"
        '    report = {\n        "passed": True,\n        "names": names,\n    }\n'
        "    return report\n"
        "def tp(args, config, language, network, inputs):\n"
        "    torch.backends.cuda.matmul.allow_tf32 = False\n"
        "    topology = ModelParallelConfig(\n        x=1\n    )\n"
        "    initialize(topology)\n"
        '    assert reference["input_hashes"] == [batch_fingerprint(s) for s in inputs]\n'
        "    bundle.load_optimizer(stage, driver.optimizer)\n"
        '    loss_module = GenerationLoss(config.model.config) if args.objective == "native_rf" else None\n'
        "    for update in range(3):\n        pass\n"
        "def report(absolute, difference, output, expected, batch, reference, dist):\n"
        "    output_errors = []\n"
        "    gate_failures, compute_parameter_checks = [], []\n"
        "    if True:\n"
        "        if not norm_passed:\n"
        "            pass\n"
        "        if True:\n"
        "            if not step_passed:\n"
        "                pass\n"
        '    compute_parameter_checks.append({"update": update, "name": spec.canonical_name, "compute_equals_master": model_master_match})\n'
        '    state_errors.append({"update": update, "name": spec.canonical_name, "field": key, **difference})\n'
        "    for key in expected:\n"
        "        for actual, target in zip(output[key], expected[key], strict=True):\n"
        '            output_errors.append({"update": 0, "slot": batch["slot_id"], "field": key, **difference})\n'
        "    return {\n"
        '                    "maximum_absolute_error": absolute.item(),\n'
        '                "scope": reference["scope"],\n'
        '                "compute_parameter_checks": compute_parameter_checks,\n'
        "    }\n"
        "def main():\n"
        '    parser.add_argument("--mode", choices=("native", "tp"), required=True)\n'
        "    if args.deterministic:\n        pass\n"
    )
    (root / "tools").mkdir(parents=True)
    for name in ("run_tp_pp_untraced_capacity_math.py", "run_tp_pp_checkpointed_chunked_mlp_math.py"):
        (root / "tools" / name).write_text(f"# {name}\n")
    monkeypatch.setattr(coordinator, "WORK", source)
    monkeypatch.setattr(coordinator, "ROOT", root)
    target = tmp_path / "snapshot"
    coordinator.snapshot_copy(target)
    assert (target / "cosmos_framework/module.py").read_text() == "cosmos_framework = True\n"
    assert (target / "tools/run_tp_pp_untraced_capacity_math.py").is_file()
    assert not (source / "tools/run_tp_pp_untraced_capacity_math.py").exists()
    assert '"reference_squared_l2": aa' in (target / "tools/compare_tp_pp_runtime_checkpoints.py").read_text()
    helper = target / "tests/megatron/tp_pp_full_source_math_cuda_compare_streamed_reference_v2.py"
    generated = helper.read_text()
    assert '"reference_squared_l2": difference.get("reference_squared_l2")' in generated
    assert 'parser.add_argument("--seed", type=int, default=42)' in generated
    assert "initialize(topology, seed=args.seed)" in generated
    assert generated.count('"seed": args.seed') == 2
    assert generated.count("validate_reference_seed(") == 3
    seed_validator = next(
        node
        for node in ast.parse(generated).body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_reference_seed"
    )
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[seed_validator], type_ignores=[])), "<seed-check>", "exec"),
        namespace,
    )
    namespace["validate_reference_seed"]({"seed": 42}, 42)
    for reference in ({}, {"seed": True}, {"seed": 41}):
        with pytest.raises(ValueError, match="requested seed"):
            namespace["validate_reference_seed"](reference, 42)
    functions = [
        node
        for node in ast.parse(generated).body
        if isinstance(node, ast.FunctionDef) and node.name in ("native", "tp")
    ]
    namespace["json"] = json
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "<seed-path>", "exec"),
        namespace,
    )
    assert namespace["native"](SimpleNamespace(seed=42), ["weight"])["seed"] == 42
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="requested seed"):
        namespace["tp"](SimpleNamespace(seed=42, reference_dir=tmp_path), None, None, None, None)
