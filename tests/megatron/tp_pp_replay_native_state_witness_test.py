# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.run_tp_pp_replay_native_state_witness import install_state_witness, read_report, validate_arguments


def arguments(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    target = tmp_path / "target.py"
    target.write_text("")
    output = tmp_path / "new.json"
    return [
        "--replay-native-state",
        "--observe-all-updates",
        "--mode",
        "tp",
        "--tp",
        "4",
        "--reference-dir",
        str(reference),
        "--run-path",
        str(target),
        "--output",
        str(output),
    ]


def test_replay_arguments_require_scope_and_existing_inputs(tmp_path):
    args = arguments(tmp_path)
    reference, output, target = validate_arguments(args)
    assert reference.is_dir() and target.is_file() and not output.exists()
    for changed in (args[:-2], args + ["--tp", "2"], [arg for arg in args if arg != "--replay-native-state"]):
        with pytest.raises(ValueError):
            validate_arguments(changed)


def test_report_is_explicitly_diagnostic_and_contains_all_updates(tmp_path):
    path = tmp_path / "rank0.json"
    report = {
        "updates": 3,
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "replay_native_state": True,
        "continuous_trajectory": False,
        "passed": False,
        "gradient_summary": [{}, {}, {}],
        "pointwise_gates_passed": False,
    }
    path.write_text(json.dumps(report))
    assert not read_report(path)["pointwise_gates_passed"]
    for key, invalid in (
        ("continuous_trajectory", True),
        ("passed", True),
        ("updates", 2),
        ("gradient_summary", [{}, {}]),
    ):
        altered = {**report, key: invalid}
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            read_report(path)


@pytest.mark.parametrize("corrupt_moment", [False, True])
def test_state_witness_checks_master_moments_step_and_compute_before_forward(tmp_path, monkeypatch, corrupt_moment):
    from cosmos_framework.training_backends.megatron_tp_pp import driver, optimizer, parameter_map

    reference = tmp_path / "reference"
    reference.mkdir()
    names = [f"p{i}" for i in range(811)]
    (reference / "manifest.json").write_text(json.dumps({"passed": True, "names": names}))
    master = torch.tensor([1.0, 2.0])
    saved = {
        "master": master.clone(),
        "exp_avg": torch.tensor([3.0, 4.0]),
        "exp_avg_sq": torch.tensor([5.0, 6.0]),
        "step": torch.tensor(7),
    }
    entries = [
        SimpleNamespace(
            name=f"local{i}",
            master=master,
            model_parameter=master.clone(),
            state={"exp_avg": saved["exp_avg"].clone(), "exp_avg_sq": saved["exp_avg_sq"].clone()},
            step=7,
            select=lambda tensor: tensor,
        )
        for i in range(811)
    ]
    if corrupt_moment:
        entries[0].state["exp_avg"][0] += 1
    specs = {f"local{i}": SimpleNamespace(canonical_name=f"net.p{i}", select=lambda tensor: tensor) for i in range(811)}
    calls = []
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: saved)
    monkeypatch.setattr(optimizer, "adam_parameter_states", lambda *args: entries)
    monkeypatch.setattr(parameter_map, "stage_parameter_manifest", lambda *args: specs)
    monkeypatch.setattr(driver.ModelParallelDriver, "forward_backward", lambda *args, **kwargs: calls.append("forward"))
    install_state_witness(reference, tmp_path / "witness")
    instance = SimpleNamespace(iteration=1, stage=object(), optimizer=object())
    if corrupt_moment:
        with pytest.raises(RuntimeError, match="exp_avg replay mismatch"):
            driver.ModelParallelDriver.forward_backward(instance, (), None)
        assert calls == []
        assert not (tmp_path / "witness/replay_state.update1.rank0.json").exists()
    else:
        driver.ModelParallelDriver.forward_backward(instance, (), None)
        assert calls == ["forward"]
        evidence = json.loads((tmp_path / "witness/replay_state.update1.rank0.json").read_text())
        assert evidence["verified_shards"] == 811 and evidence["master_moments_step_compute_exact"]


def test_expected_diagnostic_exit_is_recognized_only_after_all_witnesses(tmp_path, monkeypatch):
    from tools import run_tp_pp_replay_native_state_witness as wrapper

    args = arguments(tmp_path)
    target = tmp_path / "target.py"
    target.write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.with_name(out.stem + '.rank0.json').write_text(json.dumps({"
        "'updates': 3, 'tensor_parallel_size': 4, 'pipeline_parallel_size': 1, "
        "'replay_native_state': True, 'continuous_trajectory': False, 'passed': False, "
        "'gradient_summary': [{}, {}, {}]}))\n"
        f"raise SystemExit({wrapper.EXPECTED_EXIT!r})\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import runpy, sys\nrunpy.run_path(sys.argv[sys.argv.index('--run-path') + 1], run_name='__main__')\n"
    )
    witness = tmp_path / "witness"
    witness.mkdir()
    for update in (1, 2):
        (witness / f"replay_state.update{update}.rank0.json").write_text(
            json.dumps({"verified_shards": 811, "master_moments_step_compute_exact": True})
        )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(wrapper, "install_state_witness", lambda *unused: None)
    # The wrapper intentionally patches these global values while running.
    monkeypatch.setattr(runpy, "run_path", runpy.run_path)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    wrapper.main(["--vjp-entry", str(entry), "--witness-dir", str(witness), *args])
    assert json.loads((tmp_path / "new.rank0.json").read_text())["continuous_trajectory"] is False
