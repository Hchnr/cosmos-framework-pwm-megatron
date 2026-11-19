# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
import torch

from tools.tp_pp_fair_runtime import EXPECTED, RuntimeAudit, runtime_flags, validate_reports


def write_reports(root):
    for backend in ("tp", "fsdp"):
        directory = root / backend
        directory.mkdir()
        for rank in range(4):
            report = {
                "rank": rank,
                "backend": backend,
                "passed": True,
                "events": [
                    {
                        "phase": phase,
                        "flags": EXPECTED,
                        "compiler_runtime": {"recompile_limit": 128, "use_duck_shape": False},
                    }
                    for _ in range(21)
                    for phase in ("microbatch_before", "microbatch_after_prepare")
                ],
                "compiled_module_count": 36,
                "compiled_layer_indices": list(range(36)),
                "torch_compile_calls": [{"fullgraph": "True"}],
                "model_compile_calls": [{"fullgraph": True, "mode": None}],
                "compile_settings": {
                    "model": {"enabled": True},
                    "trainer_recompile_limit": 128,
                    "trainer_use_duck_shape": False,
                },
                "vae_compile": {"variant_keys": ["(1, 2, 3, 4)"]},
                "hostname": "same-host",
                "window_end_ns": 500,
                "steps": [{"iteration": 1201 + i, "start_ns": 10 + i * 20, "end_ns": 20 + i * 20} for i in range(7)],
            }
            (directory / f"runtime_fairness.rank{rank}.json").write_text(json.dumps(report))


def test_checks_real_backend_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "0")
    original = runtime_flags()
    try:
        from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer

        monkeypatch.setattr(torch, "compile", torch.compile)
        monkeypatch.setattr(CompileTokenizer, "on_training_step_start", CompileTokenizer.on_training_step_start)
        audit = RuntimeAudit(tmp_path, "tp")
        audit.compile_settings = {
            "trainer_recompile_limit": torch._dynamo.config.recompile_limit,
            "trainer_use_duck_shape": torch.fx.experimental._config.use_duck_shape,
        }
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        audit.check("real")
        torch.backends.cudnn.benchmark = True
        with pytest.raises(RuntimeError, match="settings differ"):
            audit.check("drift")
    finally:
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = original["matmul_allow_tf32"]
        torch.use_deterministic_algorithms(
            original["deterministic_algorithms"], warn_only=original["deterministic_warn_only"]
        )
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])


def test_all_ranks_must_match_compile_and_execute_each_slot(tmp_path):
    write_reports(tmp_path)
    assert validate_reports(tmp_path)["passed"]
    path = tmp_path / "fsdp/runtime_fairness.rank2.json"
    report = json.loads(path.read_text())
    report["compile_settings"]["model"]["enabled"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="compile settings differ"):
        validate_reports(tmp_path)
    report["compile_settings"]["model"]["enabled"] = True
    report["events"].pop(0)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="21 runtime"):
        validate_reports(tmp_path)


def test_window_uses_earliest_rank_start_and_latest_rank_end(tmp_path):
    write_reports(tmp_path)
    path = tmp_path / "tp/runtime_fairness.rank2.json"
    report = json.loads(path.read_text())
    report["steps"][2]["start_ns"] = 41
    report["window_end_ns"] = 600
    path.write_text(json.dumps(report))
    window = validate_reports(tmp_path)["windows"]["tp"]
    assert window["start_ns"] == 41
    assert window["end_ns"] == 600
    assert window["seconds"] == 559 / 1e9


def test_window_synchronizes_only_at_measured_boundaries(monkeypatch):
    import torch.distributed as dist

    audit = RuntimeAudit.__new__(RuntimeAudit)
    audit.steps = []
    audit.window_end_ns = None
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("cuda"))
    monkeypatch.setattr(dist, "barrier", lambda: calls.append("barrier"))
    for iteration in range(1200, 1207):
        audit.begin_update(iteration)
        audit.end_update(iteration)
    assert calls == ["cuda", "barrier", "cuda", "barrier"]
    assert len(audit.steps) == 7
    assert audit.window_end_ns >= audit.steps[-1]["end_ns"]


def test_actual_compiled_layer_coverage_is_tracked():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict(
        {str(i): torch.compile(torch.nn.Linear(2, 2), backend="eager") for i in range(3)}
    )
    audit = RuntimeAudit.__new__(RuntimeAudit)
    audit.backend = "tp"
    audit.compiled_module_count = None
    audit.compiled_layer_indices = None
    audit.modules(model)
    assert audit.compiled_layer_indices == [0, 1, 2]
    model.layers["1"] = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="coverage"):
        audit.modules(model)
