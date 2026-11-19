# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check grouped TP reductions against separate reductions and real MCore rows."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qualify_tp_pp_multirow_reduction import update_collectives, upgrade_kv_heads
from tools.tp_pp_multirow_reduction import GroupedRowSum

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("kind", ["gloo", "mcore", "layer"])
def test_grouped_row_sum_distributed(tmp_path, world, kind):
    target = tmp_path / f"{kind}{world}"
    worker = ROOT / "tests/megatron" / f"tp_pp_multirow_{kind}_worker.py"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}", str(worker)]
    if kind == "layer":
        command += ["--worktree", str(ROOT)]
    command += ["--output", str(target)]
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LD_LIBRARY_PATH=""
    )
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads((target / f"rank{rank}.json").read_text()) for rank in range(world)]
    assert all(row["rank"] == rank for rank, row in enumerate(rows))
    assert all(len(row["cases"]) == (9 if kind == "gloo" else 4) for row in rows)
    if kind == "layer":
        assert all(
            case["baseline_forward_sums"] == 6 and case["candidate_forward_sums"] == 2
            for row in rows
            for case in row["cases"]
        )
        assert all(case["max_relative_l2"] < 1e-4 for row in rows for case in row["cases"])


@pytest.mark.parametrize("world", [2, 4])
def test_grouped_row_sum_full_ac_aot_eager(tmp_path, world):
    target = tmp_path / f"compiled{world}"
    worker = ROOT / "tests/megatron/tp_pp_multirow_layer_worker.py"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        str(worker),
        "--worktree",
        str(ROOT),
        "--output",
        str(target),
        "--compile",
        "--full-ac",
    ]
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LD_LIBRARY_PATH=""
    )
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads((target / f"rank{rank}.json").read_text()) for rank in range(world)]
    for row in rows:
        assert len(row["cases"]) == 4
        assert all(case["max_relative_l2"] < 1e-4 and case["full_activation_checkpointing"] for case in row["cases"])
        assert all(case["candidate_forward_sums"] is None for case in row["cases"])


def test_group_rejects_mixed_dtype_before_any_collective():
    with pytest.raises(ValueError, match="same device and dtype"):
        GroupedRowSum.apply("unused", torch.ones(2), torch.ones(3, dtype=torch.float64))


def test_empty_group_preserves_unused_gradients_without_collective():
    frozen = torch.empty(0)
    active = torch.empty(0, requires_grad=True)
    unused = torch.empty(0, requires_grad=True)
    first, second, third = GroupedRowSum.apply("unused", frozen, active, unused)
    assert not first.requires_grad
    second.sum().backward()
    assert active.grad is not None
    assert unused.grad is None


def test_update_collective_counter_ignores_outside_ranges():
    markers = [SimpleNamespace(name="multirow_forward_backward_update", cpu_parent=None) for _ in range(2)]
    inside = SimpleNamespace(name="wrapper", cpu_parent=markers[0])
    events = [
        *markers,
        SimpleNamespace(name="_c10d_functional::all_reduce", cpu_parent=inside),
        SimpleNamespace(name="_c10d_functional::all_reduce", cpu_parent=markers[1]),
        SimpleNamespace(name="_c10d_functional::all_reduce", cpu_parent=None),
    ]
    assert update_collectives(events) == [1, 1]


def test_tp4_tiny_reference_uses_four_kv_heads(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tests/megatron"))
    from tp_pp_network_io_test import make_native, make_sequence

    torch.manual_seed(41)
    model, shape = upgrade_kv_heads(*make_native())
    assert shape.kv_heads == 4
    assert all(layer.self_attn.num_key_value_heads == 4 for layer in model.language_model.model.layers)
    outputs = model(make_sequence())
    assert outputs["preds_vision"] and outputs["preds_action"]
