# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.tp_pp_sharded_vae_experiment import assign_items, native_training_items


def test_stable_balanced_assignment_and_invalid_costs():
    assert assign_items([36, 27, 8], 2) == [0, 1, 1]
    assert assign_items([5, 5, 5], 4) == [0, 1, 2]
    assert assign_items([], 2) == []
    for costs, workers in (([0], 2), ([-1], 2), ([True], 2), ([1], 0)):
        with pytest.raises(ValueError):
            assign_items(costs, workers)


def test_native_flattened_sample_and_multiview_contract():
    raw = [torch.zeros((1, 3, 4, 2, 2)), torch.ones((1, 3, 2, 2, 2))]
    items = native_training_items(raw, [2], None, [2, 1], [2, 2])
    assert all(item.tensor is original for item, original in zip(items, raw, strict=True))
    assert [(item.num_views, item.frames_per_view) for item in items] == [(2, 2), (1, 2)]
    with pytest.raises(ValueError, match="full-encode"):
        native_training_items(raw, [2], [[0]])
    with pytest.raises(ValueError, match="sample grouping"):
        native_training_items(raw, [1], None)
    with pytest.raises(ValueError, match="provided together"):
        native_training_items(raw, None, None, [1, 1], None)


@pytest.mark.parametrize(
    ("mode", "workers", "owner"),
    [
        ("normal", 2, 0),
        ("normal", 2, 1),
        ("normal", 4, 0),
        ("worker_failure", 2, 0),
        ("invalid_owner", 2, 0),
        ("native_factory_error", 2, 0),
    ],
)
def test_cpu_gloo_protocol(tmp_path, mode, workers, owner):
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / f"{mode}.json"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LD_LIBRARY_PATH": "",
        "PYTHONPATH": str(root),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={workers}",
            str(root / "tests/megatron/tp_pp_sharded_vae_gloo_worker.py"),
            "--output",
            str(target),
            "--mode",
            mode,
            "--owner",
            str(owner),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    for rank in range(workers):
        result_file = target.with_name(f"{target.stem}.rank{rank}.json")
        report = json.loads(result_file.read_text())
        assert report["passed"] and report["mode"] == mode and report["rank"] == rank
