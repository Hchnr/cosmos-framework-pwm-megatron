# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise real TP projection and layer semantics with delayed collective waits."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("compiled", [False, True])
def test_delayed_row_collectives_match_real_tp_layer(world, compiled, tmp_path):
    output = tmp_path / "results"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        str(ROOT / "tests/megatron/tp_pp_async_row_layer_worker.py"),
        "--worktree",
        str(ROOT),
        "--output",
        str(output),
    ]
    if compiled:
        command += ["--compile", "--full-ac"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    for rank in range(world):
        report = json.loads((output / f"rank{rank}.json").read_text())
        assert report["rank"] == rank
        assert len(report["cases"]) == 4
        assert all(case["passed"] for case in report["cases"])
        if not compiled:
            assert all(case["baseline_forward_sums"] == case["candidate_forward_sums"] == 6 for case in report["cases"])
