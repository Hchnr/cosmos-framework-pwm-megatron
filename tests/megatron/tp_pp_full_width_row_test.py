# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("world", [2, 4])
def test_full_width_row_forward_backward_and_adam(world, tmp_path):
    worker = Path(__file__).with_name("tp_pp_full_width_row_gloo_worker.py")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "REPORT_DIR": str(tmp_path),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}", str(worker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for rank in range(world):
        report = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert report["passed"] and report["world"] == world and report["updates"] == 4
