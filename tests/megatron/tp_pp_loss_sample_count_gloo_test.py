# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("topology", ["tp2pp1dp2", "tp1pp2dp2", "tp2pp2dp1"])
def test_sample_level_denominators_across_real_gloo_groups(topology):
    worker = Path(__file__).with_name("tp_pp_loss_sample_count_gloo_worker.py")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(worker),
        "--topology",
        topology,
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{topology} passed" in result.stdout
