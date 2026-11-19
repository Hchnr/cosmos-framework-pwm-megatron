# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("kind", ["adjoint", "toy"])
def test_gloo_collective_and_three_stream_math(world, kind, tmp_path):
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/megatron" / f"tp_pp_sequence_parallel_{kind}.py"
    target = tmp_path / f"{kind}{world}"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        str(worker),
        "--backend",
        "gloo",
        "--output",
        str(target),
    ]
    if kind == "toy":
        command += ["--dtype", "float32"]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, check=True, timeout=90)
    for rank in range(world):
        report = json.loads((target / f"rank{rank}.json").read_text())
        assert report["passed"] and report["rank"] == rank and report["world"] == world
        assert report["production_acceptance"] is False
