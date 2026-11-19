# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise the installed MCore DDP hooks without accessing a GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("overlap", [False, True])
def test_dynamic_last_microbatch_changes_overlap_readiness(tmp_path, overlap):
    output = tmp_path / "reports"
    command = [
        sys.executable,
        str(ROOT / "tools/tp_pp_overlap_unused_repro.py"),
        "--cpu",
        "--output",
        str(output),
    ]
    if overlap:
        command.append("--overlap")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads((output / f"{'overlap' if overlap else 'control'}.rank0.json").read_text())
    assert report["transport"] == "fake CPU handle"
    assert report["expected_behavior_observed"]
    assert report["results"][0]["finish_error"] is None
    error = report["results"][1]["finish_error"]
    if overlap:
        assert "Communication call has not been issued" in error
    else:
        assert error is None
    assert report["full_model_acceptance"] is False
