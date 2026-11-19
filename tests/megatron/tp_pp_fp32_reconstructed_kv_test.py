# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.tp_pp_fp32_reconstructed_kv import reassemble


def test_reassembled_parameter_attaches_only_its_original_shard():
    pieces = [torch.nn.Parameter(torch.randn(2, 3)) for _ in range(2)]
    gathered = torch.cat([p.detach() for p in pieces])
    full = reassemble(gathered, pieces[0], rank=0, size=2)
    full.sum().backward()
    assert pieces[0].grad is not None and pieces[1].grad is None
    assert torch.equal(full.detach(), gathered)
    with pytest.raises(ValueError, match="shape or rank"):
        reassemble(gathered, pieces[0], rank=2, size=2)


def test_two_rank_gloo_forward_gradients_and_unused_adam(tmp_path):
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/megatron/tp_pp_fp32_reconstructed_kv_gloo_worker.py"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "LD_LIBRARY_PATH": "",
        "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in (0, 1):
        output = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert output["rank"] == rank and len(output["cases"]) == 3
        assert output["missing_grad_none_and_adam_skip"]
