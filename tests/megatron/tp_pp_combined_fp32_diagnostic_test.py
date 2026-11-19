# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_three_stream_combined_cpu(world, checkpoint, tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LD_LIBRARY_PATH": "",
        "PYTHONPATH": str(root),
        "COMBINED_AC": str(int(checkpoint)),
    }
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        str(root / "tests/megatron/tp_pp_combined_fp32_layer_worker.py"),
        "--worktree",
        str(root),
        "--output",
        str(tmp_path),
    ]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(world):
        report = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert len(report["cases"]) == 4
        assert max(row["max_relative_l2"] for row in report["cases"]) < 1e-4


def test_evidence_requires_both_reproduced_and_real_row_difference(tmp_path, monkeypatch):
    from tools import run_tp_pp_combined_fp32_sequence as sequence

    sdpa, row = tmp_path / "sdpa", tmp_path / "row"
    sdpa.mkdir()
    (row / "sequence").mkdir(parents=True)
    monkeypatch.setattr(sequence, "SDPA", sdpa)
    monkeypatch.setattr(sequence, "ROW", row)
    for directory in (sdpa, row):
        (directory / "queue_result.json").write_text(json.dumps({"completed": True, "failure": None, "returncode": 0}))
    (sdpa / "sdpa_replay.json").write_text("{}")
    case = {"ranks": [{"native_full_vs_shard_sum_common_input": {"exact": False}}]}
    (row / "sequence/summary.json").write_text(json.dumps({"passed": True, "cases": {"a": case, "b": case, "c": case}}))

    class Base:
        @staticmethod
        def sdpa_replay_decision(_):
            return True, "observed"

    assert sequence.evidence(Base())[0]
    case["ranks"][0]["native_full_vs_shard_sum_common_input"]["exact"] = True
    (row / "sequence/summary.json").write_text(json.dumps({"passed": True, "cases": {"a": case, "b": case, "c": case}}))
    assert not sequence.evidence(Base())[0]
    (row / "queue_result.json").write_text(json.dumps({"completed": True, "failure": "failed", "returncode": 1}))
    assert not sequence.evidence(Base())[0]
