# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The async-row numeric oracle must reject NaN/Inf without stranding peers."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests/megatron/tp_pp_async_row_layer_worker.py"


def load_oracle():
    spec = importlib.util.spec_from_file_location("_async_row_finite_oracle", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["reference", "candidate", "both"])
def test_nonfinite_values_fail_and_remain_standard_json(bad, side):
    oracle = load_oracle()
    good = torch.tensor([1.0, 2.0])
    bad_tensor = torch.tensor([1.0, bad])
    reference = bad_tensor if side in ("reference", "both") else good
    candidate = bad_tensor if side in ("candidate", "both") else good
    row = {"name": "output", **oracle.tensor_diff(reference, candidate)}
    assert row["reference_finite"] is (side == "candidate")
    assert row["candidate_finite"] is (side == "reference")
    maximum, failures, largest = oracle.compare_summary([row])
    assert maximum is None and failures == [row] and largest == [row]
    assert row["relative_l2"] is None and row["max_abs"] is None
    json.dumps({"maximum": maximum, "failures": failures}, allow_nan=False)


def test_finite_threshold_behavior_and_metric_overflow():
    oracle = load_oracle()
    ref = torch.tensor([1.0, 2.0])
    equal = {"name": "equal", **oracle.tensor_diff(ref, ref.clone())}
    different = {"name": "different", **oracle.tensor_diff(ref, ref + 1.0)}
    near_zero = {"name": "near_zero", **oracle.tensor_diff(torch.zeros(2), torch.full((2,), 1e-6))}
    near_relative = {"name": "near_relative", **oracle.tensor_diff(ref, ref + 1e-5)}
    extreme = torch.tensor([1e308, -1e308], dtype=torch.float64)
    overflow = {"name": "overflow", **oracle.tensor_diff(extreme, -extreme)}
    denominator_overflow = {"name": "denominator_overflow", **oracle.tensor_diff(extreme, extreme)}
    maximum, failures, largest = oracle.compare_summary([equal, different, overflow, denominator_overflow])
    assert equal["relative_l2"] == 0 and not oracle.comparison_failed(equal)
    assert near_zero["relative_l2"] > 1e-4 and not oracle.comparison_failed(near_zero)
    assert near_relative["max_abs"] > 2e-6 and not oracle.comparison_failed(near_relative)
    assert all(oracle.comparison_failed(row) for row in (different, overflow, denominator_overflow))
    assert maximum == different["relative_l2"] and failures == [different, overflow, denominator_overflow]
    assert largest[0] == overflow
    json.dumps({"maximum": maximum, "rows": largest}, allow_nan=False)


def test_empty_comparison_is_a_structured_failure():
    maximum, failures, largest = load_oracle().compare_summary([])
    assert maximum is None and len(failures) == 1 and largest == failures
    assert failures[0]["name"] == "<no compared fields>"
    json.dumps({"maximum": maximum, "failures": failures}, allow_nan=False)


def distributed_worker():
    """One bad rank must reach the same collective failure decision as its peer."""
    oracle = load_oracle()
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError("Two ranks required")
    for bad_rank in (0, 1):
        for bad in (float("nan"), float("inf"), float("-inf")):
            for side in ("reference", "candidate", "both"):
                ref = torch.tensor([1.0, 2.0])
                cand = ref.clone()
                if rank == bad_rank:
                    if side in ("reference", "both"):
                        ref[1] = bad
                    if side in ("candidate", "both"):
                        cand[1] = bad
                row = {"name": "value", **oracle.tensor_diff(ref, cand)}
                maximum, failures, largest = oracle.compare_summary([row])
                failed = oracle.synchronize_failures(failures, torch.device("cpu"))
                assert failed and bool(failures) == (rank == bad_rank)
                json.dumps({"rank": rank, "maximum": maximum, "largest": largest}, allow_nan=False)
    row = {"name": "finite", **oracle.tensor_diff(torch.ones(2), torch.ones(2))}
    for empty_rank in (0, 1):
        maximum, failures, largest = oracle.compare_summary([] if rank == empty_rank else [row])
        assert oracle.synchronize_failures(failures, torch.device("cpu"))
        assert bool(failures) == (rank == empty_rank)
        json.dumps({"maximum": maximum, "largest": largest}, allow_nan=False)
    _, failures, _ = oracle.compare_summary([row])
    assert not oracle.synchronize_failures(failures, torch.device("cpu"))
    print(f"FINITE_SYNC_PASS rank={rank}", flush=True)
    dist.destroy_process_group()


def test_rank_local_nonfinite_failure_is_collective():
    env = dict(os.environ)
    for key in ("LOAD_TRACE_DIR", "LOAD_TRACE_TAG", "LOAD_TRACE_ROOT", "COSMOS_DL_FILE_SYSTEM_SHARING"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
    )
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(Path(__file__))],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FINITE_SYNC_PASS rank=0" in result.stdout
    assert "FINITE_SYNC_PASS rank=1" in result.stdout


if __name__ == "__main__":
    distributed_worker()
