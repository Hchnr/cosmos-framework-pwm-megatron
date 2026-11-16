# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fresh-process distributed resume contracts. Requires eight idle Hopper GPUs."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch


def assert_exact_state(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=path)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_exact_state(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact_state(a, b, f"{path}.{index}")
    else:
        assert left == right, path


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(8)
@pytest.mark.parametrize("recipe,shard", [("rowwise", 2), ("rowwise_with_gw_hp", 8)])
def test_fp8_fresh_process_resume(tmp_path, recipe, shard):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 training requires Hopper or newer")
    environment = dict(
        os.environ, LD_LIBRARY_PATH="", OMP_NUM_THREADS="1", PYTORCH_ALLOC_CONF="expandable_segments:True"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "-m",
        "cosmos_framework.scripts.validate_fp8_training",
        f"--output-root={tmp_path}",
        f"--recipe={recipe}",
        f"--shard-size={shard}",
        "--compile",
    ]
    for run, steps, flags in [("split", 5, ["--async-save"]), ("split", 10, []), ("continuous", 10, [])]:
        log = tmp_path / f"{run}_{steps}.log"
        with log.open("w") as output:
            process = subprocess.run(
                command + [f"--run={run}", f"--steps={steps}", *flags],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        assert process.returncode == 0, log.read_text()[-12000:]
    root = tmp_path / "distributed" / recipe
    for rank in range(8):
        continuous = torch.load(root / "continuous" / f"result_rank{rank}.pt", weights_only=False)
        split = torch.load(root / "split" / f"result_rank{rank}.pt", weights_only=False)
        continuous["losses"] = continuous["losses"][-5:]
        assert_exact_state(continuous, split, f"rank{rank}")
    # The saved precision manifest must reject an accidental same-job switch
    # through the real DCP load path, before another update/save is performed.
    changed = "rowwise_with_gw_hp" if recipe == "rowwise" else "rowwise"
    with (tmp_path / "precision_drift.log").open("w") as output:
        process = subprocess.run(
            command + [f"--recipe={changed}", f"--job-group={recipe}", "--run=split", "--steps=11"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    assert process.returncode != 0
    assert "Training precision changed during same-job resume" in (tmp_path / "precision_drift.log").read_text()
    assert (root / "split" / "checkpoints" / "latest_checkpoint.txt").read_text().strip() == "iter_000000010"


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(8)
def test_fp8_precision_migration_and_weights_only(tmp_path):
    """Restore real DCP slots before updating in both precision directions."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 training requires Hopper or newer")
    environment = dict(os.environ, LD_LIBRARY_PATH="", OMP_NUM_THREADS="1")

    def run(recipe, name, steps, source=None, weights_only=False, shard=2):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            "-m",
            "cosmos_framework.scripts.validate_fp8_training",
            f"--output-root={tmp_path}",
            f"--recipe={recipe}",
            f"--run={name}",
            f"--steps={steps}",
            f"--shard-size={shard}",
        ]
        if source is not None:
            command.append(f"--load-path={source}")
        if weights_only:
            command.append("--weights-only")
        with (tmp_path / f"{name}.log").open("w") as output:
            process = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=300)
        assert process.returncode == 0, (tmp_path / f"{name}.log").read_text()[-12000:]
        return tmp_path / "distributed" / recipe / name

    def checkpoint(run_dir, step):
        return run_dir / "checkpoints" / f"iter_{step:09d}"

    baseline = run("bf16", "baseline", 2)
    forward = run("rowwise", "from_bf16", 3, checkpoint(baseline, 2))
    reverse = run("bf16", "from_fp8", 4, checkpoint(forward, 3))
    weights = run("rowwise", "weights_only", 1, checkpoint(baseline, 2), weights_only=True)
    for rank in range(8):
        for source, destination in [(baseline, forward), (forward, reverse)]:
            saved = torch.load(source / f"result_rank{rank}.pt", weights_only=False)
            loaded = torch.load(destination / f"loaded_rank{rank}.pt", weights_only=False)
            for key in ("model", "optimizer", "scheduler", "iteration"):
                assert_exact_state(saved[key], loaded[key], f"rank{rank}.{key}")
        saved = torch.load(baseline / f"result_rank{rank}.pt", weights_only=False)
        loaded = torch.load(weights / f"loaded_rank{rank}.pt", weights_only=False)
        assert_exact_state(saved["model"], loaded["model"], f"rank{rank}.weights_only")
        assert loaded["iteration"] == 0
        fresh = torch.load(baseline / f"loaded_rank{rank}.pt", weights_only=False)
        # get_optimizer_state_dict initializes lazy slots in the installed
        # PyTorch. Compare with a fresh optimizer instead of assuming step=0.
        assert_exact_state(fresh["optimizer"], loaded["optimizer"], f"rank{rank}.fresh_optimizer")
        assert_exact_state(fresh["scheduler"], loaded["scheduler"], f"rank{rank}.fresh_scheduler")
    # A topology change promises a working load/update, not identical RNG/data.
    run("rowwise", "resharded", 4, checkpoint(forward, 3), shard=8)
