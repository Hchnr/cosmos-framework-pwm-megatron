"""Exercise the production Action128 shell topology without allocating GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launch_env(tmp_path):
    env = os.environ.copy()
    for key in (
        "NNODES",
        "NPROC_PER_NODE",
        "DP_SHARD",
        "NODE_RANK",
        "RANK",
        "MASTER_ADDR",
        "RDZV_ENDPOINT",
        "RDZV_ID",
        "WANDB_MODE",
        "WANDB_API_KEY",
        "MAX_ITER",
        "SFT_TRAINING_MODULE",
        "IMAGINAIRE_OUTPUT_ROOT",
        "DATASET_PATH",
        "LOG_FILENAME",
        "JOB_ID",
        "RDZV_CONF",
        "RDZV_BACKEND",
        "ACTION128_PLATFORM_RENDEZVOUS",
        "ACTION128_RDZV_CONF",
        "ACTION128_DISABLE_THP",
        "EXTRA_TAIL_OVERRIDES",
    ):
        env.pop(key, None)
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "torchrun"
    fake.write_text(
        f'#!{sys.executable}\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n'
    )
    fake.chmod(0o755)
    for name in ("catalog.parquet", "vae.pth"):
        (tmp_path / name).touch()
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(f'[[dataloader_train.datasets]]\npath="{tmp_path / "catalog.parquet"}"\nratio=1\n')
    env.update(
        PATH=str(binary) + ":" + env["PATH"],
        VIRTUAL_ENV=sys.prefix,
        SFT_TOML_FILE=str(recipe),
        BASE_CHECKPOINT_PATH=str(tmp_path),
        HF_MODEL_PATH=str(tmp_path),
        WAN_VAE_PATH=str(tmp_path / "vae.pth"),
        OUTPUT_ROOT=str(tmp_path / "output"),
        CAPTURE=str(tmp_path / "capture.json"),
        WANDB_API_KEY_FILE=str(tmp_path / "absent_wandb_key"),
    )
    return env


def launch(env, *args, script="examples/launch_sft_action128_wam_nano.sh"):
    return subprocess.run(
        ["bash", script, *args],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("nodes,rank,shard,mode", [(1, 0, 8, "offline"), (16, 15, 16, "online")])
def test_local_and_platform_defaults(launch_env, nodes, rank, shard, mode):
    launch_env.update(MAX_ITER="7")
    if nodes == 1:
        launch_env.update(NNODES="1", DP_SHARD="8", WANDB_MODE="offline")
    if nodes > 1:
        launch_env.update(RANK=str(rank), MASTER_ADDR="192.0.2.1")
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert f"--nnodes={nodes}" in args
    assert f"--node_rank={rank}" in args
    assert f"model.config.parallelism.data_parallel_shard_degree={shard}" in args
    assert "model.config.parallelism.data_parallel_replicate_degree=-1" in args
    assert f"job.wandb_mode={mode}" in args
    assert "trainer.max_iter=7" in args
    assert (Path(launch_env["OUTPUT_ROOT"]) / f"logs/action128_node{rank}.log").exists()
    from cosmos_framework.utils.generator.parallelism import ParallelDims

    dims = ParallelDims(world_size=nodes * 8, dp_shard=shard, dp_replicate=-1)
    assert dims.dp_shard == shard
    assert dims.dp_replicate == (1 if nodes == 1 else 8)


def test_elastic_and_cli_precedence(launch_env):
    launch_env.update(NNODES="16", RDZV_ENDPOINT="192.0.2.1:29633", RDZV_ID="job42", MAX_ITER="7")
    result = launch(launch_env, "trainer.max_iter=9", "job.wandb_mode=offline")
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert "--rdzv-id=job42" in args
    assert args[-2:] == ["trainer.max_iter=9", "job.wandb_mode=offline"]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(NNODES="0"),
        dict(NNODES="1", DP_SHARD="16"),
        dict(DP_SHARD="0"),
        dict(NNODES="16"),
        dict(NNODES="16", MASTER_ADDR="192.0.2.1", RANK="16"),
        dict(MAX_ITER="0"),
        dict(NNODES="16", RDZV_ENDPOINT="192.0.2.1:29633"),
    ],
)
def test_bad_topology_fails_before_torchrun(launch_env, overrides):
    launch_env.update(overrides)
    assert launch(launch_env).returncode != 0
    assert not Path(launch_env["CAPTURE"]).exists()


@pytest.mark.parametrize("nodes,rank", [(12, 0), (12, 11), (20, 19)])
def test_production_topology_replaces_platform_rendezvous_and_forwards_overrides(launch_env, nodes, rank):
    if nodes != 12:
        launch_env["NNODES"] = str(nodes)
    launch_env.update(
        RANK=str(rank),
        MASTER_ADDR="192.0.2.1",
        MASTER_PORT="50012",
        JOB_ID="action160-test-job",
        RDZV_ENDPOINT="stale-worker:23456",
        RDZV_ID="none",
        RDZV_CONF="join_timeout=600",
        EXTRA_TAIL_OVERRIDES="job.wandb_mode=online trainer.compile_cache.path=/tmp/nano-action-v1",
    )
    result = launch(launch_env, "job.wandb_mode=offline", script="examples/launch_sft_action128_wam_nano_160gpu.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    for value in (
        f"--nnodes={nodes}",
        "--nproc_per_node=8",
        "--rdzv-endpoint=192.0.2.1:50012",
        "--rdzv-id=action160-test-job",
        "--rdzv-conf=join_timeout=7200,last_call_timeout=60",
    ):
        assert value in args
    assert args[-3:] == [
        "job.wandb_mode=online",
        "trainer.compile_cache.path=/tmp/nano-action-v1",
        "job.wandb_mode=offline",
    ]
    bootstrap = Path(launch_env["OUTPUT_ROOT"]) / f"logs/action128_bootstrap_node{rank}.log"
    assert "RDZV_ID=action160-test-job" in bootstrap.read_text()
    assert "trainer.compile_cache.path=/tmp/nano-action-v1" in bootstrap.read_text()


def test_160gpu_rejects_placeholder_job_id(launch_env):
    launch_env.update(RANK="0", MASTER_ADDR="192.0.2.1", JOB_ID="none")
    result = launch(launch_env, script="examples/launch_sft_action128_wam_nano_160gpu.sh")
    assert result.returncode != 0
    assert "JOB_ID must identify this job" in result.stdout + result.stderr
    assert not Path(launch_env["CAPTURE"]).exists()


def test_single_node_ignores_inherited_rendezvous(launch_env):
    launch_env.update(NNODES="1", DP_SHARD="8", RDZV_ENDPOINT="stale:23456", RDZV_ID="none")
    result = launch(launch_env, script="examples/launch_sft_action128_wam_nano_160gpu.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert not any(arg.startswith("--rdzv-") for arg in args)


@pytest.mark.parametrize("disable", ["1", "0"])
def test_160gpu_thp_policy_reaches_child_without_changing_parent(launch_env, disable):
    import ctypes

    libc = ctypes.CDLL(None)
    parent_policy = libc.prctl(42, 0, 0, 0, 0)  # PR_GET_THP_DISABLE
    binary = Path(launch_env["PATH"].split(":")[0]) / "torchrun"
    binary.write_text(
        f"#!{sys.executable}\nimport ctypes,os\n"
        'open(os.environ["CAPTURE"], "w").write(str(ctypes.CDLL(None).prctl(42, 0, 0, 0, 0)))\n'
    )
    launch_env.update(NNODES="1", DP_SHARD="8", ACTION128_DISABLE_THP=disable)
    result = launch(launch_env, script="examples/launch_sft_action128_wam_nano_160gpu.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert int(Path(launch_env["CAPTURE"]).read_text()) == (1 if disable == "1" else parent_policy)
    assert libc.prctl(42, 0, 0, 0, 0) == parent_policy


@pytest.mark.parametrize("explicit", [False, True])
def test_shared_wandb_credential_is_private_and_env_wins(launch_env, tmp_path, explicit):
    secret = tmp_path / "wandb_key"
    secret.write_text("test-file-credential\n")
    fake_bin = Path(launch_env["PATH"].split(":")[0])
    wandb = fake_bin / "wandb"
    wandb.write_text("#!/bin/sh\nexit 99\n")  # No eager login should be invoked.
    wandb.chmod(0o755)
    expected = "test-env-credential" if explicit else "test-file-credential"
    (fake_bin / "torchrun").write_text(
        f"#!{sys.executable}\nimport os\n"
        f"assert os.environ['WANDB_API_KEY'] == {expected!r}\n"
        "assert os.environ['WANDB_ENTITY'] == 'test-entity'\n"
    )
    launch_env.update(
        NNODES="1",
        DP_SHARD="8",
        WANDB_API_KEY_FILE=str(secret),
        WANDB_ENTITY="test-entity",
        WANDB_PERSIST_LOGIN="0",
    )
    if explicit:
        launch_env["WANDB_API_KEY"] = expected
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    for content in [
        result.stdout,
        result.stderr,
        *(p.read_text() for p in (Path(launch_env["OUTPUT_ROOT"]) / "logs").glob("*.log")),
    ]:
        assert "test-file-credential" not in content
        assert "test-env-credential" not in content
