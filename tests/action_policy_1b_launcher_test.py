"""Verify topology/batch contracts through the real shell entry without GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib


@pytest.fixture
def launch_env(tmp_path):
    env = os.environ.copy()
    for key in (
        "NNODES",
        "NPROC_PER_NODE",
        "NODE_RANK",
        "MASTER_ADDR",
        "RDZV_ENDPOINT",
        "RDZV_ID",
        "GLOBAL_BATCH_SIZE",
        "ACTION_SAMPLES_PER_PACK",
        "ACTION_TOML",
        "WANDB_API_KEY",
        "LOG_FILENAME",
        "IMAGINAIRE_OUTPUT_ROOT",
        "SFT_TRAINING_MODULE",
        "ACTION_REPLAY_ROOT",
        "REQUIRE_WAN_VAE",
    ):
        env.pop(key, None)
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "torchrun"
    mock.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    mock.chmod(0o755)
    checkpoint = tmp_path / "checkpoint/model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").touch()
    for name in ("catalog.parquet", "vae.pth", "fixture.pt"):
        (tmp_path / name).touch()
    env.update(
        PATH=str(binary) + ":" + env["PATH"],
        DATASET_PATH=str(tmp_path / "catalog.parquet"),
        BASE_CHECKPOINT_PATH=str(checkpoint.parent),
        HF_MODEL_PATH=str(tmp_path),
        WAN_VAE_PATH=str(tmp_path / "vae.pth"),
        ACTION_EVAL_FIXTURE=str(tmp_path / "fixture.pt"),
        OUTPUT_ROOT=str(tmp_path / "output"),
        CAPTURE=str(tmp_path / "args.json"),
        GLOBAL_BATCH_SIZE="32",
        ACTION_SAMPLES_PER_PACK="1",
    )
    return env


def launch(env):
    return subprocess.run(
        ["bash", "examples/launch_sft_action_policy_1b.sh", "trainer.max_iter=2"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("nodes,accum", [(1, 4), (2, 2), (4, 1)])
def test_static_topology_preserves_global_batch(launch_env, nodes, accum):
    launch_env.update(NNODES=str(nodes), NODE_RANK=str(nodes - 1), MASTER_ADDR="192.0.2.1")
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert "--nproc_per_node=8" in args
    assert f"--nnodes={nodes}" in args
    assert f"--node_rank={nodes - 1}" in args
    assert f"model.config.parallelism.data_parallel_shard_degree={8 * nodes}" in args
    assert f"trainer.grad_accum_iter={accum}" in args
    assert "++trainer.callbacks.small_action_audit.expected_global_batch_size=32" in args
    assert args[-1] == "trainer.max_iter=2"
    assert (Path(launch_env["OUTPUT_ROOT"]) / f"logs/action_policy_1b_node{nodes - 1}_sft.log").exists()


def test_elastic_rendezvous_and_explicit_larger_batch(launch_env):
    launch_env.update(NNODES="8", GLOBAL_BATCH_SIZE="64", RDZV_ENDPOINT="192.0.2.1:29400", RDZV_ID="test-run")
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert "--rdzv-endpoint=192.0.2.1:29400" in args
    assert "--rdzv-id=test-run" in args
    assert "model.config.parallelism.data_parallel_shard_degree=64" in args
    assert "trainer.grad_accum_iter=1" in args


@pytest.mark.parametrize(
    "override",
    [
        {"NNODES": "2"},
        {"NNODES": "2", "NODE_RANK": "2", "MASTER_ADDR": "192.0.2.1"},
        {"NNODES": "8"},
        {"GLOBAL_BATCH_SIZE": "17"},
        {"ACTION_SAMPLES_PER_PACK": "3"},
        {"ACTION_SAMPLES_PER_PACK": "0"},
        {"NNODES": "0"},
        {"RDZV_ENDPOINT": "192.0.2.1:29400"},
    ],
)
def test_invalid_launch_fails_before_torchrun(launch_env, override):
    launch_env.update(override)
    result = launch(launch_env)
    assert result.returncode != 0
    assert not Path(launch_env["CAPTURE"]).exists()


@pytest.mark.parametrize("samples,accum", [(1, 4), (2, 2), (4, 1)])
def test_packing_preserves_global_sample_batch(launch_env, samples, accum):
    launch_env["ACTION_SAMPLES_PER_PACK"] = str(samples)
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert f"dataloader_train.max_samples_per_batch={samples}" in args
    assert f"trainer.grad_accum_iter={accum}" in args


def test_production_defaults_select_final_recipe(launch_env):
    launch_env.pop("GLOBAL_BATCH_SIZE")
    launch_env.pop("ACTION_SAMPLES_PER_PACK")
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    recipe = Path(__file__).resolve().parents[1] / "examples/toml/sft_config/action_policy_1b.toml"
    assert tomllib.loads(recipe.read_text())["dataloader_train"]["visual_generation"]["vae_mode"] == "offline"
    assert tomllib.loads(recipe.read_text())["model"]["action_tower_und_short_tokens"] == 5120
    assert tomllib.loads(recipe.read_text())["model"]["action_tower_attention_backend"] == "flash3_video"
    assert f"--sft-toml={recipe}" in args
    assert "dataloader_train.max_samples_per_batch=92" in args
    assert "trainer.seed=42" in args
    assert "trainer.grad_accum_iter=1" in args
    assert "++trainer.callbacks.small_action_audit.expected_global_batch_size=736" in args
    assert args[-1] == "trainer.max_iter=2"


def test_production_requires_explicit_offline_dataset(launch_env):
    launch_env.pop("DATASET_PATH")
    result = launch(launch_env)
    assert result.returncode != 0
    assert "prepared offline.parquet" in result.stderr
    assert not Path(launch_env["CAPTURE"]).exists()


def test_offline_launch_does_not_require_local_vae_weights(launch_env):
    Path(launch_env["WAN_VAE_PATH"]).unlink()
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", [None, "tools.action_input_replay"])
def test_optional_diagnostic_entry_preserves_production_default(launch_env, module):
    if module:
        launch_env["SFT_TRAINING_MODULE"] = module
    result = launch(launch_env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(launch_env["CAPTURE"]).read_text())
    assert args[args.index("-m") + 1] == (module or "cosmos_framework.scripts.train")
