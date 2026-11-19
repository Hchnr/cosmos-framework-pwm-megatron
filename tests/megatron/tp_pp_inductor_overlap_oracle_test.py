# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.qualify_tp_pp_inductor_overlap import create_rank_cache, install_oracle_compile
from tools.run_tp_pp_inductor_overlap import OPTION_ENABLED, OPTION_PASSES, PASSES, Observation


def test_rank_cache_is_fresh_and_isolated(tmp_path):
    left, right = {}, {}
    cache0 = create_rank_cache(tmp_path / "cache", "0", left)
    cache1 = create_rank_cache(tmp_path / "cache", "1", right)
    assert cache0 != cache1 and cache0.is_dir() and cache1.is_dir()
    assert left["TORCHINDUCTOR_CACHE_DIR"] == str(cache0)
    assert left["TRITON_CACHE_DIR"] == str(cache0 / "triton")
    assert left["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] == left["TORCH_COMPILE_FORCE_DISABLE_CACHES"] == "1"
    benchmark_env = {}
    create_rank_cache(tmp_path / "benchmark", "0", benchmark_env, disable_caches=False)
    assert "TORCHINDUCTOR_FORCE_DISABLE_CACHES" not in benchmark_env
    assert "TORCH_COMPILE_FORCE_DISABLE_CACHES" not in benchmark_env
    sentinel = cache0 / "preserve"
    sentinel.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        create_rank_cache(tmp_path / "cache", "0", {})
    assert sentinel.read_text() == "existing evidence"
    with pytest.raises(ValueError):
        create_rank_cache(tmp_path / "cache", "../other", {})


def test_only_oracle_checkpoint_blocks_receive_overlap_options(tmp_path):
    class Block:
        pass

    calls = []

    def original(model, **kwargs):
        calls.append((model, kwargs))
        return model

    fake = SimpleNamespace(
        compile=original,
        _inductor=SimpleNamespace(config=SimpleNamespace(deterministic=False, force_disable_caches=True)),
        compiler=SimpleNamespace(config=SimpleNamespace(force_disable_caches=True)),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    observation = Observation(tmp_path / "rank0.json")
    install_oracle_compile(observation, torch_module=fake, wrapper_type=Block)
    ordinary = object()
    assert fake.compile(ordinary, fullgraph=True) is ordinary
    assert "options" not in calls[-1][1]
    target = Block()
    assert fake.compile(target, fullgraph=True, options={"existing": 5}) is target
    options = calls[-1][1]["options"]
    assert options["existing"] == 5 and options[OPTION_ENABLED] is True
    assert options[OPTION_PASSES] == list(PASSES)
    assert "force_disable_caches" not in options
    observation.finish()
    report = json.loads(observation.target.read_text())
    assert report["options_installed"] and report["qualification_caches_disabled"]
    # Merely installing options or compiling is never acceptance evidence.
    assert not report["effect_verified"] and report["scheduler_calls_with_collectives"] == 0


def test_oracle_rejects_determinism_before_compiling(tmp_path):
    class Block:
        pass

    calls = []
    fake = SimpleNamespace(
        compile=lambda *a, **kw: calls.append((a, kw)),
        _inductor=SimpleNamespace(config=SimpleNamespace(deterministic=True, force_disable_caches=True)),
        compiler=SimpleNamespace(config=SimpleNamespace(force_disable_caches=True)),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    observation = Observation(tmp_path / "rank0.json")
    install_oracle_compile(observation, torch_module=fake, wrapper_type=Block)
    with pytest.raises(ValueError, match="deterministic"):
        fake.compile(Block())
    assert not calls and not observation.data["options_installed"]


def test_cache_disable_must_be_active_before_compiling(tmp_path):
    fake = SimpleNamespace(
        compile=lambda *a, **kw: None,
        _inductor=SimpleNamespace(config=SimpleNamespace(deterministic=False, force_disable_caches=False)),
        compiler=SimpleNamespace(config=SimpleNamespace(force_disable_caches=False)),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    observation = Observation(tmp_path / "rank0.json")
    with pytest.raises(RuntimeError, match="did not activate"):
        install_oracle_compile(observation, torch_module=fake, wrapper_type=object)


def test_real_torch_compile_accepts_options_and_executes_on_cpu(tmp_path):
    script = """
import sys
import torch
import torch.distributed as dist
from torch._inductor import config
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from tools.run_tp_pp_inductor_overlap import Observation, PASSES, install
from tools.qualify_tp_pp_inductor_overlap import install_oracle_compile

assert torch.compiler.config.force_disable_caches is True
assert config.force_disable_caches is True
dist.init_process_group("gloo", init_method="file://" + sys.argv[1], rank=0, world_size=1)
try:
    observation = Observation(sys.argv[2])
    install(observation)
    install_oracle_compile(observation)

    class Block(torch.nn.Module):
        def forward(self, value):
            return value * 3 + 1

    # Real oracle wrapper construction validates its actual option dictionary
    # immediately; numerical full-AC execution belongs to the GPU oracle.
    checkpointed = torch.compile(
        checkpoint_wrapper(Block()),
        fullgraph=True,
    )
    assert callable(checkpointed)
    assert observation.data["options_installed"]
    function = torch.compile(
        lambda x: x * 3 + 1,
        options={
            "reorder_for_compute_comm_overlap": True,
            "reorder_for_compute_comm_overlap_passes": list(PASSES),
        },
        fullgraph=True,
    )
    values = torch.arange(4, dtype=torch.float32)
    actual = function(values)
    torch.testing.assert_close(actual, torch.tensor([1., 4., 7., 10.]))
    observation.finish()
    assert observation.data["scheduler_calls"] > 0
    # This CPU graph deliberately has no collective: no false qualification.
    assert not observation.data["effect_verified"]
    assert not torch.cuda.is_initialized()
finally:
    dist.destroy_process_group()
"""
    environ = os.environ.copy()
    environ.update(
        CUDA_VISIBLE_DEVICES="",
        TORCH_COMPILE_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "gloo"), str(tmp_path / "real-torch-report.json")],
        env=environ,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
