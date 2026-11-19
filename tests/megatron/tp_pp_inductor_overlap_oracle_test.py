# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
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
        _inductor=SimpleNamespace(config=SimpleNamespace(deterministic=False)),
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
    assert options["force_disable_caches"] is True
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
        _inductor=SimpleNamespace(config=SimpleNamespace(deterministic=True)),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    observation = Observation(tmp_path / "rank0.json")
    install_oracle_compile(observation, torch_module=fake, wrapper_type=Block)
    with pytest.raises(ValueError, match="deterministic"):
        fake.compile(Block())
    assert not calls and not observation.data["options_installed"]


def test_installed_inductor_accepts_cache_disable_with_overlap_policy():
    from torch._inductor import config

    options = {
        OPTION_ENABLED: True,
        OPTION_PASSES: list(PASSES),
        "force_disable_caches": True,
    }
    with config.patch(options):
        assert config.reorder_for_compute_comm_overlap is True
        assert config.reorder_for_compute_comm_overlap_passes == list(PASSES)
        assert config.force_disable_caches is True
