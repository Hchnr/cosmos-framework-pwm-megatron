# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The math diagnostic must preserve its requested seed and TP4 topology."""

import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tp_pp_full_source_math as helper


@pytest.mark.parametrize(("tp", "pp", "microbatches"), [(1, 2, 1), (2, 2, 2), (4, 1, 2)])
def test_four_gpu_diagnostic_topology(tp, pp, microbatches):
    topology = helper.diagnostic_topology(tp)
    assert topology.pipeline_model_parallel_size == pp
    assert topology.microbatches(4) == microbatches


def test_explicit_pp1_topology_uses_two_data_parallel_ranks_for_tp2():
    topology = helper.diagnostic_topology(2, 1)
    assert topology.pipeline_model_parallel_size == 1
    assert topology.data_parallel_size(4) == 2
    assert topology.microbatches(4) == 1


def test_native_reference_requires_the_requested_seed():
    helper.validate_reference_seed({"seed": 20261120}, 20261120)
    for reference in ({}, {"seed": 42}, {"seed": True}, {"seed": "20261120"}):
        with pytest.raises(ValueError, match="same requested seed"):
            helper.validate_reference_seed(reference, 20261120)


def test_cli_seeds_host_rng_before_native_configuration(monkeypatch):
    seen = []

    def configuration(*args, **kwargs):
        seen.append((random.random(), np.random.random(), torch.rand(()).item()))
        return None, None, None, None

    monkeypatch.setattr(helper, "configuration", configuration)
    monkeypatch.setattr(helper, "real_crop", lambda path: path)
    monkeypatch.setattr(helper, "native", lambda *args: None)
    command = [
        "tp_pp_full_source_math.py",
        "--mode",
        "native",
        "--bundle",
        "bundle",
        "--toml",
        "config",
        "--prepared-dir",
        "inputs",
        "--reference-dir",
        "reference",
        "--output",
        "output",
        "--seed",
    ]
    for seed in ("20261120", "20261120", "42"):
        monkeypatch.setattr(sys, "argv", [*command, seed])
        helper.main()
    assert seen[0] == seen[1]
    assert seen[2] != seen[0]


def test_cli_accepts_explicit_tp4_and_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(helper, "configuration", lambda *args, **kwargs: (None, None, None, None))
    monkeypatch.setattr(helper, "real_crop", lambda path: path)
    monkeypatch.setattr(helper, "tp", lambda args, *rest: seen.append((args.tp, args.pp, args.seed)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tp_pp_full_source_math.py",
            "--mode",
            "tp",
            "--tp",
            "4",
            "--pp",
            "1",
            "--seed",
            "20261120",
            "--bundle",
            "bundle",
            "--toml",
            "config",
            "--prepared-dir",
            "inputs",
            "--reference-dir",
            "reference",
            "--output",
            "output",
        ],
    )
    helper.main()
    assert seen == [(4, 1, 20261120)]


def test_tp_passes_requested_seed_to_model_parallel_initialization(monkeypatch, tmp_path):
    from cosmos_framework.training_backends.megatron_tp_pp import runtime

    class StopAfterInitialization(Exception):
        pass

    observed = []

    def initialize(topology, *, seed):
        observed.append((topology.tensor_model_parallel_size, topology.pipeline_model_parallel_size, seed))
        raise StopAfterInitialization

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "initialize", initialize)
    with pytest.raises(StopAfterInitialization):
        helper.tp(SimpleNamespace(tp=4, seed=20261120, reference_dir=tmp_path), None, None, None, None)
    assert observed == [(4, 1, 20261120)]


def test_tp_rejects_unseeded_reference_before_cuda_initialization(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    called = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: called.append(device))
    with pytest.raises(ValueError, match="same requested seed"):
        helper.tp(SimpleNamespace(tp=4, seed=20261120, reference_dir=tmp_path), None, None, None, None)
    assert not called
