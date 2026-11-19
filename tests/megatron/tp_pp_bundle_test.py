# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical bundle import preserves source masters and optimizer fragments."""

import copy
import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from tp_pp_conversion_test import make_source

from cosmos_framework.training_backends.megatron_tp_pp import bundle as bundle_module
from cosmos_framework.training_backends.megatron_tp_pp.bundle import CanonicalBundle
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import AdamParameterState
from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import ParameterLayout
from tools.convert_cosmos_checkpoint_to_megatron_tp_pp import convert


def test_append_only_normalization_matches_native_extend(tmp_path, monkeypatch):
    from cosmos_framework.checkpoint.action128_test import _extended_spec
    from cosmos_framework.model.generator.mot.action128 import Action128Normalization
    from cosmos_framework.model.generator.mot.action_tower_test import tiny_action128_spec

    source, destination = tmp_path / "source", tmp_path / "converted"
    old_spec, current_spec = tiny_action128_spec(), _extended_spec()
    old = Action128Normalization(old_spec, "action128_seam")
    model = {"net.action_normalization." + key: value for key, value in old.state_dict().items()}
    contract = {"version": 1, "spec": old_spec, "vision_spatial_position_mode": "action128_seam"}
    for name, state in {
        "model": model,
        "optim": {},
        "trainer": {"action128_contract": json.dumps(contract)},
        "scheduler": {},
    }.items():
        dcp.save(state, checkpoint_id=source / name)
    (source / "dataloader").mkdir()
    (source / "dataloader" / "rank_0.pkl").write_bytes(pickle.dumps({"global_id": 0}))
    (source / ".save_complete").touch()
    convert(source, destination, tp=2, pp=2, layers=2)
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    current = {**contract, "spec": current_spec}
    _, changed = bundle.validate_action128_extension(current)
    assert changed
    stage = nn.Module()
    stage.input_adapter = nn.Module()
    stage.input_adapter.action_normalization = Action128Normalization(current_spec, "action128_seam").to_empty(
        device="cpu"
    )
    manifest = {}
    for name, value in stage.named_buffers():
        canonical = "net." + name.removeprefix("input_adapter.")
        manifest[name] = ParameterLayout(canonical, name, tuple(value.shape), None, 0, tuple(value.shape), 0, 0, False)
    monkeypatch.setattr(bundle_module, "stage_parameter_manifest", lambda stage: manifest)
    bundle.load_model(stage)
    expected = Action128Normalization(current_spec, "action128_seam")
    for key, value in stage.input_adapter.action_normalization.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    invalid = copy.deepcopy(current)
    index = invalid["spec"]["registry"]["domains"].index(old_spec["registry"]["domains"][0])
    invalid["spec"]["transforms"][index]["scale"][0] = 2.0
    with pytest.raises(ValueError, match="changed or removed"):
        bundle.validate_action128_extension(invalid)


def test_stage_bundle_loads_model_and_true_dp_optimizer_fragment(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    topology = ModelParallelConfig()
    bundle = CanonicalBundle(destination, topology)
    stage, manifest = nn.Module(), {}
    stage.input_adapter = None
    for index, (name, saved) in enumerate(converted["parameters"].items()):
        if saved["pipeline_rank"] != 0:
            continue
        local = f"param{index}"
        shape = list(tensors[name].shape)
        shape[0] //= 2
        trainable = bool(saved["optimizer_fields"])
        stage.register_parameter(local, nn.Parameter(torch.empty(shape), requires_grad=trainable))
        manifest[local] = ParameterLayout(
            name, local, tuple(tensors[name].shape), 0, shape[0], tuple(shape), 0, 1, trainable
        )
    monkeypatch.setattr(bundle_module, "stage_parameter_manifest", lambda stage: manifest)
    bundle.load_model(stage)
    for local, layout in manifest.items():
        torch.testing.assert_close(
            stage.get_parameter(local), layout.select(tensors[layout.canonical_name]), rtol=0, atol=0
        )
    local, layout = next((key, value) for key, value in manifest.items() if value.trainable)
    fragment = AdamParameterState(local, stage.get_parameter(local), torch.empty(10), {}, 4, 14, {}, True)
    monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
    config = SimpleNamespace(adam_beta1=0.9, adam_beta2=0.99, adam_eps=1e-8, weight_decay=0.05)
    overrides = bundle.optimizer_overrides(stage, config)
    assert len(overrides) == 1
    bundle.load_optimizer(stage, None)
    for key in ("master_param", "exp_avg", "exp_avg_sq"):
        actual = fragment.master if key == "master_param" else fragment.state[key]
        expected = fragment.select(layout.select(optim[f"state.{layout.canonical_name}.{key}"]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fragment.step == 1200
    with pytest.raises(ValueError, match="topology"):
        CanonicalBundle(destination, replace(topology, pipeline_model_parallel_size=1))
    (destination / ".save_complete").unlink()
    with pytest.raises(ValueError, match="complete"):
        CanonicalBundle(destination, topology)
