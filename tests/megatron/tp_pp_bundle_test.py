# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical bundle import preserves native model and optimizer fragments."""

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
        source = (
            tensors[layout.canonical_name] if key == "master_param" else optim[f"state.{layout.canonical_name}.{key}"]
        )
        expected = fragment.select(layout.select(source))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fragment.step == 1200
    read = bundle.read
    for corruption in ("master_dtype", "master_shape", "moment_dtype"):

        def read_corrupted(name, rank):
            saved = read(name, rank)
            saved["state"] = dict(saved["state"])
            if corruption == "master_dtype":
                saved["state"]["master_param"] = saved["state"]["master_param"].bfloat16()
            elif corruption == "master_shape":
                saved["state"]["master_param"] = saved["state"]["master_param"].view(-1)[:-1]
            else:
                saved["state"]["exp_avg"] = saved["state"]["exp_avg"].bfloat16()
            return saved

        monkeypatch.setattr(bundle, "read", read_corrupted)
        with pytest.raises(ValueError, match="FP32"):
            bundle.load_optimizer(stage, None)
    monkeypatch.setattr(bundle, "read", read)
    with pytest.raises(ValueError, match="topology"):
        CanonicalBundle(destination, replace(topology, pipeline_model_parallel_size=1))
    (destination / ".save_complete").unlink()
    with pytest.raises(ValueError, match="complete"):
        CanonicalBundle(destination, topology)


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("alias", [False, True])
def test_fp32_import_uses_loaded_model_for_alias_and_separate_main(tmp_path, monkeypatch, dp, alias):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    canonical, saved = next(
        (name, info)
        for name, info in converted["parameters"].items()
        if info["pipeline_rank"] == 0 and info["optimizer_fields"]
    )
    shape = tuple(saved["shards"][0]["shape"])
    layout = ParameterLayout(canonical, "param", tuple(tensors[canonical].shape), 0, 0, shape, 0, 0, True)
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    monkeypatch.setattr(bundle, "_stage_manifest", lambda stage: {"param": layout})
    results = []
    for rank in range(dp):
        stage = nn.Module()
        stage.input_adapter = None
        stage.param = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        bundle.load_model(stage)
        before = stage.param.detach().clone()
        assert not torch.equal(before, layout.select(optim[f"state.{canonical}.master_param"]))
        start, end = before.numel() * rank // dp, before.numel() * (rank + 1) // dp
        main = stage.param.detach().view(-1)[start:end] if alias else torch.full((end - start,), -1000.0)
        fragment = AdamParameterState("param", stage.param, main, {}, start, end, {}, True)
        monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
        bundle.load_optimizer(stage, None)
        torch.testing.assert_close(stage.param, before, rtol=0, atol=0)
        torch.testing.assert_close(fragment.master, fragment.select(before), rtol=0, atol=0)
        for field in ("exp_avg", "exp_avg_sq"):
            expected = fragment.select(layout.select(optim[f"state.{canonical}.{field}"]))
            torch.testing.assert_close(fragment.state[field], expected, rtol=0, atol=0)
        assert fragment.step == 1200
        results.append(stage.param.detach().clone())
    assert all(torch.equal(value, results[0]) for value in results)


def test_bf16_import_preserves_checkpoint_fp32_master_and_compute(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    canonical, saved = next(
        (name, info)
        for name, info in converted["parameters"].items()
        if info["pipeline_rank"] == 0 and info["optimizer_fields"]
    )
    shape = tuple(saved["shards"][0]["shape"])
    layout = ParameterLayout(canonical, "param", tuple(tensors[canonical].shape), 0, 0, shape, 0, 0, True)
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    monkeypatch.setattr(bundle, "_stage_manifest", lambda stage: {"param": layout})
    stage = nn.Module()
    stage.input_adapter = None
    stage.param = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16))
    bundle.load_model(stage)
    before = stage.param.detach().clone()
    fragment = AdamParameterState("param", stage.param, torch.empty(10), {}, 4, 14, {}, True)
    monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
    bundle.load_optimizer(stage, None)
    torch.testing.assert_close(stage.param, before, rtol=0, atol=0)
    expected_master = fragment.select(layout.select(optim[f"state.{canonical}.master_param"]))
    torch.testing.assert_close(fragment.master, expected_master, rtol=0, atol=0)
    assert not torch.equal(fragment.master, fragment.select(before.float()))
    assert fragment.step == 1200


def test_fp32_diagnostic_preloaded_master_is_preserved(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    canonical, saved = next(
        (name, info)
        for name, info in converted["parameters"].items()
        if info["pipeline_rank"] == 0 and info["optimizer_fields"]
    )
    shape = tuple(saved["shards"][0]["shape"])
    layout = ParameterLayout(canonical, "param", tuple(tensors[canonical].shape), 0, 0, shape, 0, 0, True)
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    monkeypatch.setattr(bundle, "_stage_manifest", lambda stage: {"param": layout})
    stage = nn.Module()
    stage.input_adapter = None
    stage.param = nn.Parameter(torch.empty(shape))
    bundle.load_model(stage)
    with torch.no_grad():
        stage.param.copy_(layout.select(optim[f"state.{canonical}.master_param"]))
    before = stage.param.detach().clone()
    fragment = AdamParameterState("param", stage.param, torch.empty(10), {}, 4, 14, {}, True)
    monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
    bundle.load_optimizer(stage, None)
    torch.testing.assert_close(stage.param, before, rtol=0, atol=0)
    torch.testing.assert_close(fragment.master, fragment.select(before), rtol=0, atol=0)


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("alias", [False, True])
def test_fp32_source_without_master_restores_model_moments_and_step(tmp_path, monkeypatch, dp, alias):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, optim = make_source(source, include_master=False, fractional=True)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    canonical, saved = next(
        (name, info)
        for name, info in converted["parameters"].items()
        if info["pipeline_rank"] == 0 and info["optimizer_fields"]
    )
    shape = tuple(saved["shards"][0]["shape"])
    layout = ParameterLayout(canonical, "param", tuple(tensors[canonical].shape), 0, 0, shape, 0, 0, True)
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    assert bundle.manifest["parameters"][canonical]["master_origin"] == "model_fp32"
    bundle.validate_target_precision(torch.float32)
    monkeypatch.setattr(bundle, "_stage_manifest", lambda stage: {"param": layout})
    models = []
    for rank in range(dp):
        stage = nn.Module()
        stage.input_adapter = None
        stage.param = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        bundle.load_model(stage)
        before = stage.param.detach().clone()
        start, end = before.numel() * rank // dp, before.numel() * (rank + 1) // dp
        main = stage.param.detach().view(-1)[start:end] if alias else torch.full((end - start,), -1000.0)
        fragment = AdamParameterState("param", stage.param, main, {}, start, end, {}, True)
        monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
        bundle.load_optimizer(stage, None)
        torch.testing.assert_close(stage.param, before, rtol=0, atol=0)
        torch.testing.assert_close(fragment.master, fragment.select(before), rtol=0, atol=0)
        for field in ("exp_avg", "exp_avg_sq"):
            expected = fragment.select(layout.select(optim[f"state.{canonical}.{field}"]))
            torch.testing.assert_close(fragment.state[field], expected, rtol=0, atol=0)
        assert fragment.step == 1200
        models.append(stage.param.detach().clone())
    assert all(torch.equal(model, models[0]) for model in models)


def test_fp32_source_without_master_rejects_bf16_target_before_restoring_optimizer(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "converted"
    tensors, _ = make_source(source, include_master=False, fractional=True)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    canonical, saved = next(
        (name, info)
        for name, info in converted["parameters"].items()
        if info["pipeline_rank"] == 0 and info["optimizer_fields"]
    )
    shape = tuple(saved["shards"][0]["shape"])
    layout = ParameterLayout(canonical, "param", tuple(tensors[canonical].shape), 0, 0, shape, 0, 0, True)
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    with pytest.raises(ValueError, match="requires FP32"):
        bundle.validate_target_precision(torch.bfloat16)
    monkeypatch.setattr(bundle, "_stage_manifest", lambda stage: {"param": layout})
    stage = nn.Module()
    stage.input_adapter = None
    stage.param = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16))
    bundle.load_model(stage)
    before = stage.param.detach().clone()
    main = torch.full((10,), -1.0)
    fragment = AdamParameterState("param", stage.param, main, {}, 4, 14, {}, True)
    monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
    with pytest.raises(ValueError, match="requires FP32"):
        bundle.load_optimizer(stage, None)
    assert torch.equal(stage.param, before) and torch.equal(main, torch.full((10,), -1.0))
    assert not fragment.state


def test_bundle_rejects_forged_master_provenance(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    make_source(source, include_master=False)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    active = next(name for name, entry in converted["parameters"].items() if entry["optimizer_fields"])
    converted["parameters"][active]["master_origin"] = "optimizer"
    (destination / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        CanonicalBundle(destination, ModelParallelConfig())


def test_bundle_rejects_missing_hyperparameter_manifest_field(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    make_source(source, include_master=False)
    converted = convert(source, destination, tp=2, pp=2, layers=2)
    active = next(name for name, entry in converted["parameters"].items() if entry["optimizer_fields"])
    converted["parameters"][active]["hyperparameter_fields"].remove(f"param_groups.{active}.step")
    (destination / "manifest.json").write_text(json.dumps(converted, indent=2) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        CanonicalBundle(destination, ModelParallelConfig())


@pytest.mark.parametrize("initial_step,valid", [(None, True), (0.0, True), (0.5, False)])
def test_native_fp32_python_step_dcp_conversion_and_restore(tmp_path, monkeypatch, initial_step, valid):
    """Exercise real non-capturable FusedAdam state/flattening, without CUDA kernels."""
    import transformer_engine as te
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    from cosmos_framework.utils.generator.fused_adam import FusedAdam
    from tools.verify_cosmos_tp_pp_conversion import verify

    source, destination = tmp_path / "source", tmp_path / "converted"
    canonical = "net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"
    model = nn.Module()
    model.net = nn.Module()
    model.net.language_model = nn.Module()
    model.net.language_model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj_moe_gen = nn.Linear(12, 8, bias=False)
    model.net.language_model.model.layers = nn.ModuleList([layer])
    parameter = next(model.parameters())
    assert list(dict(model.named_parameters())) == [canonical]

    original_tensor = torch.tensor
    calls = []

    def cpu_tensor(*args, **kwargs):
        if str(kwargs.get("device")) == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", cpu_tensor)
        patch.setattr(te.pytorch.optimizers, "multi_tensor_applier", lambda *args, **kwargs: calls.append(True))
        optimizer = FusedAdam([parameter], lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05, capturable=False)
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if initial_step is not None:
            optimizer.param_groups[0]["step"] = initial_step
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert calls and set(optimizer.state[parameter]) == {"exp_avg", "exp_avg_sq"}
    flattened = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True))
    step = flattened[f"param_groups.{canonical}.step"]
    assert type(step) is (int if initial_step is None else float)
    for component, state in {
        "model": model.state_dict(),
        "optim": flattened,
        "trainer": {"iteration": 1},
        "scheduler": {"last_epoch": 1},
    }.items():
        dcp.save(state, checkpoint_id=source / component)
    (source / "dataloader").mkdir()
    (source / "dataloader/rank_0.pkl").write_bytes(pickle.dumps({"global_id": 1}))
    (source / ".save_complete").touch()
    if not valid:
        with pytest.raises(ValueError, match="Invalid optimizer step"):
            convert(source, destination, tp=2, pp=2, layers=2)
        assert not (destination / ".save_complete").exists()
        return

    converted = convert(source, destination, tp=2, pp=2, layers=2)
    assert converted["parameters"][canonical]["master_origin"] == "model_fp32"
    assert verify(destination)["passed"]
    bundle = CanonicalBundle(destination, ModelParallelConfig())
    with pytest.raises(ValueError, match="requires FP32"):
        bundle.validate_target_precision(torch.bfloat16)
    shape = tuple(converted["parameters"][canonical]["shards"][0]["shape"])
    layout = ParameterLayout(canonical, "param", tuple(parameter.shape), 0, 0, shape, 0, 0, True)
    monkeypatch.setattr(bundle, "_stage_manifest", lambda stage: {"param": layout})
    stage = nn.Module()
    stage.input_adapter = None
    stage.param = nn.Parameter(torch.empty(shape))
    bundle.load_model(stage)
    before = stage.param.detach().clone()
    fragment = AdamParameterState("param", stage.param, torch.empty(10), {}, 4, 14, {}, True)
    monkeypatch.setattr(bundle_module, "adam_parameter_states", lambda stage, optimizer: [fragment])
    config = SimpleNamespace(adam_beta1=0.9, adam_beta2=0.99, adam_eps=1e-8, weight_decay=0.05)
    assert len(bundle.optimizer_overrides(stage, config)) == 1
    bundle.load_optimizer(stage, None)
    assert torch.equal(stage.param, before)
    assert torch.equal(fragment.master, fragment.select(before))
    for field in ("exp_avg", "exp_avg_sq"):
        expected = layout.select(flattened[f"state.{canonical}.{field}"])
        assert torch.equal(fragment.state[field], fragment.select(expected))
    assert fragment.step == int(step)

    # A corrupted converted shard must fail before optimizer state changes,
    # even though legitimate source conversion already validates the step.
    shard_path = destination / converted["parameters"][canonical]["shards"][0]["path"]
    saved = torch.load(shard_path, weights_only=False)
    state_before = {field: value.clone() for field, value in fragment.state.items()}
    group_before = dict(fragment.group)
    main_before = fragment.master.clone()
    for invalid_step in (1.5, -1, float("nan"), True, torch.tensor([1, 2])):
        corrupted = dict(saved)
        corrupted["hyperparameters"] = {**saved["hyperparameters"], "step": invalid_step}
        torch.save(corrupted, shard_path)
        with pytest.raises(ValueError, match="Invalid Adam step"):
            bundle.optimizer_overrides(stage, config)
        with pytest.raises(ValueError, match="Invalid Adam step"):
            bundle.load_optimizer(stage, None)
        assert fragment.group == group_before
        assert torch.equal(fragment.master, main_before)
        assert all(torch.equal(fragment.state[field], value) for field, value in state_before.items())
