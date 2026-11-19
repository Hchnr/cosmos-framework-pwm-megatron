# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contract checks for the opt-in norm candidate, not GPU qualification."""

import hashlib
import importlib.util
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture
def candidate(monkeypatch):
    path = Path(__file__).parents[2] / "cosmos_framework/training_backends/megatron_tp_pp/native_fp32_norm.py"
    name = "cosmos_framework.training_backends.megatron_tp_pp.native_fp32_norm"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("flag", "dtype", "dp", "clip", "expected"),
    [("0", torch.bfloat16, 4, 0.0, False), ("1", torch.float32, 1, 0.3, True)],
)
def test_opt_in_and_default_path(candidate, flag, dtype, dp, clip, expected):
    assert candidate.requested(flag, dtype=dtype, dp_size=dp, clip_grad=clip) is expected


@pytest.mark.parametrize(
    ("flag", "dtype", "dp", "clip"),
    [
        ("2", torch.float32, 1, 0.3),
        ("1", torch.bfloat16, 1, 0.3),
        ("1", torch.float32, 2, 0.3),
        ("1", torch.float32, 1, 0.0),
        ("1", torch.float32, 1, float("nan")),
        ("1", torch.float32, 1, float("inf")),
    ],
)
def test_ineligible_opt_in_fails_explicitly(candidate, flag, dtype, dp, clip):
    with pytest.raises(ValueError):
        candidate.requested(flag, dtype=dtype, dp_size=dp, clip_grad=clip)


def test_meta_registration_order_and_coverage_without_reading_values(candidate):
    with torch.device("meta"):
        model = nn.Module()
        model.register_parameter("z", nn.Parameter(torch.empty(3)))
        model.register_parameter("a", nn.Parameter(torch.empty(4)))
        model.child = nn.Linear(4, 3, bias=False)
        model.register_buffer("counter", torch.empty(1))
    entries = {
        "net.child.weight": {"optimizer_fields": ["master_param"]},
        "net.a": {"optimizer_fields": []},
        "net.z": {"optimizer_fields": ["master_param"]},
        "net.counter": {"optimizer_fields": []},
    }
    assert candidate.ordered_trainable_names(model, {"parameters": entries}) == [
        "net.z",
        "net.child.weight",
    ]
    entries["net.counter"]["optimizer_fields"] = ["master_param"]
    with pytest.raises(ValueError, match="coverage"):
        candidate.ordered_trainable_names(model, {"parameters": entries})


@pytest.mark.parametrize("fail", [False, True])
def test_production_meta_rng_scope_restores_cpu_rngs_even_on_error(candidate, fail):
    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)
    torch_before, py_before, numpy_before = (
        torch.random.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state(),
    )

    def mutate():
        with candidate.preserve_training_rng():
            torch.rand(3)
            random.random()
            np.random.rand()
            if fail:
                raise ValueError("simulated metadata error")

    if fail:
        with pytest.raises(ValueError, match="metadata error"):
            mutate()
    else:
        mutate()
    numpy_after = np.random.get_state()
    assert torch.equal(torch_before, torch.random.get_rng_state()) and py_before == random.getstate()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_production_meta_rng_scope_restores_mcore_tracker_without_gpu(candidate, monkeypatch):
    from megatron.core.tensor_parallel import random as mcore_random

    tracker = mcore_random.CudaRNGStatesTracker()
    tracker._is_initialized = True
    tracker.states_ = {"model": torch.tensor([1, 2, 3], dtype=torch.uint8)}
    tracker.seeds_ = {42}
    tracker._current_state_name = "default-rng"
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.random, "fork_rng", lambda *, devices: nullcontext())
    monkeypatch.setattr(mcore_random, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True)
    monkeypatch.setattr(mcore_random, "get_cuda_rng_tracker", lambda: tracker)
    with candidate.preserve_training_rng():
        tracker.states_["model"].add_(1)
        tracker.seeds_.add(12)
        tracker._current_state_name = "model"
    assert torch.equal(tracker.states_["model"], torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert tracker.seeds_ == {42}
    assert tracker._current_state_name == "default-rng"


@pytest.mark.parametrize(
    ("flag", "changed", "peer_enabled", "error"),
    [
        ("0", {}, 0, None),
        ("1", {}, 1, None),
        ("0", {}, 1, "agree"),
        ("1", {"use_precision_aware_optimizer": True}, 1, "unsupported"),
        ("1", {"loss_scale": 8.0}, 1, "unsupported"),
        ("1", {"clip_grad": float("nan")}, 1, "invalid"),
        ("1", {"clip_grad": float("inf")}, 1, "invalid"),
        ("1", {"grad_norm_skip_threshold": 2.0}, 1, "unsupported"),
        ("1", {"optimizer": "sgd"}, 1, "unsupported"),
    ],
)
def test_all_rank_flag_agreement_and_optimizer_restrictions(candidate, monkeypatch, flag, changed, peer_enabled, error):
    config = dict(
        clip_grad=0.3,
        optimizer="adam",
        fp16=False,
        bf16=False,
        use_precision_aware_optimizer=False,
        loss_scale=None,
        grad_norm_skip_threshold=float("inf"),
        use_distributed_optimizer=True,
    )
    config.update(changed)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    calls = []

    def reduce(tensor, *, op):
        calls.append(1)
        tensor[1] += peer_enabled

    monkeypatch.setattr(candidate.dist, "all_reduce", reduce)
    kwargs = dict(dtype=torch.float32, dp_size=1, optimizer_config=SimpleNamespace(**config), device="cpu")
    if error:
        with pytest.raises(ValueError, match=error):
            candidate.agree_requested(flag, **kwargs)
    else:
        assert candidate.agree_requested(flag, **kwargs) is (flag == "1")
    assert len(calls) == 1


def test_tiny_network_meta_qualification_preserves_rng_and_stays_cpu(candidate, monkeypatch):
    repo = Path("/share-mixed/project/eai_pwm/repos/cosmos-framework-pwm-megatron")
    monkeypatch.syspath_prepend(str(repo / "tests/megatron"))
    from tp_pp_network_io_test import make_native

    path = Path(__file__).parents[2] / "tools/qualify_native_fp32_norm_network.py"
    spec = importlib.util.spec_from_file_location("_native_norm_small_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu_before, py_before, numpy_before = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
    assert not torch.cuda.is_initialized()
    names = module.tiny_meta_order(make_native, candidate.ordered_trainable_names)
    assert len(names) == 63 and len(set(names)) == len(names)
    assert all(name.startswith("net.") for name in names)
    assert torch.equal(cpu_before, torch.random.get_rng_state()) and py_before == random.getstate()
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert not torch.cuda.is_initialized()


def test_meta_order_child_runs_without_cuda_before_parent_initialization(monkeypatch):
    path = Path(__file__).parents[2] / "tools/qualify_native_fp32_norm_network.py"
    spec = importlib.util.spec_from_file_location("_native_norm_child_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parents[2]))
    # Even if the caller has visible GPUs, the derived-order child must hide them.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert not torch.cuda.is_initialized()
    names = module.meta_order_before_parent_cuda_initialization()
    assert len(names) == 63 and len(set(names)) == len(names)
    assert all(name.startswith("net.") for name in names)
    assert not torch.cuda.is_initialized()


def test_meta_order_rejects_parent_cuda_initialized_before_child(monkeypatch):
    path = Path(__file__).parents[2] / "tools/qualify_native_fp32_norm_network.py"
    spec = importlib.util.spec_from_file_location("_native_norm_parent_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before parent CUDA initialization"):
        module.meta_order_before_parent_cuda_initialization()


def test_tp_shards_rebuilt_in_native_order_and_unused_excluded(candidate, monkeypatch):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    class Stage(nn.Module):
        def __init__(self):
            super().__init__()
            self.column = nn.Parameter(torch.zeros(2, 3))
            self.row = nn.Parameter(torch.zeros(2, 2))
            self.replicated = nn.Parameter(torch.zeros(2))
            self.config = SimpleNamespace(params_dtype=torch.float32)
            self.column.main_grad = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            self.row.main_grad = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
            self.replicated.main_grad = torch.tensor([1000.0, 1000.0])

    stage = Stage()
    specs = {
        name: SimpleNamespace(canonical_name="net." + name, axis=axis)
        for name, axis in (("column", 0), ("row", 1), ("replicated", None))
    }
    monkeypatch.setattr(parameter_map, "stage_parameter_manifest", lambda _: specs)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(ps, "get_model_parallel_group", lambda: "mp")
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    calls = []

    def gather(pieces, local, group):
        assert group == "tp"
        calls.append(tuple(local.shape))
        pieces[0].copy_(local)
        pieces[1].copy_(local + 11)

    monkeypatch.setattr(candidate.dist, "all_gather", gather)
    monkeypatch.setattr(candidate.dist, "all_reduce", lambda *args, **kwargs: None)
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.column, stage.row},
        _native_fp32_norm_calls=0,
    )
    order = ["net.row", "net.replicated", "net.column"]
    actual = candidate.gradient_norm(driver, order)
    full_row = torch.cat([stage.row.main_grad, stage.row.main_grad + 11], dim=1)
    full_column = torch.cat([stage.column.main_grad, stage.column.main_grad + 11], dim=0)
    expected = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([full_row, full_column], 2)), 2).item()
    assert actual == expected
    assert calls == [(2, 3), (2, 2)]
    assert driver._native_fp32_norm_calls == 1


def test_missing_global_owner_is_rejected(candidate, monkeypatch):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    stage = nn.Linear(2, 2, bias=False)
    stage.config = SimpleNamespace(params_dtype=torch.float32)
    stage.weight.main_grad = torch.ones_like(stage.weight)
    monkeypatch.setattr(
        parameter_map,
        "stage_parameter_manifest",
        lambda _: {
            "weight": SimpleNamespace(canonical_name="net.weight", axis=None),
        },
    )
    for name, value in (
        ("get_tensor_model_parallel_group", None),
        ("get_model_parallel_group", None),
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 1),
    ):
        monkeypatch.setattr(ps, name, lambda value=value: value)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(candidate.dist, "all_reduce", lambda *args, **kwargs: None)
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.weight},
        _native_fp32_norm_calls=0,
    )
    with pytest.raises(ValueError, match="exactly one"):
        candidate.gradient_norm(driver, ["net.weight", "net.missing"])


def test_duplicate_pp_owner_is_rejected_on_all_model_parallel_ranks(candidate, monkeypatch):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    stage = nn.Linear(2, 2, bias=False)
    stage.config = SimpleNamespace(params_dtype=torch.float32)
    stage.weight.main_grad = torch.ones_like(stage.weight)
    monkeypatch.setattr(
        parameter_map,
        "stage_parameter_manifest",
        lambda _: {"weight": SimpleNamespace(canonical_name="net.weight", axis=None)},
    )
    for name, value in (
        ("get_tensor_model_parallel_group", "tp"),
        ("get_model_parallel_group", "mp"),
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 1),
    ):
        monkeypatch.setattr(ps, name, lambda value=value: value)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    calls = []

    def reduce(tensor, *, op, group):
        calls.append((op, group))
        # Preflight MAX, norm SUM, usage MAX, then ownership SUM. Simulate
        # a second pipeline stage claiming the same canonical parameter.
        if len(calls) == 4:
            tensor.add_(1)

    monkeypatch.setattr(candidate.dist, "all_reduce", reduce)
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.weight},
        _native_fp32_norm_calls=0,
    )
    with pytest.raises(ValueError, match="exactly one"):
        candidate.gradient_norm(driver, ["net.weight"])
    assert len(calls) == 4


def test_unsupported_axis_fails_before_any_tp_gather(candidate, monkeypatch):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    stage = nn.Linear(2, 2, bias=False)
    stage.config = SimpleNamespace(params_dtype=torch.float32)
    stage.weight.main_grad = torch.ones_like(stage.weight)
    monkeypatch.setattr(
        parameter_map,
        "stage_parameter_manifest",
        lambda _: {"weight": SimpleNamespace(canonical_name="net.weight", axis=3)},
    )
    for name, value in (
        ("get_tensor_model_parallel_group", "tp"),
        ("get_model_parallel_group", "mp"),
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 2),
    ):
        monkeypatch.setattr(ps, name, lambda value=value: value)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(candidate.dist, "all_reduce", lambda *args, **kwargs: None)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError("TP gather was reached before validation")

    monkeypatch.setattr(candidate.dist, "all_gather", unexpected_gather)
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.weight},
        _native_fp32_norm_calls=0,
    )
    with pytest.raises(ValueError, match="shard axis"):
        candidate.gradient_norm(driver, ["net.weight"])


def test_different_rank_registration_order_fails_before_tp_gather(candidate, monkeypatch):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    stage = nn.Linear(2, 2, bias=False)
    stage.config = SimpleNamespace(params_dtype=torch.float32)
    stage.weight.main_grad = torch.ones_like(stage.weight)
    monkeypatch.setattr(
        parameter_map,
        "stage_parameter_manifest",
        lambda _: {"weight": SimpleNamespace(canonical_name="net.weight", axis=None)},
    )
    for name, value in (
        ("get_tensor_model_parallel_group", "tp"),
        ("get_model_parallel_group", "mp"),
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 2),
    ):
        monkeypatch.setattr(ps, name, lambda value=value: value)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    calls = []

    def reduce(tensor, *, op, group):
        calls.append(op)
        if len(calls) == 2:
            tensor.zero_()  # Simulate a distinct order hash on a peer.

    monkeypatch.setattr(candidate.dist, "all_reduce", reduce)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError("TP gather was reached before order validation")

    monkeypatch.setattr(candidate.dist, "all_gather", unexpected_gather)
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.weight},
        _native_fp32_norm_calls=0,
    )
    with pytest.raises(ValueError, match="order differs"):
        candidate.gradient_norm(driver, ["net.weight"])
    assert len(calls) == 3


@pytest.mark.parametrize(
    "peer_layout",
    [
        [("net.b", None, [2, 2]), ("net.a", 0, [2, 2])],
        [("net.a", 1, [2, 2]), ("net.b", None, [2, 2])],
        [("net.a", 0, [3, 2]), ("net.b", None, [2, 2])],
    ],
    ids=["order", "axis", "shape"],
)
def test_mismatched_tp_layout_is_rejected_before_data_gather(candidate, monkeypatch, peer_layout):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    stage = nn.Module()
    stage.a = nn.Parameter(torch.zeros(2, 2))
    stage.b = nn.Parameter(torch.zeros(2, 2))
    stage.config = SimpleNamespace(params_dtype=torch.float32)
    for parameter in stage.parameters():
        parameter.main_grad = torch.ones_like(parameter)
    monkeypatch.setattr(
        parameter_map,
        "stage_parameter_manifest",
        lambda _: {
            "a": SimpleNamespace(canonical_name="net.a", axis=0),
            "b": SimpleNamespace(canonical_name="net.b", axis=None),
        },
    )
    for name, value in (
        ("get_tensor_model_parallel_group", "tp"),
        ("get_model_parallel_group", "mp"),
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 2),
    ):
        monkeypatch.setattr(ps, name, lambda value=value: value)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    peer_digest = torch.tensor(
        list(hashlib.sha256(json.dumps(peer_layout, separators=(",", ":")).encode()).digest()), dtype=torch.uint8
    )
    calls = []

    def reduce(tensor, *, op, group):
        calls.append((op, group))
        if group == "tp":
            tensor.copy_(
                torch.minimum(tensor, peer_digest)
                if op == candidate.dist.ReduceOp.MIN
                else torch.maximum(tensor, peer_digest)
            )

    monkeypatch.setattr(candidate.dist, "all_reduce", reduce)
    monkeypatch.setattr(
        candidate.dist, "all_gather", lambda *args, **kwargs: pytest.fail("data gather reached before TP validation")
    )
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.a, stage.b},
        _native_fp32_norm_calls=0,
    )
    with pytest.raises(ValueError, match="TP ranks have different"):
        candidate.gradient_norm(driver, ["net.a", "net.b"])
    assert [group for _, group in calls[-3:]] == ["tp", "tp", "mp"]


@pytest.mark.parametrize("failure", ["manifest", "duplicate_order", "gradient_shape", "model_group", "optimizer_group"])
def test_local_preflight_errors_are_reduced_before_raising(candidate, monkeypatch, failure):
    import megatron.core.parallel_state as ps

    from cosmos_framework.training_backends.megatron_tp_pp import parameter_map

    stage = nn.Linear(2, 2, bias=False)
    stage.config = SimpleNamespace(params_dtype=torch.float32)
    stage.weight.main_grad = torch.ones(3 if failure == "gradient_shape" else 2, 2)
    if failure == "model_group":
        stage.weight.grad_norm_group = "mtp"
    optimizer_parameter = nn.Parameter(torch.zeros(1))
    if failure == "optimizer_group":
        optimizer_parameter.grad_norm_group = "mtp"

    def manifest(_):
        if failure == "manifest":
            raise ValueError("invalid local manifest")
        return {"weight": SimpleNamespace(canonical_name="net.weight", axis=0)}

    monkeypatch.setattr(parameter_map, "stage_parameter_manifest", manifest)
    for name, value in (
        ("get_tensor_model_parallel_group", "tp"),
        ("get_model_parallel_group", "mp"),
        ("get_tensor_model_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 2),
    ):
        monkeypatch.setattr(ps, name, lambda value=value: value)
    monkeypatch.setattr(candidate.dist, "get_world_size", lambda: 2)
    calls = []
    monkeypatch.setattr(candidate.dist, "all_reduce", lambda tensor, **kwargs: calls.append(kwargs["group"]))
    monkeypatch.setattr(candidate.dist, "all_gather", lambda *args, **kwargs: pytest.fail("unexpected data gather"))
    driver = SimpleNamespace(
        stage=stage,
        topology=SimpleNamespace(data_parallel_size=lambda _: 1),
        _globally_used={stage.weight},
        _native_fp32_norm_calls=0,
        optimizer=SimpleNamespace(chained_optimizers=[SimpleNamespace(get_parameters=lambda: [optimizer_parameter])]),
    )
    order = ["net.weight"] * (2 if failure == "duplicate_order" else 1)
    with pytest.raises(ValueError, match="unsupported stage"):
        candidate.gradient_norm(driver, order)
    assert calls == ["mp"]
