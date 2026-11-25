# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU fixtures for MCore's real dp_reshardable optimizer loading template."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.training_backends.megatron_tp_pp import checkpoint
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states, initialize_adam_state


@pytest.fixture
def gloo(tmp_path, monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()
    assert not torch.cuda.is_initialized()


def make_optimizer(*, separate_groups=False, separate_buckets=False, inner_type=None, padding=0):
    from megatron.core.optimizer.distrib_optimizer import Adam, DistributedOptimizer, Range

    stage = nn.Module()
    stage.register_parameter("first", nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0])))
    stage.register_parameter("second", nn.Parameter(torch.tensor([5.0, 6.0, 7.0, 8.0])))
    params = list(stage.parameters())
    main_params = [param.detach().clone() for param in params]
    # FusedAdam's constructor allocates its CUDA overflow buffer even when no
    # step is run. Instantiate its actual Optimizer base state on CPU; the
    # tests exercise MCore's real state_dict/load_state_dict, not its CUDA step.
    groups = [{"params": [param]} for param in main_params] if separate_groups else main_params
    inner = object.__new__(inner_type or Adam)
    torch.optim.Optimizer.__init__(inner, groups, {"lr": 0.01, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.01})
    inner.name_to_dtype_map = {name: torch.float32 for name in ("master_param", "exp_avg", "exp_avg_sq")}
    inner.store_param_remainders = False
    inner._scales = {}
    wrapper = object.__new__(DistributedOptimizer)
    wrapper.config = SimpleNamespace(
        optimizer="adam",
        bf16=False,
        fp16=False,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
    )
    wrapper.optimizer = inner
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    wrapper.grad_scaler = None
    wrapper.is_stub_optimizer = False
    wrapper.data_parallel_group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    wrapper.data_parallel_group_idx = 0
    wrapper.distributed_optimizer_instance_id = 0
    assert not separate_buckets or padding == 0
    bucket_sizes = [4, 4] if separate_buckets else [8 + padding]
    wrapper.per_bucket_numel = bucket_sizes
    wrapper.per_bucket_numel_unpadded = bucket_sizes
    wrapper.buffers = [
        SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.zeros(n), numel_unpadded=n) for n in bucket_sizes])
    ]
    dtype = (torch.float32, torch.float32)
    param_map = {
        param: {
            "param": Range(0, 4),
            "gbuf_world": Range(i * (4 + padding), i * (4 + padding) + 4),
            "gbuf_local": Range(0, 4) if separate_buckets else Range(i * (4 + padding), i * (4 + padding) + 4),
        }
        for i, param in enumerate(params)
    }
    buckets = (
        [{"param_map": {param: param_map[param]}} for param in params]
        if separate_buckets
        else [{"param_map": param_map}]
    )
    wrapper.gbuf_ranges = [{dtype: buckets}]
    wrapper.model_param_gbuf_map = {param: (0, dtype, i if separate_buckets else 0) for i, param in enumerate(params)}
    wrapper.model_param_group_index_map = {
        param: ((i, 0) if separate_groups else (0, i)) for i, param in enumerate(params)
    }
    return stage, wrapper


def template(wrapper, *, loading):
    return wrapper.sharded_state_dict(
        {}, is_loading=loading, metadata={"distrib_optim_sharding_type": "dp_reshardable", "checkpoint_version": 3.1}
    )


def keys_in_template(state):
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    return {value.key for value in nested_values(state) if isinstance(value, ShardedTensor)}


def add_moments(stage, wrapper):
    for i, entry in enumerate(adam_parameter_states(stage, wrapper)):
        value = torch.full_like(entry.model_parameter, float(i + 1))
        initialize_adam_state(entry, step=4, master=entry.model_parameter, exp_avg=value * 0.1, exp_avg_sq=value * 0.01)


def test_unprepared_mcore_template_loses_saved_moment_keys(gloo):
    stage, wrapper = make_optimizer()
    list(adam_parameter_states(stage, wrapper))  # Current identity-only driver enumeration.
    assert len(wrapper.optimizer.state) == 2
    keys = keys_in_template(template(wrapper, loading=True))
    assert any(key.endswith(".param") for key in keys)
    assert not any(key.endswith(".exp_avg") or key.endswith(".exp_avg_sq") for key in keys)


@pytest.mark.parametrize("enumerate_before", [False, True])
def test_fresh_template_and_real_mcore_dcp_restore(gloo, tmp_path, enumerate_before):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer()
    add_moments(source_stage, source)
    expected = {
        entry.name: (entry.master.detach().clone(), entry.state["exp_avg"].clone(), entry.state["exp_avg_sq"].clone())
        for entry in adam_parameter_states(source_stage, source)
    }
    path = tmp_path / "dcp"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    target_stage, target = make_optimizer()
    if enumerate_before:
        list(adam_parameter_states(target_stage, target))
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    target_template = template(target, loading=True)
    keys = keys_in_template(target_template)
    assert any(key.endswith(".exp_avg") for key in keys)
    assert any(key.endswith(".exp_avg_sq") for key in keys)
    loaded = dist_checkpointing.load({"optimizer": target_template}, str(path))
    target.load_state_dict(loaded["optimizer"])
    for entry in adam_parameter_states(target_stage, target):
        master, avg, avg_sq = expected[entry.name]
        torch.testing.assert_close(entry.master, master, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg"], avg, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg_sq"], avg_sq, rtol=0, atol=0)
        assert entry.step == 4


@pytest.mark.parametrize("separate_groups", [False, True])
def test_canonical_import_keeps_complete_existing_state(gloo, tmp_path, separate_groups):
    from megatron.core import dist_checkpointing

    from cosmos_framework.training_backends.megatron_tp_pp.bundle import CanonicalBundle

    source_stage, source = make_optimizer(separate_groups=separate_groups)
    for i, group in enumerate(source.optimizer.param_groups):
        group["lr_mult"] = i + 1
    add_moments(source_stage, source)
    path = tmp_path / "canonical_source_dcp"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    stage, wrapper = make_optimizer(separate_groups=separate_groups)
    for i, group in enumerate(wrapper.optimizer.param_groups):
        group["lr_mult"] = i + 1
    entries = list(adam_parameter_states(stage, wrapper))
    bundle = object.__new__(CanonicalBundle)
    bundle.manifest = {"parameters": {entry.name: {"master_origin": "model_fp32"} for entry in entries}}
    bundle._stage_manifest = lambda stage: {
        entry.name: SimpleNamespace(canonical_name=entry.name, tensor_rank=0) for entry in entries
    }
    bundle.read = lambda name, rank: {
        "state": {
            "master_param": torch.full((4,), 42.0),
            "exp_avg": torch.full((4,), 0.25),
            "exp_avg_sq": torch.full((4,), 0.125),
        },
        "hyperparameters": {
            "lr": 0.01,
            "initial_lr": 0.01,
            "weight_decay": 0.01,
            "betas": (0.9, 0.99),
            "eps": 1e-8,
            "step": 4,
        },
    }
    bundle.load_optimizer(stage, wrapper)
    mapping = wrapper.optimizer.state
    values = {
        entry.name: (entry.state, entry.state["exp_avg"].clone(), entry.state["exp_avg_sq"].clone())
        for entry in entries
    }
    checkpoint._prepare_optimizer_for_load(
        wrapper, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    assert wrapper.optimizer.state is mapping
    for entry in adam_parameter_states(stage, wrapper):
        state, avg, avg_sq = values[entry.name]
        assert entry.state is state
        torch.testing.assert_close(entry.state["exp_avg"], avg, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg_sq"], avg_sq, rtol=0, atol=0)
        torch.testing.assert_close(entry.master, entry.model_parameter, rtol=0, atol=0)
        assert entry.step == 4
    loaded = dist_checkpointing.load({"optimizer": template(wrapper, loading=True)}, str(path))
    wrapper.load_state_dict(loaded["optimizer"])
    for expected, actual in zip(
        adam_parameter_states(source_stage, source), adam_parameter_states(stage, wrapper), strict=True
    ):
        torch.testing.assert_close(actual.state["exp_avg"], expected.state["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(actual.state["exp_avg_sq"], expected.state["exp_avg_sq"], rtol=0, atol=0)
        assert actual.step == expected.step


@pytest.mark.parametrize("missing", ["empty", "missing_first_moment", "missing_second_moment"])
def test_partial_moments_fail_before_mutating_any_optimizer(gloo, tmp_path, missing):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer(separate_groups=True)
    add_moments(source_stage, source)
    path = tmp_path / "complete_dcp"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    stage, wrapper = make_optimizer(separate_groups=True)
    add_moments(stage, wrapper)
    params = [p for group in wrapper.optimizer.param_groups for p in group["params"]]
    broken = wrapper.optimizer.state[params[1]]
    if missing == "empty":
        broken.clear()
    elif missing == "missing_first_moment":
        del broken["exp_avg"]
    else:
        del broken["exp_avg_sq"]
    before_ids = {id(param): id(value) for param, value in wrapper.optimizer.state.items()}
    with pytest.raises(ValueError, match="At least one rank"):
        checkpoint._prepare_optimizer_for_load(
            wrapper, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
        )
    assert {id(param): id(value) for param, value in wrapper.optimizer.state.items()} == before_ids


def test_non_distributed_optimizer_is_untouched(gloo, tmp_path):
    from torch.distributed import checkpoint as dcp

    path = tmp_path / "plain_dcp"
    dcp.save({"value": torch.ones(1)}, checkpoint_id=path)
    parameter = nn.Parameter(torch.ones(2))
    inner = torch.optim.AdamW([parameter])
    inner.state[parameter]  # Existing empty state is outside the affected MCore path.
    mapping, entry = inner.state, inner.state[parameter]
    checkpoint._prepare_optimizer_for_load(
        SimpleNamespace(optimizer=inner), path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    assert inner.state is mapping and inner.state[parameter] is entry


def test_distributed_pytorch_adam_complete_state_is_preserved(gloo, tmp_path):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer()
    add_moments(source_stage, source)
    path = tmp_path / "adam_source_dcp"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    stage, wrapper = make_optimizer(inner_type=torch.optim.AdamW)
    entries = list(adam_parameter_states(stage, wrapper))
    for entry in entries:
        initialize_adam_state(
            entry,
            step=4,
            master=entry.model_parameter,
            exp_avg=torch.full((4,), 0.25),
            exp_avg_sq=torch.full((4,), 0.125),
        )
    before = [entry.state for entry in entries]
    checkpoint._prepare_optimizer_for_load(
        wrapper, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    assert all(entry.state is state for entry, state in zip(adam_parameter_states(stage, wrapper), before, strict=True))


def test_existing_mixed_state_can_restore_from_real_mcore_dcp(gloo, tmp_path):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer(separate_buckets=True)
    add_moments(source_stage, source)
    source_entries = list(adam_parameter_states(source_stage, source))
    source_entries[1].state.clear()
    expected = source_entries[0].state["exp_avg"].clone()
    path = tmp_path / "mixed_dcp"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))

    target_stage, target = make_optimizer(separate_buckets=True)
    add_moments(target_stage, target)
    target_entries = list(adam_parameter_states(target_stage, target))
    target_entries[0].state["exp_avg"].zero_()
    target_entries[1].state.clear()
    before = [entry.state for entry in target_entries]
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    assert all(
        entry.state is state for entry, state in zip(adam_parameter_states(target_stage, target), before, strict=True)
    )
    loaded = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(path))
    target.load_state_dict(loaded["optimizer"])
    restored = list(adam_parameter_states(target_stage, target))
    torch.testing.assert_close(restored[0].state["exp_avg"], expected, rtol=0, atol=0)
    assert restored[1].state == {}


@pytest.mark.parametrize("trained_index", [0, 1])
@pytest.mark.parametrize("padding", [0, 2])
def test_partial_same_bucket_real_dcp_matches_existing_template(gloo, tmp_path, trained_index, padding):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer(padding=padding)
    add_moments(source_stage, source)
    source_entries = list(adam_parameter_states(source_stage, source))
    source_entries[1 - trained_index].state.clear()
    expected = source_entries[trained_index].state["exp_avg"].clone()
    path = tmp_path / "partial_same_bucket"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    target_stage, target = make_optimizer(padding=padding)
    add_moments(target_stage, target)
    target_entries = list(adam_parameter_states(target_stage, target))
    target_entries[trained_index].state["exp_avg"].zero_()
    target_entries[1 - trained_index].state.clear()
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    loaded = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(path))
    target.load_state_dict(loaded["optimizer"])
    restored = list(adam_parameter_states(target_stage, target))
    torch.testing.assert_close(restored[trained_index].state["exp_avg"], expected, rtol=0, atol=0)
    assert restored[1 - trained_index].state == {}


@pytest.mark.parametrize("mode", ["no_shards", "missing_parameter_metadata"])
def test_existing_parameters_cannot_bypass_metadata_preflight(gloo, tmp_path, monkeypatch, mode):
    from megatron.core import dist_checkpointing
    from torch.distributed.checkpoint import FileSystemReader

    source_stage, source = make_optimizer()
    add_moments(source_stage, source)
    path = tmp_path / "source"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    target_stage, target = make_optimizer()
    list(adam_parameter_states(target_stage, target))
    if mode == "no_shards":
        monkeypatch.setattr(target, "sharded_param_state_dp_reshardable", lambda *args, **kwargs: {})
    else:
        real = FileSystemReader.read_metadata

        def without_param(reader):
            metadata = real(reader)
            metadata.state_dict_metadata = {
                k: v for k, v in metadata.state_dict_metadata.items() if not k.endswith(".param")
            }
            return metadata

        monkeypatch.setattr(FileSystemReader, "read_metadata", without_param)
    with pytest.raises(ValueError, match="At least one rank"):
        checkpoint._prepare_optimizer_for_load(
            target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
        )


@pytest.mark.parametrize("enumerate_before", [False, True])
def test_all_untrained_state_can_restore_from_real_mcore_dcp(gloo, tmp_path, enumerate_before):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer()
    list(adam_parameter_states(source_stage, source))
    path = tmp_path / "untrained_dcp"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    target_stage, target = make_optimizer()
    if enumerate_before:
        list(adam_parameter_states(target_stage, target))
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    loaded = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(path))
    target.load_state_dict(loaded["optimizer"])
    assert all(not entry.state for entry in adam_parameter_states(target_stage, target))


def test_new_empty_target_restores_attested_mixed_source(gloo, tmp_path):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer(separate_buckets=True)
    add_moments(source_stage, source)
    list(adam_parameter_states(source_stage, source))[1].state.clear()
    path = tmp_path / "mixed_source"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    expected = list(adam_parameter_states(source_stage, source))[0].state["exp_avg"].clone()
    stage, target = make_optimizer(separate_buckets=True)
    entries = list(adam_parameter_states(stage, target))
    mapping, values = target.optimizer.state, [entry.state for entry in entries]
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    assert target.optimizer.state is mapping
    assert all(entry.state is state for entry, state in zip(adam_parameter_states(stage, target), values, strict=True))
    loaded = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(path))
    target.load_state_dict(loaded["optimizer"])
    restored = list(adam_parameter_states(stage, target))
    torch.testing.assert_close(restored[0].state["exp_avg"], expected, rtol=0, atol=0)
    assert restored[1].state == {}


@pytest.mark.parametrize("trained_index", [0, 1])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("existing", [False, True])
def test_attested_mixed_bucket_restores_fresh_or_canonical_target(gloo, tmp_path, trained_index, padding, existing):
    from megatron.core import dist_checkpointing

    source_stage, source = make_optimizer(padding=padding)
    add_moments(source_stage, source)
    source_entries = list(adam_parameter_states(source_stage, source))
    source_entries[1 - trained_index].state.clear()
    expected = {key: source_entries[trained_index].state[key].clone() for key in ("exp_avg", "exp_avg_sq")}
    path = tmp_path / "mixed_source"
    path.mkdir()
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(path))
    stage, target = make_optimizer(padding=padding)
    if existing:
        add_moments(stage, target)
    else:
        list(adam_parameter_states(stage, target))
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    loaded = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(path))
    target.load_state_dict(loaded["optimizer"])
    restored = list(adam_parameter_states(stage, target))
    for key, value in expected.items():
        torch.testing.assert_close(restored[trained_index].state[key], value, rtol=0, atol=0)
    assert restored[1 - trained_index].state == {}


def test_multiple_chained_optimizers_use_prefixed_keys_and_are_atomic(gloo, tmp_path):
    from megatron.core import dist_checkpointing
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    metadata = {"distrib_optim_sharding_type": "dp_reshardable", "checkpoint_version": 3.1}
    source_pairs = [make_optimizer() for _ in range(2)]
    for stage, wrapper in source_pairs:
        add_moments(stage, wrapper)
    source = ChainedOptimizer([wrapper for _, wrapper in source_pairs])
    path = tmp_path / "chained_dcp"
    path.mkdir()
    dist_checkpointing.save(
        {"optimizer": source.sharded_state_dict({}, is_loading=False, metadata=metadata)}, str(path)
    )

    target_pairs = [make_optimizer() for _ in range(2)]
    for stage, wrapper in target_pairs:
        list(adam_parameter_states(stage, wrapper))
    target = ChainedOptimizer([wrapper for _, wrapper in target_pairs])
    mapping0 = target_pairs[0][1].optimizer.state
    add_moments(*target_pairs[1])
    list(adam_parameter_states(*target_pairs[1]))[1].state.clear()
    before = {id(param): id(state) for param, state in mapping0.items()}
    with pytest.raises(ValueError, match="At least one rank"):
        checkpoint._prepare_optimizer_for_load(
            target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
        )
    assert {id(param): id(state) for param, state in mapping0.items()} == before
    for state in target_pairs[1][1].optimizer.state.values():
        state.clear()
    checkpoint._prepare_optimizer_for_load(
        target, path, device="cpu", metadata_sha256=checkpoint._metadata_sha256(path)
    )
    assert all(not wrapper.optimizer.state for _, wrapper in target_pairs)
    loaded = dist_checkpointing.load(
        {"optimizer": target.sharded_state_dict({}, is_loading=True, metadata=metadata)}, str(path)
    )
    target.load_state_dict(loaded["optimizer"])
    for (source_stage, source_wrapper), (target_stage, target_wrapper) in zip(source_pairs, target_pairs, strict=True):
        for expected, actual in zip(
            adam_parameter_states(source_stage, source_wrapper),
            adam_parameter_states(target_stage, target_wrapper),
            strict=True,
        ):
            torch.testing.assert_close(actual.state["exp_avg"], expected.state["exp_avg"], rtol=0, atol=0)
            torch.testing.assert_close(actual.state["exp_avg_sq"], expected.state["exp_avg_sq"], rtol=0, atol=0)


def test_incomplete_or_malformed_dcp_ranges_fail_closed():
    def chunk(start, length):
        return SimpleNamespace(offsets=(start,), sizes=(length,))

    key = "state.exp_avg"
    assert not checkpoint._checkpoint_range_present({}, key, 4, 8)
    assert checkpoint._checkpoint_range_present({key: SimpleNamespace(chunks=[chunk(0, 5), chunk(5, 5)])}, key, 2, 9)
    assert not checkpoint._checkpoint_range_present({key: SimpleNamespace(chunks=[chunk(0, 2)])}, key, 4, 8)
    for chunks in ([chunk(0, 5)], [chunk(0, 3), chunk(5, 5)]):
        with pytest.raises(ValueError, match="Incomplete checkpoint coverage"):
            checkpoint._checkpoint_range_present({key: SimpleNamespace(chunks=chunks)}, key, 2, 8)
    with pytest.raises(ValueError, match="flat optimizer"):
        checkpoint._checkpoint_range_present(
            {key: SimpleNamespace(chunks=[SimpleNamespace(offsets=(0, 0), sizes=(1, 1))])}, key, 0, 1
        )
