# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TP2/DP2 CPU integration: real MCore sharding, DCP and failure agreement.

The checkpoint is a new, tiny test fixture supplied on the command line.  A
real Gloo DP group owns different portions of the same bucket: parameter 0
crosses the DP boundary, the bucket contains a gap, and its tail is padded.
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/megatron")]
sys.dont_write_bytecode = True
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import torch

assert not torch.cuda.is_initialized()


def no_cuda(*args, **kwargs):
    raise AssertionError("CUDA initialization is forbidden in this CPU test")


torch.cuda._lazy_init = no_cuda
torch.cuda.current_device = lambda: "cpu"
real_tensor = torch.tensor


def cpu_control_tensor(*args, **kwargs):
    if kwargs.get("device") == "cuda":
        kwargs["device"] = "cpu"
    return real_tensor(*args, **kwargs)


torch.tensor = cpu_control_tensor

import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from torch.distributed.checkpoint import FileSystemReader
from tp_pp_fresh_checkpoint_restore_test import add_moments, make_optimizer, template

from cosmos_framework.training_backends.megatron_tp_pp import checkpoint
from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states


def sharded_optimizer(dp_group, tp_rank):
    stage, wrapper = make_optimizer()
    params = list(stage.parameters())
    dp_rank = dp_group.rank()
    ranges = DistributedOptimizer._build_model_gbuf_param_range_map(
        {params[0]: (5, 9, 0), params[1]: (10, 14, 0)}, Range(dp_rank * 8, (dp_rank + 1) * 8), 0
    )
    local_params = list(ranges)
    masters = [
        param.detach().view(-1)[ranges[param]["param"].start : ranges[param]["param"].end].clone()
        for param in local_params
    ]
    wrapper.optimizer.state.clear()
    wrapper.optimizer.param_groups[0]["params"] = masters
    wrapper.data_parallel_group = dp_group
    wrapper.data_parallel_group_idx = tp_rank
    wrapper.per_bucket_numel = [16]
    wrapper.per_bucket_numel_unpadded = [14]
    wrapper.buffers = [SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.zeros(16), numel_unpadded=14)])]
    dtype = (torch.float32, torch.float32)
    wrapper.gbuf_ranges = [{dtype: [{"param_map": ranges}]}]
    wrapper.model_param_gbuf_map = {param: (0, dtype, 0) for param in local_params}
    wrapper.model_param_group_index_map = {param: (0, index) for index, param in enumerate(local_params)}
    offsets = [(ranges[param]["gbuf_world"].start, ranges[param]["gbuf_world"].end) for param in local_params]
    assert offsets == ([(5, 8)] if dp_rank == 0 else [(8, 9), (10, 14)])
    return stage, wrapper


def separate_bucket_optimizer(dp_group, tp_rank):
    """Two crossing parameters in distinct buckets with DP-end padding."""
    stage, wrapper = make_optimizer()
    params = list(stage.parameters())
    dp_rank = dp_group.rank()
    ranges = []
    for bucket, param in enumerate(params):
        offset = bucket * 8
        ranges.append(
            DistributedOptimizer._build_model_gbuf_param_range_map(
                {param: (offset + 2, offset + 6, bucket)},
                Range(offset + dp_rank * 4, offset + (dp_rank + 1) * 4),
                offset,
            )
        )
    masters = [
        param.detach().view(-1)[ranges[index][param]["param"].start : ranges[index][param]["param"].end].clone()
        for index, param in enumerate(params)
    ]
    wrapper.optimizer.state.clear()
    wrapper.optimizer.param_groups[0]["params"] = masters
    wrapper.data_parallel_group = dp_group
    wrapper.data_parallel_group_idx = tp_rank
    wrapper.per_bucket_numel = [8, 8]
    wrapper.per_bucket_numel_unpadded = [6, 6]
    wrapper.buffers = [
        SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.zeros(8), numel_unpadded=6) for _ in params])
    ]
    dtype = (torch.float32, torch.float32)
    wrapper.gbuf_ranges = [{dtype: [{"param_map": value} for value in ranges]}]
    wrapper.model_param_gbuf_map = {param: (0, dtype, index) for index, param in enumerate(params)}
    wrapper.model_param_group_index_map = {param: (0, index) for index, param in enumerate(params)}
    for index, param in enumerate(params):
        local = ranges[index][param]["gbuf_local"]
        assert (local.start, local.end) == ((2, 4) if dp_rank == 0 else (0, 2))
    return stage, wrapper


def state_identity(optimizer):
    return {
        id(param): (id(value), tuple(sorted(value)), tuple((key, id(item)) for key, item in sorted(value.items())))
        for param, value in optimizer.optimizer.state.items()
    }


@dataclass
class Topology:
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    use_distributed_optimizer: bool = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, create_gloo_process_groups=False
    )
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    dp_group = ps.get_data_parallel_group()
    assert dp_group.size() == 2 and ps.get_tensor_model_parallel_world_size() == 2
    tp_rank = ps.get_tensor_model_parallel_rank()
    if rank == 0:
        args.checkpoint.mkdir(parents=False, exist_ok=False)
    dist.barrier()

    stage, source = sharded_optimizer(dp_group, tp_rank)
    add_moments(stage, source)
    expected = {
        entry.name: (entry.master.clone(), entry.state["exp_avg"].clone(), entry.state["exp_avg_sq"].clone())
        for entry in adam_parameter_states(stage, source)
    }
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(args.checkpoint))
    dist.barrier()
    digest = checkpoint._metadata_sha256(args.checkpoint)
    digests = [None] * 4
    dist.all_gather_object(digests, digest)
    assert digests == [digest] * 4

    # A single rank sees metadata missing one moment; every rank must fail
    # before any existing optimizer-state entry is modified.
    stage, target = sharded_optimizer(dp_group, tp_rank)
    list(adam_parameter_states(stage, target))
    before = state_identity(target)
    original_reader = FileSystemReader.read_metadata
    if rank == 2:

        def damaged_reader(reader):
            metadata = original_reader(reader)
            metadata.state_dict_metadata = {
                key: value for key, value in metadata.state_dict_metadata.items() if not key.endswith(".exp_avg_sq")
            }
            return metadata

        FileSystemReader.read_metadata = damaged_reader
    try:
        checkpoint._prepare_optimizer_for_load(target, args.checkpoint, device="cpu", metadata_sha256=digest)
    except ValueError as error:
        assert "At least one rank" in str(error)
    else:
        raise AssertionError("cross-rank metadata failure was not propagated")
    finally:
        FileSystemReader.read_metadata = original_reader
    assert state_identity(target) == before
    dist.barrier()

    # Different ranks may see an I/O error or distinct manifest bytes. No rank
    # may commit a prepared template or enter the mutable load in either case.
    manifest_path = args.checkpoint / "manifest.json"
    if rank == 0:
        manifest_path.write_text(
            json.dumps(
                {
                    "format_version": checkpoint.FORMAT_VERSION,
                    "topology": asdict(Topology()),
                    "world_size": 4,
                    "dcp_metadata_sha256": digest,
                }
            )
        )
        (args.checkpoint / ".save_complete").touch()
    dist.barrier()
    real_read_bytes = Path.read_bytes
    for bad_rank in range(4):
        for mode in ("unreadable", "different_valid_bytes"):

            def read_bytes(path):
                if rank == bad_rank and path == manifest_path:
                    if mode == "unreadable":
                        raise OSError("injected manifest read failure")
                    return real_read_bytes(path) + b" "
                return real_read_bytes(path)

            Path.read_bytes = read_bytes
            review_driver = SimpleNamespace(topology=Topology(), optimizer=target)
            try:
                checkpoint.load(review_driver, args.checkpoint)
            except ValueError as error:
                expected_error = (
                    "At least one rank" if mode == "unreadable" else "Checkpoint manifest differs across ranks"
                )
                assert expected_error in str(error)
            else:
                raise AssertionError("rank-specific manifest failure did not propagate")
            finally:
                Path.read_bytes = real_read_bytes
            assert not getattr(review_driver, "_checkpoint_restore_failed", False)
            assert state_identity(target) == before
            dist.barrier()

    # Real, physical mutation of this test fixture: remove both moment keys
    # from TP group 0 in the serialized metadata while retaining the old pin.
    metadata_path = args.checkpoint / ".metadata"
    if rank == 0:
        raw = metadata_path.read_bytes()
        metadata = pickle.loads(raw)  # Locally generated test fixture only.
        filtered = {
            key: value
            for key, value in metadata.state_dict_metadata.items()
            if not ("dp_group_idx_0." in key and key.endswith((".exp_avg", ".exp_avg_sq")))
        }
        assert len(filtered) == len(metadata.state_dict_metadata) - 2
        metadata.state_dict_metadata = filtered
        metadata_path.write_bytes(pickle.dumps(metadata))
    dist.barrier()
    try:
        checkpoint._prepare_optimizer_for_load(target, args.checkpoint, device="cpu", metadata_sha256=digest)
    except ValueError as error:
        assert "At least one rank" in str(error)
    else:
        raise AssertionError("changed DCP metadata was not rejected")
    assert state_identity(target) == before
    dist.barrier()
    if rank == 0:
        metadata_path.write_bytes(raw)
    dist.barrier()
    assert checkpoint._metadata_sha256(args.checkpoint) == digest

    # A valid fresh optimizer uses the attested metadata to construct its
    # actual DP-local template and restore all moments and masters exactly.
    checkpoint._prepare_optimizer_for_load(target, args.checkpoint, device="cpu", metadata_sha256=digest)
    loaded = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(args.checkpoint))
    target.load_state_dict(loaded["optimizer"])
    for entry in adam_parameter_states(stage, target):
        master, avg, avg_sq = expected[entry.name]
        torch.testing.assert_close(entry.master, master, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg"], avg, rtol=0, atol=0)
        torch.testing.assert_close(entry.state["exp_avg_sq"], avg_sq, rtol=0, atol=0)
        assert entry.step == 4
    if rank == 0:
        print("PASS real TP2/DP2 same-bucket shard + padding + manifest vote + full exact restoration", flush=True)

    # For the real global DCP planner, mixed moments in the same bucket are
    # rejected at save time due to incomplete tensor coverage. Different
    # buckets avoid that planner restriction and still exercise true DP.
    mixed_path = args.checkpoint.parent / (args.checkpoint.name + ".mixed")
    if rank == 0:
        mixed_path.mkdir(parents=False, exist_ok=False)
    dist.barrier()
    stage, source = separate_bucket_optimizer(dp_group, tp_rank)
    add_moments(stage, source)
    for entry in adam_parameter_states(stage, source):
        if entry.name == "second":
            entry.state.clear()
    mixed_expected = {
        entry.name: (entry.master.clone(), {key: value.clone() for key, value in entry.state.items()})
        for entry in adam_parameter_states(stage, source)
    }
    dist_checkpointing.save({"optimizer": template(source, loading=False)}, str(mixed_path))
    dist.barrier()
    mixed_digest = checkpoint._metadata_sha256(mixed_path)
    stage, target = separate_bucket_optimizer(dp_group, tp_rank)
    list(adam_parameter_states(stage, target))
    checkpoint._prepare_optimizer_for_load(target, mixed_path, device="cpu", metadata_sha256=mixed_digest)
    mixed = dist_checkpointing.load({"optimizer": template(target, loading=True)}, str(mixed_path))
    target.load_state_dict(mixed["optimizer"])
    for entry in adam_parameter_states(stage, target):
        master, states = mixed_expected[entry.name]
        torch.testing.assert_close(entry.master, master, rtol=0, atol=0)
        assert set(entry.state) == set(states)
        for key, value in states.items():
            torch.testing.assert_close(entry.state[key], value, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    dist.barrier()
    if rank == 0:
        print("PASS real TP2/DP2 separate-bucket mixed exact restoration", flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
