"""Exercise actual CPU staging, spawn queue, background DCP write and reload."""

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from checkpoint_smoke import Owner, optimizer, state, step
from distributed_smoke import Block
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from cosmos_framework.training_backends.checkpoint_metadata import extract_chunk_metadata, restore_chunk_metadata


def writer(queue, rank, world, port, path):
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    staged, metadata = queue.get()
    restore_chunk_metadata(staged, metadata)
    dcp.save(staged, checkpoint_id=path)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    torch.manual_seed(42)
    owner = Owner().cuda().to(torch.bfloat16)
    owner.net._cosmos_megatron_options = {
        "shard_degree": dist.get_world_size() // args.replicas,
        "replicate_degree": args.replicas,
        "unit_types": (Block,),
        "sharding_strategy": args.sharding_strategy,
    }
    opt = optimizer(owner, "FusedAdam")
    step(owner, opt, 1)
    original = state(owner, opt)
    expected = {name: uneven_dtensor_to_full_tensor(t).clone() for name, t in original["model"].items()}
    metadata = extract_chunk_metadata(original)
    assert metadata
    staged = _create_cpu_state_dict(original, pin_memory=True, share_memory=True)
    _copy_state_dict(original, staged)
    torch.cuda.synchronize()
    path = Path("/tmp/cosmos_megatron_async_smoke")
    if dist.get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()
    context = torch.multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=writer, args=(queue, dist.get_rank(), dist.get_world_size(), int(os.environ["MASTER_PORT"]) + 2, path)
    )
    process.start()
    queue.put((staged, metadata))
    # Training proceeds while the background writer owns the staged snapshot.
    step(owner, opt, 2)
    expected_after_step = {name: tensor.to_local().clone() for name, tensor in get_model_state_dict(owner).items()}
    process.join(timeout=120)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    assert process.exitcode == 0, f"background writer exit code: {process.exitcode}"
    # Corrupt current weights, then load the background checkpoint in-place.
    with torch.no_grad():
        for parameter in owner.net.parameters():
            parameter.zero_()
    restored = state(owner, opt)
    dcp.load(restored, checkpoint_id=path)
    set_model_state_dict(owner, restored["model"])
    opt.load_state_dict(restored["optimizer"])
    for name, tensor in get_model_state_dict(owner).items():
        torch.testing.assert_close(uneven_dtensor_to_full_tensor(tensor), expected[name], rtol=0, atol=0)
    step(owner, opt, 2)
    for name, tensor in get_model_state_dict(owner).items():
        torch.testing.assert_close(tensor.to_local(), expected_after_step[name], rtol=0, atol=0)
    if dist.get_rank() == 0:
        print(
            '{"passed": true, "async_cpu_staging_spawn_queue_reload": true, "weights_bitwise_equal": true, "continued_training_bitwise_equal": true}'
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
