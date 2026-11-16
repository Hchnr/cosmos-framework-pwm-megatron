"""FSDP2 -> Megatron full AdamW state continuation with Cosmos optimizer container."""

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from distributed_smoke import Block
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from cosmos_framework.training_backends.megatron import get_runtime
from cosmos_framework.utils.generator.optimizer import OptimizersContainer


class Owner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(Block(), Block())


def optimizer(owner, kind):
    return OptimizersContainer(
        owner,
        kind,
        lr=1e-3,
        weight_decay=0.05,
        fused=True,
        lr_multipliers={"gate": 2.0},
        disable_weight_decay_for_1d_params=True,
    )


def step(owner, opt, number):
    opt.zero_grad()
    for micro in range(3):
        torch.manual_seed(number * 100 + micro + dist.get_rank())
        x = torch.randn(17, 64, device="cuda", dtype=next(owner.net.parameters()).dtype)
        runtime = get_runtime(owner)
        if runtime:
            with runtime.accumulate(micro == 2):
                loss = owner.net(x).square().mean() / 3
                loss.backward()
        else:
            loss = owner.net(x).square().mean() / 3
            loss.backward()
    opt.step()


def state(owner, opt):
    return {"model": get_model_state_dict(owner), "optimizer": opt.state_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    source = Owner().cuda().to(dtype)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
    for block in source.net:
        fully_shard(block, mesh=mesh)
    fully_shard(source.net, mesh=mesh)
    source_opt = optimizer(source, args.optimizer)
    step(source, source_opt, 1)
    checkpoint = Path("/tmp/cosmos_megatron_checkpoint_smoke")
    if dist.get_rank() == 0:
        shutil.rmtree(checkpoint, ignore_errors=True)
    dist.barrier()
    dcp.save(state(source, source_opt), checkpoint_id=checkpoint)
    target = Owner().cuda().to(dtype)
    target.net._cosmos_megatron_options = {
        "shard_degree": dist.get_world_size() // args.replicas,
        "replicate_degree": args.replicas,
        "unit_types": (Block,),
        "sharding_strategy": args.sharding_strategy,
    }
    target_opt = optimizer(target, args.optimizer)
    restored = state(target, target_opt)
    dcp.load(restored, checkpoint_id=checkpoint)
    set_model_state_dict(target, restored["model"])
    target_opt.load_state_dict(restored["optimizer"])
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    for number in (2, 3):
        step(source, source_opt, number)
        step(target, target_opt, number)
        source_state, target_state = source.net.state_dict(), target.net.state_dict()
        for name, expected in source_state.items():
            torch.testing.assert_close(
                uneven_dtensor_to_full_tensor(target_state[name]),
                expected.full_tensor(),
                atol=0.002 if dtype == torch.bfloat16 else 1e-5,
                rtol=0.01 if dtype == torch.bfloat16 else 1e-4,
            )
    # Ensure Megatron can save its own nonuniform shards, not just read FSDP2.
    dcp.save(state(target, target_opt), checkpoint_id=checkpoint / "megatron")
    roundtrip = Owner().cuda().to(dtype)
    for block in roundtrip.net:
        fully_shard(block, mesh=mesh)
    fully_shard(roundtrip.net, mesh=mesh)
    roundtrip_opt = optimizer(roundtrip, args.optimizer)
    restored = state(roundtrip, roundtrip_opt)
    dcp.load(restored, checkpoint_id=checkpoint / "megatron")
    set_model_state_dict(roundtrip, restored["model"])
    roundtrip_opt.load_state_dict(restored["optimizer"])
    step(source, source_opt, 4)
    step(roundtrip, roundtrip_opt, 4)
    for name, expected in source.net.state_dict().items():
        torch.testing.assert_close(
            roundtrip.net.state_dict()[name].full_tensor(),
            expected.full_tensor(),
            atol=0.002 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.01 if dtype == torch.bfloat16 else 1e-4,
        )
    if dist.get_rank() == 0:
        print('{"passed": true, "checkpoint": "FSDP2 -> Megatron -> FSDP2", "continued_steps": 3}')
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
