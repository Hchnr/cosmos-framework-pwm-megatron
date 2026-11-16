"""FP32 optimizer masters with BF16 compute, against native FSDP2."""

import copy
import os

import torch
import torch.distributed as dist
from distributed_smoke import Block
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from cosmos_framework.training_backends.megatron import MegatronRuntime


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    source = nn.Sequential(nn.Linear(64, 64, bias=False), Block(), Block()).cuda()
    source[0].requires_grad_(False)
    full_master = {name: value.detach().clone() for name, value in source.state_dict().items()}
    target = copy.deepcopy(source).to(torch.bfloat16)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in source:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(source, mesh=mesh, mp_policy=policy)
    runtime = MegatronRuntime(
        target,
        shard_degree=dist.get_world_size(),
        unit_types=(Block,),
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        initial_master_weights=full_master,
    )
    optimizers = [
        torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3) for net in (source, target)
    ]
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    for step in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        runtime.zero_grad()
        for micro in range(3):
            torch.manual_seed(step * 100 + micro + dist.get_rank())
            x = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
            expected = source(x)
            with runtime.accumulate(micro == 2):
                actual = target(x)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)
                (actual.float().square().mean() / 3).backward()
            (expected.float().square().mean() / 3).backward()
        optimizers[0].step()
        optimizers[1].step()
        runtime.after_step()
        source_state, target_state = source.state_dict(), target.state_dict()
        for name in full_master:
            torch.testing.assert_close(
                uneven_dtensor_to_full_tensor(target_state[name]),
                source_state[name].full_tensor(),
                rtol=0.002,
                atol=0.0001,
            )
    if dist.get_rank() == 0:
        print('{"passed": true, "steps": 3, "compute_dtype": "bfloat16", "master_dtype": "float32"}')
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
