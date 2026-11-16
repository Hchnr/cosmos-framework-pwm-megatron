"""Run with torchrun --standalone --nproc-per-node=4 (no pytest required)."""

import argparse
import copy
import json
import os

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.training_backends.megatron import MegatronRuntime


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 128, bias=False)
        self.up = nn.Linear(64, 128, bias=False)
        self.down = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from cosmos_framework.callbacks.norm_monitor import _reduce_norm_stats

    # Megatron may omit a gradient entirely on ranks with empty local shards.
    stats = {}
    if dist.get_rank() % 2 == 0:
        value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
        stats["uneven"] = {"sq_sum": value.square(), "max": value}
    _reduce_norm_stats(stats)
    expected_sum = sum((rank + 1) ** 2 for rank in range(dist.get_world_size()) if rank % 2 == 0)
    expected_max = max(rank + 1 for rank in range(dist.get_world_size()) if rank % 2 == 0)
    assert stats["uneven"]["sq_sum"].item() == expected_sum
    assert stats["uneven"]["max"].item() == expected_max
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(64, 64, bias=False), Block(), Block()).cuda()
    net[0].requires_grad_(False)
    baseline = copy.deepcopy(net)
    runtime = MegatronRuntime(
        net,
        shard_degree=dist.get_world_size() // args.replicas,
        replicate_degree=args.replicas,
        unit_types=(Block,),
        sharding_strategy=args.sharding_strategy,
    )
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=1e-3)
    ref_opt = torch.optim.AdamW((p for p in baseline.parameters() if p.requires_grad), lr=1e-3)
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

    runtime.zero_grad()
    from cosmos_framework.callbacks.grad_clip import GradClip
    from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor

    clipping = GradClip(clip_norm=0.01, error_if_nonfinite=True)
    max_output_error = 0.0
    for step in range(3):
        for micro in range(3):
            torch.manual_seed(42 + step * 100 + micro * 10 + dist.get_rank())
            x = torch.randn(17, 64, device="cuda")
            expected = baseline(x)
            with runtime.accumulate(micro == 2):
                actual = net(x)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                max_output_error = max(max_output_error, (actual - expected).abs().max().item())
                (actual.square().mean() / 3).backward()
            (expected.square().mean() / 3).backward()
        for name, param in baseline.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.div_(dist.get_world_size())
        params = dict(net.named_parameters())
        gradients = {
            name: param.grad if param.grad is not None else torch.zeros_like(param)
            for name, param in params.items()
            if param.requires_grad
        }
        runtime.prepare_state_dict(gradients)
        for name, param in baseline.named_parameters():
            if param.grad is not None:
                actual_grad = uneven_dtensor_to_full_tensor(gradients[name])
                torch.testing.assert_close(actual_grad, param.grad, rtol=1e-4, atol=1e-5)
        clipping.on_before_optimizer_step(net, opt, None, None)
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), max_norm=0.01)
        gradients = {
            name: param.grad if param.grad is not None else torch.zeros_like(param)
            for name, param in params.items()
            if param.requires_grad
        }
        runtime.prepare_state_dict(gradients)
        for name, param in baseline.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(
                    uneven_dtensor_to_full_tensor(gradients[name]), param.grad, atol=1e-6, rtol=1e-4
                )
        opt.step()
        runtime.after_step()
        ref_opt.step()
        state = net.state_dict()
        for name, expected in baseline.state_dict().items():
            actual = uneven_dtensor_to_full_tensor(state[name])
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            # Geometry diagnostics/export receive Parameters rather than state
            # tensors; their uneven-shard materialization must also be correct.
            torch.testing.assert_close(
                materialize_tensor(dict(net.named_parameters())[name]), expected, rtol=1e-4, atol=1e-5
            )
        opt.zero_grad(set_to_none=True)
        runtime.zero_grad()
        ref_opt.zero_grad(set_to_none=True)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "passed": True,
                    "ranks": dist.get_world_size(),
                    "steps": 3,
                    "gradient_accumulation": 3,
                    "replicas": args.replicas,
                    "gradient_clipping": True,
                    "max_output_error": max_output_error,
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
