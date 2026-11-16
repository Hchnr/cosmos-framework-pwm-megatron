"""EMA local-shard layout, FP32 values, evaluation swap/restore and DCP restart."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from distributed_smoke import Block
from torch import nn

from cosmos_framework.training_backends.ema import MegatronEmaModelUpdater, shard_ema_model
from cosmos_framework.training_backends.megatron import MegatronRuntime
from cosmos_framework.utils.generator.materialize_tensor import materialize_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharding-strategy", default="optim_grads_params")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    online = nn.Sequential(nn.Linear(64, 64, bias=False), Block(), Block()).cuda()
    online[0].requires_grad_(False)
    ema = copy.deepcopy(online).float().cpu().requires_grad_(False)
    expected = {n: p.detach().cuda().clone() for n, p in ema.named_parameters()}
    online.bfloat16()
    runtime = MegatronRuntime(
        online, shard_degree=2, replicate_degree=2, unit_types=(Block,), sharding_strategy=args.sharding_strategy
    )
    shard_ema_model(runtime, ema)
    worker = MegatronEmaModelUpdater(runtime)
    optimizer = torch.optim.AdamW([p for p in online.parameters() if p.requires_grad], lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        runtime.zero_grad()
        for micro in range(3):
            torch.manual_seed(step * 100 + micro + dist.get_rank())
            x = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
            with runtime.accumulate(micro == 2):
                (online(x).float().square().mean() / 3).backward()
        optimizer.step()
        runtime.after_step()
        worker.update_average(online, ema, beta=0.9)
        for name, value in online.state_dict().items():
            expected[name].mul_(0.9).add_(materialize_tensor(value).float(), alpha=0.1)
        for name, value in ema.state_dict().items():
            torch.testing.assert_close(materialize_tensor(value), expected[name], rtol=0, atol=0)
        training = {n: materialize_tensor(v).clone() for n, v in online.state_dict().items()}
        worker.cache(online.parameters())
        worker.copy_to(ema, online)
        for name, value in online.state_dict().items():
            torch.testing.assert_close(materialize_tensor(value), expected[name].bfloat16(), rtol=0, atol=0)
        with torch.no_grad():
            actual = online(x)
            reference = copy.deepcopy(nn.Sequential(nn.Linear(64, 64, bias=False), Block(), Block())).cuda().bfloat16()
            reference.load_state_dict({n: v.bfloat16() for n, v in expected.items()})
            torch.testing.assert_close(actual, reference(x), rtol=0, atol=0)
        worker.restore(online.parameters())
        for name, value in online.state_dict().items():
            torch.testing.assert_close(materialize_tensor(value), training[name], rtol=0, atol=0)
    checkpoint = Path("/tmp") / f"cosmos_megatron_ema_{os.environ['MASTER_PORT']}"
    saved = {"net": online.state_dict(), "net_ema": ema.state_dict()}
    dcp.save(saved, checkpoint_id=checkpoint)
    for parameter in ema.parameters():
        parameter.to_local().zero_()
    restored = {"net": online.state_dict(), "net_ema": ema.state_dict()}
    dcp.load(restored, checkpoint_id=checkpoint)
    online.load_state_dict(restored["net"])
    ema.load_state_dict(restored["net_ema"])
    for name, value in ema.state_dict().items():
        torch.testing.assert_close(materialize_tensor(value), expected[name], rtol=0, atol=0)
    result = dict(
        passed=True,
        steps=3,
        replicas=2,
        shards=2,
        sharding_strategy=args.sharding_strategy,
        frozen_parameters=True,
        fp32_initial_values_preserved=True,
        ema_values_bitwise_equal=True,
        evaluation_forward_bitwise_equal=True,
        training_weights_restored_bitwise=True,
        checkpoint_roundtrip=True,
    )
    if dist.get_rank() == 0:
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
