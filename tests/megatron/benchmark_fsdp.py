"""Matched FSDP2/Megatron benchmark; synthetic dense blocks, not end-to-end training.

torchrun --standalone --nproc-per-node=4 tests/megatron/benchmark_fsdp.py --backend megatron
"""

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from cosmos_framework.training_backends.megatron import MegatronRuntime


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Linear(hidden, 3 * hidden, bias=False)
        self.up = nn.Linear(hidden, 3 * hidden, bias=False)
        self.down = nn.Linear(3 * hidden, hidden, bias=False)

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("pytorch", "megatron"), required=True)
    parser.add_argument("--hidden", type=int, default=1280)
    parser.add_argument("--tokens", type=int, default=2944)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    blocks = []
    for _ in range(args.layers):
        block = checkpoint_wrapper(Block(args.hidden))
        if args.compile:
            block = torch.compile(block, fullgraph=True, dynamic=False)
        blocks.append(block)
    model = nn.Sequential(*blocks).cuda().to(torch.bfloat16)
    runtime = None
    if args.backend == "megatron":
        runtime = MegatronRuntime(model, shard_degree=dist.get_world_size(), unit_types=(type(blocks[0]),))
    else:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
        for block in model:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    times = []
    optimizer.zero_grad(set_to_none=True)
    if runtime:
        runtime.zero_grad()
    for step in range(args.warmup + args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        if step == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for micro in range(args.accumulation):
            context = runtime.accumulate(micro == args.accumulation - 1) if runtime else nullcontext()
            with context:
                loss = model(x).float().square().mean() / args.accumulation
                loss.backward()
        optimizer.step()
        if runtime:
            runtime.after_step()
        optimizer.zero_grad(set_to_none=True)
        if runtime:
            runtime.zero_grad()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - start, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if step >= args.warmup:
            times.append(elapsed.item())
    memory = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        report = vars(args) | {
            "output": str(args.output),
            "world_size": dist.get_world_size(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
            "activation_checkpointing": "full",
            "seed": 42,
            "median_step_seconds": statistics.median(times),
            "step_seconds": times,
            "peak_allocated_bytes": memory.item(),
            "scope": "synthetic dense SwiGLU stack",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
