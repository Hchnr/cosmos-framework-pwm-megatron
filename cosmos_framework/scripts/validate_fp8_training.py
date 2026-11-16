# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small real-GPU FP8/FSDP/AC/compile/DCP regression, not a video-quality benchmark.

Run under torchrun. Reusing --run resumes its latest DCP. Separate processes with
--steps=5 then --steps=10 can be compared exactly to an uninterrupted 10-step run
using the per-rank result files (model, optimizer, RNG, LR and losses).
"""

import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from cosmos_framework.checkpoint.dcp import DistributedCheckpointer, ModelWrapper
from cosmos_framework.configs.base.defaults.fp8_training import FP8TrainingConfig
from cosmos_framework.model.generator.fp8_training import (
    convert_fp8_training,
    validate_fp8_optimizer,
    validate_precision_resume,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP
from cosmos_framework.utils.config import CheckpointConfig, JobConfig
from cosmos_framework.utils.generator.optimizer import build_lr_scheduler, build_optimizer
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        config = SimpleNamespace(hidden_size=128, intermediate_size=256, hidden_act="silu")
        self.mlp_moe_gen = Qwen3VLTextMLP(config)
        self.mlp = Qwen3VLTextMLP(config)

    def forward(self, x):
        return x + self.mlp_moe_gen(x)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for block in self.language_model.model.layers:
            x = block(x)
        return x


class Model(nn.Module):
    def __init__(self, recipe, *, compiled, mesh):
        super().__init__()
        self.config = SimpleNamespace()
        with torch.device("meta"):
            self.net = Network().to(torch.bfloat16)
        self.manifest = convert_fp8_training(
            self.net, FP8TrainingConfig(enabled=recipe != "bf16", recipe=recipe if recipe != "bf16" else "rowwise")
        )
        for i, block in enumerate(self.net.language_model.model.layers):
            block = checkpoint_wrapper(block, preserve_rng_state=True)
            if compiled:
                block = torch.compile(block, fullgraph=True, dynamic=False)
            fully_shard(block, mesh=mesh)
            self.net.language_model.model.layers[i] = block
        fully_shard(self.net, mesh=mesh)
        self.net.to_empty(device="cuda")
        # Deterministic values across both shard and replicate meshes. A real
        # checkpoint load below replaces these when testing warm-start/resume.
        for name, parameter in self.net.named_parameters():
            with torch.no_grad():
                parameter.fill_(0.01 if "gate_proj" in name else 0.02)

    def forward(self, x):
        return self.net(x)

    def training_precision_state(self):
        return json.dumps(self.manifest, sort_keys=True)

    def validate_training_precision_resume(self, saved, *, warm_start):
        validate_precision_resume(self.training_precision_state(), saved, warm_start=warm_start)


def local_cpu(value):
    if isinstance(value, DTensor):
        return value.to_local().detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: local_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(local_cpu(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--job-group", help="Override the recipe-named group to test same-job precision drift")
    parser.add_argument("--recipe", choices=["bf16", "rowwise", "rowwise_with_gw_hp"], default="rowwise")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--async-save", action="store_true")
    parser.add_argument("--load-path", default="")
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world % args.shard_size:
        raise ValueError("world size must be divisible by shard size")
    replicate = world // args.shard_size
    mesh = (
        init_device_mesh("cuda", (replicate, args.shard_size), mesh_dim_names=("replicate", "shard"))
        if replicate > 1
        else init_device_mesh("cuda", (args.shard_size,), mesh_dim_names=("shard",))
    )
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    job = JobConfig(project="distributed", group=args.job_group or args.recipe, name=args.run, wandb_mode="disabled")
    model = Model(args.recipe, compiled=args.compile, mesh=mesh)
    optimizer = build_optimizer(
        model,
        lr=1e-3,
        weight_decay=0,
        betas=(0.9, 0.95),
        eps=1e-6,
        fused=True,
        keys_to_select=["moe_gen"],
        optimizer_type="AdamW",
    )
    validate_fp8_optimizer(model.net, optimizer, model.manifest)
    scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler_type="LambdaCosine",
        cycle_lengths=[100],
        f_max=[1.0],
        f_min=[1.0],
        f_start=[0.0],
        warm_up_steps=[2],
        verbosity_interval=0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpointer = DistributedCheckpointer(
        CheckpointConfig(
            load_path=args.load_path,
            load_training_state=not args.weights_only,
            dcp_async_mode_enabled=args.async_save,
            dcp_load_dedup=True,
        ),
        job,
    )
    start = checkpointer.load(model, optimizer, scheduler, scaler)
    # Snapshot before any updates to verify warm-start really restores slots.
    run_dir = Path(job.path_local)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        local_cpu(
            {
                "model": ModelWrapper(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "iteration": start,
            }
        ),
        run_dir / f"loaded_rank{dist.get_rank()}.pt",
    )
    losses = []
    for iteration in range(start, args.steps):
        optimizer.zero_grad()
        total = 0.0
        for _ in range(2):
            x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
            x[-16:] = 0
            target = torch.randn_like(x)
            loss = (model(x).float() - target.float()).square().mean() / 2
            loss.backward()
            total += loss.detach().item()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad.to_local()).all()
        optimizer.step()
        scheduler.step()
        losses.append([iteration + 1, total])
    checkpointer.save(model, optimizer, scheduler, scaler, args.steps)
    checkpointer.finalize()
    result = local_cpu(
        {
            "model": ModelWrapper(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iteration": args.steps,
            "rng": get_rand_state_dict(),
            "losses": losses,
            "manifest": model.manifest,
        }
    )
    torch.save(result, run_dir / f"result_rank{dist.get_rank()}.pt")
    if dist.get_rank() == 0:
        print(json.dumps({"run": args.run, "start": start, "end": args.steps, "losses": losses}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
