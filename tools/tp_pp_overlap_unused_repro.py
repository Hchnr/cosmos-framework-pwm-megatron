# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reproduce MCore overlap readiness with a dynamically absent parameter.

GPU mode uses actual NCCL and real MCore DDP; CPU mode replaces only transport
and device allocation while retaining the installed DDP hooks and bucket logic.
This synthetic two-update test is not a production-model qualification.
"""

import argparse
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig


class DynamicModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.always = torch.nn.Parameter(torch.ones(4, device=device))
        self.optional = torch.nn.Parameter(torch.ones(4, device=device))

    def forward(self, include_optional):
        result = self.always.square().sum()
        if include_optional:
            result = result + self.optional.square().sum()
        return result


class CompletedWork:
    def wait(self):
        return None


def run(overlap, cpu, output):
    if cpu:
        if torch.cuda.is_available():
            raise ValueError("CPU case requires CUDA_VISIBLE_DEVICES=''")
        fd, name = tempfile.mkstemp(prefix="mcore-overlap-cpu-")
        os.close(fd)
        dist.init_process_group("gloo", init_method="file://" + name, rank=0, world_size=1)
        device = torch.device("cpu")
        original_current_device = torch.cuda.current_device
        torch.cuda.current_device = lambda: device
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")
        original_current_device = None
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        conf = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1, use_cpu_initialization=cpu)
        ddp = DistributedDataParallel(
            conf,
            DistributedDataParallelConfig(overlap_grad_reduce=overlap, use_distributed_optimizer=not cpu),
            DynamicModel(device),
        )
        if len(ddp.bucket_groups) != 1:
            raise RuntimeError("Expected a single two-parameter bucket")
        bucket = ddp.bucket_groups[0]
        if cpu:

            def start(*args, **kwargs):
                if overlap:
                    bucket.grad_reduce_handle = CompletedWork()

            bucket.start_grad_sync = start
        results = []
        for update in range(2):
            ddp.zero_grad_buffer()
            # The optional parameter is used in both microbatches in update 0,
            # and only in the first microbatch in update 1.
            for micro in range(2):
                with ddp.no_sync() if micro == 0 else nullcontext():
                    ddp(update == 0 or micro == 0).backward()
            error = None
            try:
                ddp.finish_grad_sync()
            except AssertionError as exc:
                error = str(exc)
            results.append({"update": update, "finish_error": error})
            if error:
                break
        expected = (
            len(results) == 2
            and results[0]["finish_error"] is None
            and (
                "Communication call has not been issued" in (results[1]["finish_error"] or "")
                if overlap
                else results[1]["finish_error"] is None
            )
        )
        if not cpu:
            votes = [None] * dist.get_world_size()
            dist.all_gather_object(votes, expected)
            expected = all(votes)
        result = {
            "scope": "synthetic two-update DDP overlap readiness; not full-model acceptance",
            "transport": "fake CPU handle" if cpu else "real NCCL",
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "overlap": overlap,
            "results": results,
            "expected_behavior_observed": expected,
            "full_model_acceptance": False,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{'overlap' if overlap else 'control'}.rank{dist.get_rank()}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        if not expected:
            raise SystemExit("DDP overlap experiment did not match its expected behavior")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
        if original_current_device is not None:
            torch.cuda.current_device = original_current_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.overlap, args.cpu, args.output)


if __name__ == "__main__":
    main()
