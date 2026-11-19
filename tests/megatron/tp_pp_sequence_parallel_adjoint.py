"""Validate collective adjoints and the replicated-loss scale negative control."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from tools.tp_pp_sequence_parallel_experiment import gather, reduce_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=180))
    try:
        results = {}
        for scale in (1, world):
            x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
            y = gather(x, dist.group.WORLD)
            (y.sum() / scale).backward()
            expected = torch.full_like(x, world / scale)
            torch.testing.assert_close(x.grad, expected, atol=0, rtol=0)
            results[f"gather_loss_divisor_{scale}"] = x.grad[0, 0].item()
        partial = torch.full((world * 2, 3), float(rank + 1), device=device, requires_grad=True)
        result = reduce_scatter(partial, dist.group.WORLD)
        torch.testing.assert_close(result, torch.full_like(result, world * (world + 1) / 2), atol=0, rtol=0)
        result.sum().backward()
        torch.testing.assert_close(partial.grad, torch.ones_like(partial), atol=0, rtol=0)
        try:
            reduce_scatter(torch.ones((world * 2 - 1, 3), device=device), dist.group.WORLD)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-divisible physical length was not rejected")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": rank,
                    "world": world,
                    "backend": args.backend,
                    "loss_scale_negative_control": results,
                    "production_acceptance": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
