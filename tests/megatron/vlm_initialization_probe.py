"""Measure VLM construction peak and hash exact sharded masters across init paths."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from vlm_smoke import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    options = SimpleNamespace(replicas=2, compile=True, sharding_strategy="optim_grads_params", moe=False)
    model, _ = build("megatron", args.model_path, options)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        local = value.to_local() if hasattr(value, "to_local") else value
        digest.update(name.encode())
        digest.update(str((tuple(local.shape), local.dtype)).encode())
        digest.update(local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    row = dict(rank=dist.get_rank(), master_sha256=digest.hexdigest(), initialization_seconds=elapsed)
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, row)
    if dist.get_rank() == 0:
        result = dict(
            cpu_initialization=os.environ.get("COSMOS_MEGATRON_CPU_INITIALIZATION", "0") == "1",
            model_path=str(args.model_path),
            sharding_strategy="optim_grads_params",
            replicas=2,
            peak_allocated_bytes=peak.item(),
            ranks=ranks,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
