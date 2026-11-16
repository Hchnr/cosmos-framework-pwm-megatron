"""Paired-process standalone pretrained VLM timing with GA, vision and compile."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from vlm_smoke import build

from cosmos_framework.utils.distributed import ddp_sync_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("pytorch", "megatron"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--sharding-strategy", default="optim_grads")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    assert args.tokens >= 68 and args.steps > args.warmup >= 1
    args.compile, args.moe, args.learning_rate = True, False, 1e-5
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    model, optimizer = build(args.backend, args.model_path, args)
    config = model.hf_config
    torch.manual_seed(42 + dist.get_rank())
    ids = torch.randint(0, 200, (1, args.tokens), device="cuda")
    ids[:, 1], ids[:, 2:66], ids[:, 66] = (
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    )
    vision = config.vision_config
    pixels = torch.randn(256, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2, device="cuda")
    batch = dict(
        input_ids=ids,
        labels=ids.clone(),
        attention_mask=torch.ones_like(ids),
        pixel_values=pixels,
        image_grid_thw=torch.tensor([[1, 16, 16]], device="cuda"),
    )
    batch["labels"][:, 1:67] = -100
    digest = hashlib.sha256()
    for name, value in sorted(batch.items()):
        digest.update(name.encode())
        digest.update(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    observations = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad()
        for micro in range(3):
            with ddp_sync_grad(model, enabled=micro == 2):
                _, loss = model.training_step(dict(batch), step)
                (loss / 3).backward()
        optimizer.step()
        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter() - started, device="cuda", dtype=torch.float64)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        assert torch.isfinite(loss).item()
        observations.append(dict(step=step, seconds=seconds.item(), loss=loss.item()))
        if dist.get_rank() == 0:
            print(json.dumps(dict(backend=args.backend, **observations[-1])), flush=True)
    local = dict(
        rank=dist.get_rank(),
        input_sha256=digest.hexdigest(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        observations=observations,
    )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local)
    if dist.get_rank() == 0:
        steady = [row["seconds"] for row in observations[args.warmup :]]
        result = dict(
            backend=args.backend,
            model_path=str(args.model_path),
            compile=True,
            attention_implementation="sdpa",
            compute_dtype="bfloat16",
            master_dtype="float32",
            gradient_dtype="float32",
            gradient_communication_dtype="float32",
            frozen_vision_encoder=True,
            frozen_mm_projector=True,
            peak_memory_includes_warmup=True,
            peak_memory_includes_initialization=False,
            full_activation_checkpointing=True,
            gradient_accumulation=3,
            optimizer="torch.optim.AdamW(fused=True)",
            learning_rate=args.learning_rate,
            world_size=dist.get_world_size(),
            replicas=args.replicas,
            megatron_sharding_strategy=args.sharding_strategy,
            sequence_tokens=args.tokens,
            image_grid_thw=[1, 16, 16],
            warmup_steps=args.warmup,
            median_seconds=statistics.median(steady),
            ranks=ranks,
            scope="Synthetic fixed visual-token batches; excludes live data loading, checkpoint I/O and VAE.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
