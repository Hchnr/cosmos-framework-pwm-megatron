"""Measure VLM construction peak and hash exact sharded masters across init paths."""

import argparse
import hashlib
import json
import os
import resource
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
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--verify-hf", action="store_true")
    parser.add_argument("--forward-backward", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    options = SimpleNamespace(replicas=args.replicas, compile=True, sharding_strategy="optim_grads_params", moe=False)
    model, optimizer = build("megatron", args.model_path, options)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    computation = None
    if args.forward_backward:
        from cosmos_framework.training_backends.megatron import get_runtime
        from cosmos_framework.utils.distributed import ddp_sync_grad

        optimizer.zero_grad()
        torch.manual_seed(42 + dist.get_rank())
        ids = torch.randint(0, 200, (1, 17), device="cuda")
        with ddp_sync_grad(model, enabled=True):
            _, loss = model.training_step(
                dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids)), 0
            )
            loss.backward()
        norm = get_runtime(model).total_grad_norm(error_if_nonfinite=True)
        assert torch.isfinite(loss).item()
        torch.cuda.synchronize()
        computation = dict(loss=loss.item(), gradient_norm=norm.item(), optimizer_update=False)
        peak.fill_(torch.cuda.max_memory_allocated())
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        local = value.to_local() if hasattr(value, "to_local") else value
        digest.update(name.encode())
        digest.update(str((tuple(local.shape), local.dtype)).encode())
        digest.update(local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    verified = 0
    if args.verify_hf:
        from safetensors import safe_open

        index_path = args.model_path / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            with safe_open(str(args.model_path / "model.safetensors"), framework="pt", device="cpu") as reader:
                weight_map = dict.fromkeys(reader.keys(), "model.safetensors")
        for name, value in model.model.net.state_dict().items():
            name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
            if name == "lm_head.weight" and name not in weight_map and model.hf_config.tie_word_embeddings:
                name = "model.language_model.embed_tokens.weight"
            local = value.to_local() if hasattr(value, "to_local") else value
            chunks = getattr(getattr(value, "_local_tensor", None), "__create_chunk_list__", None)
            if chunks is None:
                slices = tuple(slice(None) for _ in value.shape)
            else:
                chunk = chunks()[0]
                slices = tuple(slice(o, o + n) for o, n in zip(chunk.offsets, chunk.sizes, strict=True))
            with safe_open(str(args.model_path / weight_map[name]), framework="pt", device="cpu") as reader:
                expected = reader.get_slice(name)[slices].to(dtype=local.dtype)
            torch.testing.assert_close(local.cpu(), expected, rtol=0, atol=0, msg=name)
            verified += 1
    row = dict(
        rank=dist.get_rank(),
        independently_verified_hf_tensors=verified,
        computation=computation,
        master_sha256=digest.hexdigest(),
        initialization_seconds=elapsed,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, row)
    if dist.get_rank() == 0:
        result = dict(
            meta_initialization=os.environ.get("COSMOS_MEGATRON_META_INITIALIZATION", "0") == "1",
            cpu_initialization=os.environ.get("COSMOS_MEGATRON_CPU_INITIALIZATION", "0") == "1",
            model_path=str(args.model_path),
            sharding_strategy="optim_grads_params",
            replicas=args.replicas,
            peak_allocated_bytes=peak.item(),
            ranks=ranks,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
