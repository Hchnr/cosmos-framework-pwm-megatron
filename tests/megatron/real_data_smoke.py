"""Materialize a real batch from the unchanged 58-source catalog dataflow."""

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cosmos_framework.utils.lazy_config import instantiate


def summarize(value):
    if isinstance(value, torch.Tensor):
        result = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.is_floating_point():
            result["finite"] = bool(torch.isfinite(value).all())
        return result
    if isinstance(value, dict):
        return {str(key): summarize(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return {"length": len(value), "first": summarize(value[0]) if value else None}
    return {"type": type(value).__name__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    config = OmegaConf.load(args.config).dataloader_train
    config.num_workers = 1
    config.prefetch_factor = 1
    config.max_samples_per_batch = 1
    config.packing_pool_size = 1
    config.microbatch_timeout_sec = 0
    start = time.monotonic()
    loader = instantiate(config)
    batch = next(iter(loader))
    args.batch_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.batch_output)
    report = {
        "passed": True,
        "seed": 42,
        "sources": len(config.dataset_specs),
        "scope": "one real catalog batch, original processing, one physical worker, max one sample",
        "elapsed_seconds": time.monotonic() - start,
        "batch": summarize(batch),
        "batch_output": str(args.batch_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
