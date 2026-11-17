"""Collect uninterrupted identities from the original 58-source catalog loader."""

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
import torch.distributed as dist
from live_loader_trace import batch_identity
from omegaconf import OmegaConf

from cosmos_framework.utils import misc
from cosmos_framework.utils.lazy_config import instantiate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--gen-tokens", type=int, default=8192)
    parser.add_argument("--und-tokens", type=int)
    parser.add_argument("--initial-seeds-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault(
        "ACTION128_EPISODE_CACHE_DIR", "/share-mixed/project/eai_pwm/cache/action128_episode_metadata_v1"
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    misc.set_random_seed(seed=42, by_rank=True)
    config = OmegaConf.load(args.config).dataloader_train
    config.num_workers = config.prefetch_factor = config.packing_pool_size = 1
    config.max_samples_per_batch = 1
    if args.und_tokens is not None:
        config.fixed_und_tokens = args.und_tokens
    config.fixed_gen_tokens = args.gen_tokens
    config.max_sequence_length = int(config.fixed_und_tokens) + args.gen_tokens
    config.microbatch_timeout_sec = 0
    loader = instantiate(config)
    # Fresh data cursors, with the exact initial per-stream seeds from the
    # training run. Model initialization consumes CPU RNG before worker startup.
    if args.initial_seeds_from is not None:
        with (args.initial_seeds_from / "dataloader" / f"rank_{dist.get_rank()}.pkl").open("rb") as handle:
            checkpoint_state = pickle.load(handle)
        assert loader.schedule_identity == checkpoint_state["__schedule__"]
        assert loader.dataset_fingerprints == checkpoint_state["__fingerprints__"]
    if args.resume:
        assert args.initial_seeds_from is not None
        from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback

        JointCosmosDataLoaderStateCallback(loader).load_state_dict(checkpoint_state)
    elif args.initial_seeds_from is not None:
        loader._base_seeds = checkpoint_state["__execution__"]["base_seeds"]
    else:
        # Match StepMeasurement.on_train_start before the fresh live iterator.
        misc.set_random_seed(seed=42, by_rank=True)
    records = []
    try:
        iterator = iter(loader)
        for _ in range(args.batches):
            batch = next(iterator)
            records.append(batch_identity(batch))
            loader.commit_batch(batch)
    finally:
        loader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output.with_name(f"{args.output.stem}_rank_{dist.get_rank()}.json")
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"LIVE_LOADER_REFERENCE rank={dist.get_rank()} batches={len(records)}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
