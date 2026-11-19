"""Read full-capacity logical streams without constructing a decoder or VAE.

The resulting first-update manifest is consumed by the independent native
loader verifier. This only measures raw data continuity, never training fit,
noise equality, numerical correctness, or production throughput.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots
from cosmos_framework.training_backends.megatron_tp_pp.bundle import CanonicalBundle
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.data_state import import_logical_state
from cosmos_framework.training_backends.megatron_tp_pp.logical_loader import LogicalDataLoader
from cosmos_framework.training_backends.megatron_tp_pp.planning import _raw_record
from cosmos_framework.utils.lazy_config import instantiate


def run(args):
    if args.logical_dp < 1 or args.num_workers < 0:
        raise ValueError("Logical DP must be positive and physical workers nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("OUTPUT_ROOT", str(args.output.resolve() / "runtime"))
    config = load_experiment_from_toml(str(args.toml))
    topology = ModelParallelConfig(**OmegaConf.to_container(config.model.config.parallelism.megatron, resolve=True))
    if topology.logical_data_parallel_size != args.logical_dp:
        raise ValueError("This raw audit retains the recipe's original logical DP and complete update")
    bundle = CanonicalBundle(args.bundle, topology)
    saved, changed = bundle.validate_action128_extension(
        {
            "version": 1,
            "spec": OmegaConf.to_container(config.model.config.action128_spec, resolve=True),
            "vision_spatial_position_mode": config.model.config.diffusion_expert_config.vision_spatial_position_mode,
        }
    )
    template = instantiate(config.dataloader_train, parallel_dims=SimpleNamespace(dp_coord=(0, args.logical_dp)))
    ranks = tuple(range(args.logical_dp))
    states, rng, source = import_logical_state(template, bundle, topology, ranks, saved, changed)
    # Native modality classification requires only these configured keys.
    preparation = SimpleNamespace(
        input_image_key=config.model.config.input_image_key,
        input_video_key=config.model.config.input_video_key,
        input_video_latent_key=getattr(config.model.config, "input_video_latent_key", "video_latent"),
    )
    preparation.is_image_batch = lambda raw: OmniMoTModel.is_image_batch(preparation, raw)
    loader = LogicalDataLoader(template, states, num_workers=args.num_workers, prefetch_factor=1)
    slots = logical_slots(
        topology, update=source["source_trainer_iteration"], data_parallel_size=1, data_parallel_rank=0
    )
    records = []
    started = time.monotonic()
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.toml, Path(__file__), *Path("cosmos_framework").rglob("*.py")]
    }
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    try:
        for slot in slots:
            raw = next(loader)
            records.append(_raw_record(loader, raw, slot, preparation))
            loader.commit_batch(raw)
            del raw
            if len(records) == 1 or len(records) % 32 == 0 or len(records) == len(slots):
                print(json.dumps({"raw_slots": len(records), "expected_slots": len(slots)}), flush=True)
                temporary = args.output / "plan.partial.json"
                temporary.write_text(json.dumps(records, indent=2) + "\n")
    finally:
        loader.close()
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        raise ValueError("Raw data audit source changed during execution")
    torch.save(
        {"logical_loader": loader.logical_state_dict(), "logical_training_rng": rng.state_dict()},
        args.output / "final_logical_state.pt",
    )
    (args.output / "plan.partial.json").rename(args.output / "plan.json")
    report = {
        "passed": True,
        "scope": "raw full-capacity logical data manifest only; independent native comparison pending",
        "logical_data_parallel_size": args.logical_dp,
        "global_slots": len(records),
        "actual_samples": sum(row["samples"] for row in records),
        "physical_workers": args.num_workers,
        "elapsed_seconds": time.monotonic() - started,
        "source": source,
        "independent_native_comparison_passed": False,
        "production_capacity_576_slot_acceptance": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-dp", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
