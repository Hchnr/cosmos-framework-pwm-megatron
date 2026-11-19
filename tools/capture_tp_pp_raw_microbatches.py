"""Capture real CPU microbatches and require their existing native-plan hashes.

Used to recover node-local diagnostic inputs and prepare equal-microbatch
benchmarks that include VAE/noising. Does not build a decoder or use CUDA.
"""

import argparse
import json
import os
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
from cosmos_framework.training_backends.megatron_tp_pp.rng import LogicalTrainingRNG
from cosmos_framework.utils.lazy_config import instantiate
from tools.verify_tp_pp_native_data_replay import runtime_rank_states


def cpu_loader_base_seeds(self, rank, sources):
    """The native loader seed draw consumes CPU RNG only, leaving CUDA intact."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(self.states[rank]["torch"])
        seeds = [int(torch.empty((), dtype=torch.int64).random_().item()) for _ in range(sources)]
        self.states[rank]["torch"] = torch.get_rng_state()
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-dp", type=int, required=True)
    parser.add_argument("--logical-rank", type=int, default=0)
    parser.add_argument("--microsteps", type=int, default=3)
    parser.add_argument("--runtime-checkpoint", type=Path)
    args = parser.parse_args()
    if not 0 <= args.logical_rank < args.logical_dp or not 1 <= args.microsteps <= 3:
        parser.error("Invalid logical rank or microstep count")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["OUTPUT_ROOT"] = str(args.output.resolve() / "runtime")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = os.environ["OUTPUT_ROOT"]
    config = load_experiment_from_toml(str(args.toml))
    topology = ModelParallelConfig(
        logical_data_parallel_size=args.logical_dp, global_packed_microbatches_per_update=args.logical_dp * 3
    )
    bundle = CanonicalBundle(args.bundle, topology)
    saved, changed = bundle.validate_action128_extension(
        {
            "version": 1,
            "spec": OmegaConf.to_container(config.model.config.action128_spec, resolve=True),
            "vision_spatial_position_mode": config.model.config.diffusion_expert_config.vision_spatial_position_mode,
        }
    )
    template = instantiate(config.dataloader_train, parallel_dims=SimpleNamespace(dp_coord=(0, args.logical_dp)))
    if args.runtime_checkpoint:
        states, rng = runtime_rank_states(args.runtime_checkpoint)
        states = {args.logical_rank: states[args.logical_rank]}
        rng = LogicalTrainingRNG({args.logical_rank: rng[args.logical_rank]})
        iteration = json.loads((args.runtime_checkpoint / "manifest.json").read_text())["iteration"]
    else:
        # Avoid the general preparation-RNG context's CUDA snapshot on this
        # CPU-only raw capture. No CUDA, NumPy or Python RNG draw occurs here.
        LogicalTrainingRNG.draw_loader_base_seeds = cpu_loader_base_seeds
        states, rng, source = import_logical_state(template, bundle, topology, (args.logical_rank,), saved, changed)
        iteration = source["source_trainer_iteration"]
    expected = {tuple(row["slot"]): row for row in json.loads(args.plan.read_text())}
    preparation = SimpleNamespace(
        input_image_key=config.model.config.input_image_key,
        input_video_key=config.model.config.input_video_key,
        input_video_latent_key=getattr(config.model.config, "input_video_latent_key", "video_latent"),
    )
    preparation.is_image_batch = lambda raw: OmniMoTModel.is_image_batch(preparation, raw)
    loader = LogicalDataLoader(template, states, num_workers=2, prefetch_factor=1)
    slots = logical_slots(
        topology, update=iteration, data_parallel_size=args.logical_dp, data_parallel_rank=args.logical_rank
    )
    records = []
    try:
        for slot in slots[: args.microsteps]:
            raw = next(loader)
            record = _raw_record(loader, raw, slot, preparation)
            target = expected[(slot.update, slot.micro_step, slot.logical_rank)]
            for key in ("sha256", "samples", "source", "tokens"):
                if record[key] != target[key]:
                    raise ValueError(f"Native-plan mismatch at {record['slot']}: {key}")
            torch.save(raw, args.output / f"micro{slot.micro_step:03d}.raw.pt")
            loader.commit_batch(raw)
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        loader.close()
        template.close()
    torch.save(rng.state_dict(), args.output / "original_logical_rng.pt")
    (args.output / "manifest.json").write_text(
        json.dumps({"passed": True, "scope": __doc__, "records": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
