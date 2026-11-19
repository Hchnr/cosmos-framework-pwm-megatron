"""Compare a replay plan with separate, unvirtualized native loaders.

Checks raw media/packing/augmentation, optionally native VAE/noising. Supports
the planned 52→58-source reset and a completed same-topology runtime checkpoint.
"""

import argparse
import copy
import json
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.dataflow.shared_prefetch import (
    LOADER_META_KEY,
    CosmosDataStream,
    SharedPrefetchDataLoader,
)
from cosmos_framework.training_backends.megatron_tp_pp.planning import batch_fingerprint
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader
from cosmos_framework.utils.generator.rand_state import preserve_training_rng, set_rand_state_dict
from cosmos_framework.utils.lazy_config import instantiate


def runtime_rank_states(path):
    reader = CanonicalDCPReader(path)
    loaders, rng = {}, {}
    for key in reader.metadata:
        if key.startswith("rank_state/shard_"):
            values = reader.read(key)
            if len(values) != 1:
                raise ValueError("Unexpected runtime rank-state object layout")
            saved = values[0]
            if saved["logical_loader"] is not None:
                if set(loaders).intersection(saved["logical_loader"]):
                    raise ValueError("Duplicate runtime logical rank owners")
                loaders.update(saved["logical_loader"])
                rng.update(saved["logical_training_rng"])
    return loaders, rng


def native_preparation(config):
    """Original Omni helper with a tokenizer and contract buffers, no decoder."""
    from torch import nn

    from cosmos_framework.model.generator.mot.action128 import Action128Normalization
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    class NativePreparation(OmniMoTModel):
        def set_up_parallelism(self):
            self.parallel_dims = None

        def set_up_model(self):
            self.net = nn.Module()
            self.net.action_normalization = Action128Normalization(
                self.config.action128_spec, self.config.diffusion_expert_config.vision_spatial_position_mode
            ).cuda()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return NativePreparation(config)


def verify(
    toml,
    bundle,
    plan,
    *,
    logical_dp,
    runtime_checkpoint=None,
    prepared=False,
    capture_prepared_dir=None,
    compile_tokenizer=False,
    rank_range=None,
):
    records = json.loads(Path(plan).read_text())
    reader = CanonicalDCPReader(Path(bundle) / "logical_state" / "trainer")
    iteration = (
        json.loads((Path(runtime_checkpoint) / "manifest.json").read_text())["iteration"]
        if runtime_checkpoint is not None
        else reader.read("iteration")
    )
    if not records or {record["slot"][0] for record in records} != {iteration}:
        raise ValueError("Plan must be the first update after the specified checkpoint")
    if prepared and any("prepared_noise_sha256" not in record for record in records):
        raise ValueError("Prepared verification requires a completed COSMOS_TP_PP_TRACE_PREPARED=1 plan")
    if capture_prepared_dir is not None:
        if not prepared:
            raise ValueError("Capturing prepared inputs requires --prepared")
        capture_prepared_dir.mkdir(parents=True, exist_ok=False)
    config = load_experiment_from_toml(str(toml))
    template = instantiate(config.dataloader_train, parallel_dims=SimpleNamespace(dp_coord=(0, logical_dp)))
    if template._exact_period is None:
        raise ValueError("The planned native reference requires exact_periodic source selection")
    old_contract = json.loads(reader.read("action128_contract"))
    callback = JointCosmosDataLoaderStateCallback(template)
    # Use precisely the native catalog extension validation; no model/RNG reset
    # beyond the already specified append-only dataset boundary is permitted.
    import pickle

    with (Path(bundle) / "logical_state/dataloader/rank_0.pkl").open("rb") as handle:
        old_loader = pickle.load(handle)
    if runtime_checkpoint is None and not callback.validate_action128_extension(old_loader, old_contract["spec"]):
        raise ValueError("This verifier requires the native appended-source reset")
    if OmegaConf.to_container(config.model.config.action128_spec, resolve=True) == old_contract["spec"]:
        raise ValueError("Expected the configured Action128 extension")
    ratios = Counter(template._exact_period)
    ranks = sorted({record["slot"][2] for record in records})
    if ranks != list(range(logical_dp)):
        raise ValueError("Plan must cover every configured logical data rank")
    if rank_range is not None:
        start, end = rank_range
        if not 0 <= start < end <= logical_dp:
            raise ValueError("Native rank range must satisfy 0 <= start < end <= logical DP")
        ranks = ranks[start:end]
    loaders, rank_rng = runtime_rank_states(runtime_checkpoint) if runtime_checkpoint is not None else ({}, {})
    preparation = native_preparation(config.model.config) if prepared else None
    if compile_tokenizer:
        if preparation is None:
            raise ValueError("Tokenizer compilation requires --prepared")
        compiler = instantiate(config.trainer.callbacks.compile_tokenizer)
        compiler.config = config
        compiler.on_training_step_start(preparation, {}, iteration=iteration)
    checked = []
    for rank in ranks:
        environment = {key: value for key, value in os.environ.items() if key.startswith("COSMOS_DL_STATE_")}
        streams = {}
        for source, (name, original) in enumerate(zip(template._names, template._loaders, strict=True)):
            dataset = copy.copy(original.dataset)
            dataset._distributor = copy.copy(dataset._distributor)
            dataset._dp_rank, dataset._dp_world_size = rank, logical_dp
            streams[name] = {"dataloader": CosmosDataStream(dataset, original.num_workers), "ratio": ratios[source]}
        reference = SharedPrefetchDataLoader(
            streams,
            seed=template._seed,
            selection_mode=template._selection_mode,
            num_workers=template.num_workers,
            prefetch_factor=template.prefetch_factor,
            pin_memory=False,
            worker_timeout=template.worker_timeout,
        )
        # Native extension starts a fresh loader after restoring training RNG.
        # _start draws CPU base seeds only; do not initialize a CUDA context here.
        if runtime_checkpoint is not None:
            JointCosmosDataLoaderStateCallback(reference).load_state_dict(copy.deepcopy(loaders[rank]))
            training_rng = rank_rng[rank]
        else:
            from cosmos_framework.training_backends.megatron_tp_pp.rng import RNG_FIELDS

            values = reader.read_many(f"rng_state_{rank}.{key}" for key in RNG_FIELDS)
            training_rng = {key: values[f"rng_state_{rank}.{key}"] for key in RNG_FIELDS}
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(training_rng["torch"])
                reference._base_seeds = [int(torch.empty((), dtype=torch.int64).random_()) for _ in streams]
                training_rng["torch"] = torch.get_rng_state()
        reference._preserve_resume_rng = True
        cursor = loaders[rank]["global_id"] if runtime_checkpoint is not None else 0
        expected = sorted((record for record in records if record["slot"][2] == rank), key=lambda r: r["slot"][1])
        try:
            with preserve_training_rng() if prepared else nullcontext():
                if prepared:
                    set_rand_state_dict(training_rng)
                for step, record in enumerate(expected):
                    if record["slot"][1] != step:
                        raise ValueError("Micro steps must be contiguous from the extension boundary")
                    batch = next(reference)
                    reference.commit_batch(batch)
                    # Normalize transport-only numbering to the flattened TP/PP plan;
                    # data, source, cursor and complete worker RNG remain untouched.
                    batch["logical_data_rank"] = rank
                    batch[LOADER_META_KEY]["global_id"] = (cursor + step) * logical_dp + rank
                    batch[LOADER_META_KEY]["source"] += rank * len(streams)
                    digest = batch_fingerprint(batch)
                    if digest != record["sha256"]:
                        raise AssertionError(f"Native media/packing/augmentation mismatch at {record['slot']}")
                    result = {"slot": record["slot"], "source": batch["dataset_name"], "sha256": digest}
                    if prepared:
                        with torch.no_grad():
                            native = preparation.prepare_generation_training_inputs(batch, iteration)
                        inputs = {
                            "gen_data_noised": native["gen_data_noised"],
                            "timesteps": native["timesteps_vision"],
                            "timesteps_action": native["timesteps_action"],
                            "timesteps_sound": native["timesteps_sound"],
                            "is_image_batch": native["gen_data_clean"].is_image_batch,
                        }
                        actual = batch_fingerprint(inputs, allow_cuda=True)
                        if actual != record["prepared_noise_sha256"]:
                            raise AssertionError(f"Native VAE/noise/timestep mismatch at {record['slot']}")
                        result["prepared_noise_sha256"] = actual
                        if capture_prepared_dir is not None:
                            # torch.load(map_location='cpu') reconstructs the original
                            # dataclasses, without a second packing implementation.
                            torch.save(
                                native["packed_sequence"],
                                capture_prepared_dir / f"rank{rank}.micro{step}.pt",
                            )
                            torch.save(inputs, capture_prepared_dir / f"rank{rank}.micro{step}.loss.pt")
                    if prepared and "prepared_packed_sha256" in record:
                        packed = batch_fingerprint(native["packed_sequence"], allow_cuda=True)
                        if packed != record["prepared_packed_sha256"]:
                            raise AssertionError(f"Native prepared text/masks/packing mismatch at {record['slot']}")
                        result["prepared_packed_sha256"] = packed
                    checked.append(result)
        finally:
            reference.close()
            for key in list(os.environ):
                if key.startswith("COSMOS_DL_STATE_") and key not in environment:
                    os.environ.pop(key)
            os.environ.update(environment)
        print(json.dumps({"native_checked_rank": rank, "checked_slots": len(checked)}), flush=True)
    return {
        "passed": True,
        "scope": "separate native per-logical-rank loaders and optional native VAE/noising versus TP/PP replay",
        "logical_data_parallel_size": logical_dp,
        "checked_logical_data_ranks": ranks,
        "checked_slots": len(checked),
        "slots": checked,
        "noise_equality_independently_checked": prepared,
        "native_tokenizer_compile_requested": compile_tokenizer,
        "production_576_slot_raw_data_accepted": logical_dp == 192 and len(checked) == 576,
        "production_capacity_576_slot_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--logical-dp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-checkpoint", type=Path)
    parser.add_argument("--prepared", action="store_true", help="Use one CUDA GPU for native VAE/noise verification")
    parser.add_argument("--capture-prepared-dir", type=Path, help="Save verified native packed inputs for math replay")
    parser.add_argument("--compile-tokenizer", action="store_true", help="Use the recipe's native AOT callback")
    parser.add_argument(
        "--rank-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Verify a disjoint rank subset; combine reports for full acceptance",
    )
    args = parser.parse_args()
    os.environ.setdefault("OUTPUT_ROOT", str(args.output.parent.resolve() / "native_reference_runtime"))
    report = verify(
        args.toml,
        args.bundle,
        args.plan,
        logical_dp=args.logical_dp,
        runtime_checkpoint=args.runtime_checkpoint,
        prepared=args.prepared,
        capture_prepared_dir=args.capture_prepared_dir,
        compile_tokenizer=args.compile_tokenizer,
        rank_range=args.rank_range,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "slots"}))


if __name__ == "__main__":
    main()
