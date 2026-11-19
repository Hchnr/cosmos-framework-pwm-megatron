"""One-update intervention on the nonpersistent UND warmup flag.

This diagnostic deliberately skips final checkpoint writes. It records actual
inputs/loss/norm, and is not a replacement for strict restart acceptance.
"""

import argparse
import json
import os
from pathlib import Path

import tomllib
import torch
import torch.distributed as dist

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.training_backends.megatron_tp_pp import train
from cosmos_framework.utils import distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--disable-compile", action="store_true", help="Isolate compiled execution while retaining AC")
    parser.add_argument("--restore-compute-from-master", action="store_true")
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path("examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / ".save_complete").exists():
        raise ValueError("Probe requires a completed initial checkpoint")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    bundle = json.loads((args.bundle / "manifest.json").read_text())
    root = args.output_root.resolve()
    if root.is_relative_to(checkpoint) or root.is_relative_to(args.bundle.resolve()):
        raise ValueError("Probe output must be outside the immutable checkpoint and bundle")
    os.environ.update(
        OUTPUT_ROOT=str(root / "runtime"),
        IMAGINAIRE_OUTPUT_ROOT=str(root / "runtime"),
        BASE_CHECKPOINT_PATH=bundle["source"],
        MEGATRON_TP_PP_IMPORT_PATH=str(args.bundle.resolve()),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()
    if dist.get_world_size() != manifest["world_size"]:
        raise ValueError("Probe requires original checkpoint topology")
    recipe = tomllib.loads(args.toml.read_text())
    job = root / "runtime" / recipe["job"]["project"] / recipe["job"]["group"] / "und_resume_probe"
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=False)
        (job / "checkpoints").mkdir(parents=True)
        (job / "checkpoints" / checkpoint.name).symlink_to(checkpoint, target_is_directory=True)
    dist.barrier()
    topology = manifest["topology"]
    config = load_experiment_from_toml(
        str(args.toml),
        extra_overrides=[
            "job.name=und_resume_probe",
            "trainer.seed=42",
            f"trainer.max_iter={manifest['iteration'] + 1}",
            "checkpoint.save_iter=100000",
            "trainer.callbacks.compile_tokenizer.enabled=false",
            "trainer.cudnn.deterministic=true",
            "trainer.cudnn.benchmark=false",
            *(["model.config.compile.enabled=false"] if args.disable_compile else []),
            f"model.config.parallelism.megatron.logical_data_parallel_size={topology['logical_data_parallel_size']}",
            f"model.config.parallelism.megatron.global_packed_microbatches_per_update={topology['global_packed_microbatches_per_update']}",
        ],
    )

    def intervene(driver):
        if driver.iteration != manifest["iteration"]:
            raise ValueError("Probe did not restore the requested initial state")
        adapter = driver.stage.input_adapter
        before = getattr(adapter, "_warm_full_und", None) if adapter is not None else None
        if adapter is not None:
            adapter._warm_full_und = False
        rebuilt = 0
        if args.restore_compute_from_master:
            from megatron.core import parallel_state as ps

            from cosmos_framework.training_backends.megatron_tp_pp.optimizer import adam_parameter_states

            if ps.get_data_parallel_world_size() != 1:
                raise ValueError("Direct master-to-compute intervention requires DP1")
            with torch.no_grad():
                for entry in adam_parameter_states(driver.stage, driver.optimizer):
                    entry.select(entry.model_parameter).copy_(entry.master)
                    rebuilt += 1
        (root / f"intervention.rank{dist.get_rank()}.json").write_text(
            json.dumps(
                {
                    "iteration": driver.iteration,
                    "warm_full_und_before": before,
                    "warm_full_und_after": False if adapter is not None else None,
                    "checkpoint_writes_disabled": True,
                    "compile_disabled": args.disable_compile,
                    "compute_parameters_rebuilt_from_master": rebuilt,
                },
                indent=2,
            )
            + "\n"
        )

    def no_checkpoint(*positional, **keywords):
        print(json.dumps({"probe_checkpoint_write_skipped": True}), flush=True)

    original = train._save
    try:
        train._save = no_checkpoint
        train.run(config, initialized_callback=intervene)
    finally:
        train._save = original


if __name__ == "__main__":
    main()
