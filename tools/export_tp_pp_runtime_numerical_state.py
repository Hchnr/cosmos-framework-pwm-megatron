# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Restore a completed production checkpoint and export its actual Adam state.

Run with torchrun on the checkpoint's original four ranks. No updates execute;
the source checkpoint is only read through a link in an isolated temporary job.
"""

import argparse
import json
import os
from pathlib import Path

import tomllib
import torch
import torch.distributed as dist

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.training_backends.megatron_tp_pp.numerical_state import export_numerical_state
from cosmos_framework.training_backends.megatron_tp_pp.train import run
from cosmos_framework.utils import distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path("examples/toml/sft_config/action128_pretrain_state_h1_58src_megatron_tp_pp.toml"),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / ".save_complete").is_file():
        raise ValueError("Cannot export an incomplete runtime checkpoint")
    saved = json.loads((checkpoint / "manifest.json").read_text())
    bundle = json.loads((args.bundle / "manifest.json").read_text())
    root = args.output_root.resolve()
    if root.is_relative_to(checkpoint) or root.is_relative_to(args.bundle.resolve()):
        raise ValueError("Numerical observations must be outside the immutable checkpoint and bundle")
    os.environ.update(
        OUTPUT_ROOT=str(root / "runtime"),
        IMAGINAIRE_OUTPUT_ROOT=str(root / "runtime"),
        MEGATRON_TP_PP_IMPORT_PATH=str(args.bundle.resolve()),
        BASE_CHECKPOINT_PATH=bundle["source"],
        COSMOS_TP_PP_PARALLEL_PREPARATION="0",
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()
    if dist.get_world_size() != saved["world_size"]:
        raise ValueError("Numerical export must restore the original checkpoint topology")
    recipe = tomllib.loads(args.toml.read_text())
    job_root = root / "runtime" / recipe["job"]["project"] / recipe["job"]["group"] / "numerical_export"
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=False)
        (job_root / "checkpoints").mkdir(parents=True)
        (job_root / "checkpoints" / checkpoint.name).symlink_to(checkpoint, target_is_directory=True)
    dist.barrier()
    topology = saved["topology"]
    overrides = [
        "job.name=numerical_export",
        "trainer.seed=42",
        f"trainer.max_iter={saved['iteration']}",
        "trainer.callbacks.compile_tokenizer.enabled=false",
        f"model.config.parallelism.megatron.logical_data_parallel_size={topology['logical_data_parallel_size']}",
        f"model.config.parallelism.megatron.global_packed_microbatches_per_update={topology['global_packed_microbatches_per_update']}",
        f"model.config.parallelism.megatron.tensor_model_parallel_size={topology['tensor_model_parallel_size']}",
        f"model.config.parallelism.megatron.pipeline_model_parallel_size={topology['pipeline_model_parallel_size']}",
    ]
    config = load_experiment_from_toml(str(args.toml), extra_overrides=overrides)

    def observe(driver):
        if driver.iteration != saved["iteration"] or driver.iteration != config.trainer.max_iter:
            raise ValueError("Refusing to export a source import or execute extra training updates")
        export_numerical_state(driver, root / "canonical", bundle["parameters"], source_checkpoint=checkpoint)

    run(config, initialized_callback=observe)


if __name__ == "__main__":
    main()
