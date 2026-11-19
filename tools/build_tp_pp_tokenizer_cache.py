# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the native tokenizer's immutable AOT cache on spare GPUs before TP/PP.

Uses the recipe's unchanged native VAE and CompileTokenizer callback. Compiling
on all four GPUs avoids serial owner-only compilation during later training.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils.lazy_config import instantiate
from tools.verify_tp_pp_native_data_replay import native_preparation


def serialize_type(value):
    if isinstance(value, type):
        return value.__module__ + "." + value.__qualname__
    raise TypeError(f"Unsupported cache identity value: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    config = load_experiment_from_toml(str(args.toml))
    tokenizer = native_preparation(config.model.config)
    compiler = instantiate(config.trainer.callbacks.compile_tokenizer)
    compiler.config = config
    compiler.process_group = dist.group.WORLD
    if not compiler.enabled:
        raise ValueError("The recipe must enable its native tokenizer compiler")
    # Bind the artifact to weights, recipe, native compiler source and runtime.
    vae = Path(os.environ["WAN_VAE_PATH"])
    weights = hashlib.sha256()
    with vae.open("rb") as handle:
        while block := handle.read(4 << 20):
            weights.update(block)
    source = Path("cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py")
    identity = {
        "vae_sha256": weights.hexdigest(),
        "tokenizer": OmegaConf.to_container(config.model.config.tokenizer, resolve=True),
        "compiler": OmegaConf.to_container(config.trainer.callbacks.compile_tokenizer, resolve=True),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "device_capability": torch.cuda.get_device_capability(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    identity = json.loads(json.dumps(identity, default=serialize_type))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    os.environ.update(
        COSMOS_TOKENIZER_AOT_CACHE_MODE="build",
        COSMOS_TOKENIZER_AOT_ARTIFACT_DIR=str(args.artifact_dir.resolve()),
        COSMOS_TOKENIZER_AOT_FINGERPRINT=fingerprint,
        COSMOS_TOKENIZER_AOT_WORK_DIR=str(args.work_dir.resolve()),
    )
    compiler.on_training_step_start(tokenizer, {}, iteration=compiler.compile_after_iterations)
    if dist.get_rank() == 0:
        (args.artifact_dir / "build_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        print(json.dumps({"passed": True, "artifact": str(args.artifact_dir), "fingerprint": fingerprint}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
