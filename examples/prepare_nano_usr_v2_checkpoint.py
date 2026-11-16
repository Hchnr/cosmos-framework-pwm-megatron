# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Assemble the USR v2 restart checkpoint on CPU, preserving generation training state.

Run once from the repository root with LD_LIBRARY_PATH='' and PYTHONPATH=.
Both sources are read-only. The output is published only after exact verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.scripts._merge_understanding_dcp import (
    _allocate,
    _tensor_metadata,
    _validate_model_family,
    resolve_model_dcp,
    validate_merge_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION = REPO_ROOT / (
    "outputs/usr_v1/cosmos3-t2v/nano-128gpu/"
    "nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/"
    "checkpoints/iter_000012600"
)
UNDERSTANDING = Path(
    "/share/project/eai_pwm/models/cosmos-private-trained-models/understanding_training/"
    "r7_premix_unfrozen_samplemean_ga2_4node_20260921_r01_1056/model"
)
OUTPUT = Path("/share/project/eai_pwm/models/Cosmos3-Nano-USR-v2-iter12600-R7-1056-Resume-DCP")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(generation: Path, understanding: Path, output: Path) -> dict:
    generation_root, generation_model = resolve_model_dcp(generation)
    _, understanding_model = resolve_model_dcp(understanding)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    full = _tensor_metadata(generation_model)
    und = _tensor_metadata(understanding_model)
    # This T2V recipe has load_vlm_visual=false. Only allow unused vision weights;
    # any other unexpected/missing VLM key, shape or dtype must still fail.
    ignored = sorted(set(und) - set(full))
    if any(not key.startswith("net.language_model.visual.") for key in ignored):
        raise ValueError(f"Unexpected non-visual Understanding keys: {ignored[:8]}")
    selected = {key: value for key, value in und.items() if key not in ignored}
    _validate_model_family(generation_model, understanding_model)
    validate_merge_metadata(full, selected)
    components = ("optim", "scheduler", "trainer")
    for component in components:
        if not (generation_root / component / ".metadata").is_file():
            raise FileNotFoundError(f"Missing generation training state: {component}")

    print(f"Loading {len(full)} tensors; replacing {len(selected)} VLM tensors", flush=True)
    state = _allocate(full)
    dcp.load(state, storage_reader=FileSystemReader(generation_model), no_dist=True)
    dcp.load(
        {key: state[key] for key in selected},
        storage_reader=FileSystemReader(understanding_model),
        no_dist=True,
    )
    updated = set(selected)
    for key in selected:
        ema_key = f"net_ema.{key.removeprefix('net.')}"
        if ema_key in state:
            state[ema_key].copy_(state[key])
            updated.add(ema_key)

    staging = output.with_name(f".{output.name}.prepare-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        print("Saving merged model", flush=True)
        dcp.save(
            state,
            storage_writer=FileSystemWriter(staging / "model", thread_count=8),
            planner=CustomSavePlanner(),
        )
        print("Reading back and checking every merged tensor", flush=True)
        merged_metadata = _tensor_metadata(staging / "model")
        if set(merged_metadata) != set(full):
            raise RuntimeError("Merged model key set changed")
        actual = _allocate(merged_metadata)
        dcp.load(actual, storage_reader=FileSystemReader(staging / "model"), no_dist=True)
        for key, expected in state.items():
            value = actual[key]
            if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
                raise RuntimeError(f"Merged tensor verification failed: {key}")
        del actual, state

        # Copy rather than symlink: pruning the old run must not break this resume.
        copied_hashes = {}
        for component in components:
            print(f"Copying and verifying {component}", flush=True)
            shutil.copytree(generation_root / component, staging / component)
            for source in sorted((generation_root / component).rglob("*")):
                if source.is_file():
                    relative = source.relative_to(generation_root)
                    digest = sha256(source)
                    if digest != sha256(staging / relative):
                        raise RuntimeError(f"Training state copy differs: {relative}")
                    copied_hashes[str(relative)] = digest

        # Training DCPs may keep config.yaml at the run root instead of model/config.json.
        for source, target in (
            (generation_model / "config.json", staging / "model" / "config.json"),
            (generation_root.parent.parent / "config.yaml", staging / "source_config.yaml"),
        ):
            if source.is_file():
                shutil.copy2(source, target)
        trainer = {"iteration": 0}
        dcp.load(trainer, storage_reader=FileSystemReader(staging / "trainer"), no_dist=True)
        report = {
            "generation_checkpoint": str(generation_root),
            "understanding_checkpoint": str(understanding_model),
            "output_path": str(output),
            "source_metadata_sha256": {
                str(path): sha256(path) for path in (generation_model / ".metadata", understanding_model / ".metadata")
            },
            "iteration": trainer["iteration"],
            "full_tensors_verified": len(full),
            "vlm_tensors_replaced": len(selected),
            "generation_tensors_preserved": len(full) - len(updated),
            "unused_visual_tensors": ignored,
            "training_state_sha256": copied_hashes,
            "dataloader_state": "omitted: restart data cursors; subsequent checkpoints resume normally",
        }
        (staging / "merge_verification.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Verified resume checkpoint at step {trainer['iteration']}: {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=GENERATION)
    parser.add_argument("--understanding", type=Path, default=UNDERSTANDING)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.generation, args.understanding, args.output)
