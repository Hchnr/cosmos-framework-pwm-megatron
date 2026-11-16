# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Implementation helpers for merging a slim Understanding DCP into a full DCP."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from cosmos_framework.checkpoint.dcp import CustomSavePlanner

_GENERATION_KEY_PARTS = (
    "_moe_gen",
    "k_norm_und_for_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "image_modality_embed",
    "video_modality_embed",
    "media_modality_embed",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "sound2llm",
    "llm2sound",
    "sound_modality_embed",
)


@dataclass(frozen=True)
class MergeReport:
    output_path: str
    full_tensors: int
    understanding_tensors: int
    generation_tensors_preserved: int
    ema_tensors_updated: int
    bytes_materialized: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_dcp(path: Path) -> tuple[Path, Path]:
    """Return ``(checkpoint_root, model_component_dir)`` for either accepted layout."""
    resolved = path.expanduser().resolve()
    if (resolved / ".metadata").is_file():
        return resolved.parent, resolved
    if (resolved / "model" / ".metadata").is_file():
        return resolved, resolved / "model"
    raise FileNotFoundError(
        f"Expected a DCP .metadata file at {resolved / '.metadata'} or {resolved / 'model' / '.metadata'}"
    )


def _tensor_metadata(path: Path) -> dict[str, TensorStorageMetadata]:
    metadata = FileSystemReader(path).read_metadata().state_dict_metadata
    non_tensors = [key for key, value in metadata.items() if not isinstance(value, TensorStorageMetadata)]
    if non_tensors:
        raise TypeError(f"Model DCP contains unsupported non-tensor entries: {non_tensors[:8]}")
    return dict(metadata)  # type: ignore[arg-type]


def _is_generation_key(key: str) -> bool:
    return any(part in key for part in _GENERATION_KEY_PARTS)


def _regular_key(key: str) -> bool:
    return key.startswith("net.") and not key.startswith("net_ema.")


def validate_merge_metadata(
    full: dict[str, TensorStorageMetadata],
    understanding: dict[str, TensorStorageMetadata],
) -> None:
    forbidden = sorted(key for key in understanding if _is_generation_key(key))
    if forbidden:
        raise ValueError(f"Understanding checkpoint still contains generation parameters; first keys: {forbidden[:8]}")
    non_regular = sorted(key for key in understanding if not _regular_key(key) and not key.startswith("net_ema."))
    if non_regular:
        raise ValueError(f"Unexpected Understanding checkpoint keys: {non_regular[:8]}")

    expected_understanding = {key for key in full if _regular_key(key) and not _is_generation_key(key)}
    actual_understanding = {key for key in understanding if _regular_key(key)}
    missing = sorted(expected_understanding - actual_understanding)
    unexpected = sorted(actual_understanding - expected_understanding)
    if missing or unexpected:
        raise ValueError(
            "Understanding/full checkpoint key sets are incompatible: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    mismatches: list[str] = []
    for key in actual_understanding:
        source = understanding[key]
        target = full[key]
        if source.size != target.size or source.properties.dtype != target.properties.dtype:
            mismatches.append(
                f"{key}: understanding={tuple(source.size)}/{source.properties.dtype}, "
                f"full={tuple(target.size)}/{target.properties.dtype}"
            )
    if mismatches:
        raise ValueError(f"Understanding/full tensor shapes or dtypes differ: {mismatches[:8]}")

    if not any("language_model.model.layers." in key for key in actual_understanding):
        raise ValueError("Understanding checkpoint contains no language-model layer weights")
    if not any("language_model.model.embed_tokens.weight" in key for key in actual_understanding):
        raise ValueError("Understanding checkpoint contains no token embedding")


def _allocate(metadata: dict[str, TensorStorageMetadata]) -> dict[str, torch.Tensor]:
    return {key: torch.empty(value.size, dtype=value.properties.dtype, device="cpu") for key, value in metadata.items()}


def _copy_model_config(source_root: Path, source_model: Path, output_root: Path) -> None:
    config = source_model / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Full generation DCP has no model config: {config}")
    # Validate before publishing a copied configuration.
    json.loads(config.read_text(encoding="utf-8"))
    shutil.copy2(config, output_root / "model" / "config.json")
    checkpoint_json = source_root / "checkpoint.json"
    if checkpoint_json.is_file():
        shutil.copy2(checkpoint_json, output_root / "checkpoint.json")


def _model_family_signature(model_dir: Path) -> tuple[str, str] | None:
    """Read the architecture identity saved beside a model DCP, when available."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config.get("model", {})
    model_config = model.get("config", {}) if isinstance(model, dict) else {}
    vlm_config = model_config.get("vlm_config", {}) if isinstance(model_config, dict) else {}
    model_instance = vlm_config.get("model_instance", {}) if isinstance(vlm_config, dict) else {}
    instance_config = model_instance.get("config", {}) if isinstance(model_instance, dict) else {}
    base_config = instance_config.get("base_config", {}) if isinstance(instance_config, dict) else {}
    target = str(model_instance.get("_target", model_instance.get("_target_", "")))
    # Released Nano checkpoints use the registered short target while checkpoints
    # produced by the current TOML/Hydra path save the concrete Python target.
    # They identify the same model family; keep rejecting genuinely different
    # targets after normalizing this known serialization difference.
    if target.rsplit(".", 1)[-1] == "Qwen3VLTextForCausalLM":
        target = "qwen3_vl_text_for_causal_lm"

    json_file = str(base_config.get("json_file", ""))
    # The same bundled config may be recorded as a cosmos3:// resource or as its
    # repository-relative source path.  The basename is the stable family ID.
    json_file = json_file.rsplit("/", 1)[-1]
    return target, json_file


def _validate_model_family(generation_model: Path, understanding_model: Path) -> None:
    generation_family = _model_family_signature(generation_model)
    understanding_family = _model_family_signature(understanding_model)
    if generation_family is None or understanding_family is None:
        return
    if generation_family != understanding_family:
        raise ValueError(
            "Understanding/full checkpoint model families differ: "
            f"understanding={understanding_family}, full={generation_family}"
        )


def merge_understanding_dcp(
    generation_checkpoint_path: Path,
    understanding_checkpoint_path: Path,
    output_path: Path,
) -> MergeReport:
    """Create a full warm-start DCP using UND weights from a slim checkpoint."""
    generation_root, generation_model = resolve_model_dcp(generation_checkpoint_path)
    _, understanding_model = resolve_model_dcp(understanding_checkpoint_path)
    output = output_path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    full_metadata = _tensor_metadata(generation_model)
    understanding_metadata = _tensor_metadata(understanding_model)
    _validate_model_family(generation_model, understanding_model)
    validate_merge_metadata(full_metadata, understanding_metadata)

    state = _allocate(full_metadata)
    dcp.load(state, storage_reader=FileSystemReader(generation_model), no_dist=True)
    # Values in this subset alias ``state`` tensors, so the partial DCP load
    # overwrites Understanding weights in-place without allocating a second model.
    understanding_keys = {key for key in understanding_metadata if _regular_key(key)}
    understanding_state = {key: state[key] for key in understanding_keys}
    dcp.load(understanding_state, storage_reader=FileSystemReader(understanding_model), no_dist=True)

    ema_updates = 0
    for key in understanding_keys:
        ema_key = f"net_ema.{key.removeprefix('net.')}"
        if ema_key in state:
            state[ema_key].copy_(state[key])
            ema_updates += 1

    temporary = output.with_name(f".{output.name}.merge-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary merge directory already exists: {temporary}")
    try:
        (temporary / "model").mkdir(parents=True)
        total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        threads = max(1, math.ceil(total_bytes / (5 * 1024**3)))
        dcp.save(
            state,
            storage_writer=FileSystemWriter(temporary / "model", thread_count=threads),
            planner=CustomSavePlanner(),
        )
        _copy_model_config(generation_root, generation_model, temporary)

        temporary_metadata = _tensor_metadata(temporary / "model")
        if set(temporary_metadata) != set(full_metadata):
            raise RuntimeError("Merged DCP key set differs from the source full checkpoint")

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return MergeReport(
        output_path=str(output),
        full_tensors=len(full_metadata),
        understanding_tensors=len(understanding_keys),
        generation_tensors_preserved=sum(_is_generation_key(key) for key in full_metadata),
        ema_tensors_updated=ema_updates,
        bytes_materialized=total_bytes,
    )
