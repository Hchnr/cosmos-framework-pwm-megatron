# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a vanilla Qwen3-VL checkpoint into a pure-Understanding OmniMoT DCP.

The output is a model-only warm-start checkpoint. It intentionally contains no
optimizer, scheduler, trainer, dataloader, generation, action, sound, or EMA
state. Existing source files are opened read-only, and an existing output path
is never overwritten.
"""

from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_DEVICE": "cpu"})

import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.inference.common.config import unstructure_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SOURCE = Path("/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct")
_DEFAULT_OUTPUT = Path("/home/baiyu/project/model_ckpts/Qwen3-VL-4B-OmniMoT-Understanding-DCP")
_DEFAULT_RECIPE = _ROOT / "examples/toml/sft_config/qwen3vl_v2_0a_understanding_qwen4b.toml"
_MODEL_PREFIX = "net.language_model."
_MAX_SHARD_SIZE = 5 * 1024**3


class Args(pydantic.BaseModel):
    source_path: Annotated[Path, tyro.conf.arg(aliases=("-i",))] = _DEFAULT_SOURCE
    """Local Hugging Face Qwen3-VL checkpoint directory."""

    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))] = _DEFAULT_OUTPUT
    """New model-only DCP directory. Must not already exist."""

    recipe_path: Path = _DEFAULT_RECIPE
    """Pure-Understanding Qwen3-VL-4B TOML used to build the exact target topology."""

    check_finite: bool = True
    """Scan every floating-point target tensor for NaN or Inf before saving."""


def _pure_understanding_overrides() -> list[str]:
    """Return topology overrides that make the conversion target unambiguous."""

    return [
        "+model.config.vlm_config.model_instance.config.include_visual=true",
        "model.config.vlm_config.pretrained_weights.enabled=false",
        "model.config.load_vlm_visual=true",
        "model.config.load_vision_tokenizer=false",
        "model.config.vision_gen=false",
        "model.config.action_gen=false",
        "model.config.sound_gen=false",
        "model.config.predict_text_tokens=true",
        "model.config.ema.enabled=false",
        "model.config.lora_enabled=false",
        "model.config.compile.enabled=false",
        "model.config.activation_checkpointing.mode=none",
    ]


def _build_target_config(source_path: Path, recipe_path: Path) -> Cosmos3OmniConfig:
    """Build the same model config used by the 4B training recipe."""

    os.environ["QWEN3_VL_4B_PATH"] = str(source_path)
    experiment = load_experiment_from_toml(
        recipe_path,
        extra_overrides=_pure_understanding_overrides(),
    )
    model_dict = unstructure_config(experiment.model)
    _validate_target_config(model_dict, source_path)

    # Keep the framework-native import paths in config.json. The public alias
    # table on this branch does not yet cover every type used by the 4B recipe,
    # whereas these internal strings round-trip through Cosmos3OmniConfig and
    # are exactly what the training loader consumes.
    return Cosmos3OmniConfig(model=model_dict)


def _validate_target_config(model_dict: dict[str, Any], source_path: Path) -> None:
    config = model_dict["config"]
    expected = {
        "load_vlm_visual": True,
        "load_vision_tokenizer": False,
        "vision_gen": False,
        "action_gen": False,
        "sound_gen": False,
        "predict_text_tokens": True,
        "lora_enabled": False,
    }
    actual = {key: config[key] for key in expected}
    if actual != expected:
        raise ValueError(f"Target is not pure Understanding: expected {expected}, got {actual}")
    if config["ema"]["enabled"]:
        raise ValueError("Target EMA must be disabled for a model-only warm-start DCP")

    pretrained = config["vlm_config"]["pretrained_weights"]
    if pretrained["enabled"]:
        raise ValueError("Target construction must not auto-load pretrained weights")
    if Path(pretrained["backbone_path"]).resolve() != source_path.resolve():
        raise ValueError(
            "Resolved Qwen backbone does not match --source-path: "
            f"{pretrained['backbone_path']} != {source_path}"
        )
    model_instance = config["vlm_config"]["model_instance"]
    if model_instance["config"].get("include_visual") is not True:
        raise ValueError("The family-native Qwen visual tower is not enabled")


def _source_tensor_keys(source_path: Path) -> set[str]:
    """Read safetensors key metadata without loading tensor payloads."""

    index_files = sorted(source_path.glob("*.safetensors.index.json"))
    if index_files:
        keys: set[str] = set()
        for index_file in index_files:
            index = json.loads(index_file.read_text())
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Invalid safetensors index without weight_map: {index_file}")
            keys.update(weight_map)
        return keys

    tensor_files = sorted(source_path.glob("*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {source_path}")
    keys = set()
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _is_visual_key(key: str) -> bool:
    return key.startswith("visual.") or key.startswith("model.visual.") or ".visual." in key


def _validate_loaded_keys(
    target_keys: set[str],
    loaded_keys: set[str],
    source_keys: set[str],
    *,
    tied_embeddings: bool,
) -> tuple[int, int]:
    """Validate full VLM and visual-tower coverage after the HF load."""

    source_visual = {key for key in source_keys if _is_visual_key(key)}
    target_visual = {key for key in target_keys if _is_visual_key(key)}
    if not source_visual:
        raise ValueError("Source checkpoint has no Qwen visual tensors")
    if not target_visual:
        raise ValueError("Target model has no Qwen visual tensors")
    unloaded_visual = target_visual - loaded_keys
    if unloaded_visual:
        raise ValueError(
            f"HF conversion left {len(unloaded_visual)} visual tensor(s) unloaded; "
            f"first up to 10: {sorted(unloaded_visual)[:10]}"
        )

    allowed_missing = {key for key in target_keys if tied_embeddings and key.endswith("lm_head.weight")}
    missing = target_keys - loaded_keys - allowed_missing
    if missing:
        raise ValueError(
            f"HF conversion left {len(missing)} target tensor(s) unloaded; "
            f"first up to 10: {sorted(missing)[:10]}"
        )
    return len(source_visual), len(target_visual)


def _initialize_moe_gen_pathway(model: torch.nn.Module, loaded_keys: set[str]) -> set[str]:
    """Mirror loaded Understanding weights into OmniMoT's fixed gen pathway."""

    target_keys = set(model.state_dict())
    moe_gen_keys = {key for key in target_keys if "_moe_gen" in key}
    if not moe_gen_keys:
        raise ValueError("Unified OmniMoT target has no _moe_gen pathway")

    tied_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    allowed_tied = {key for key in target_keys if tied_embeddings and key.endswith("lm_head.weight")}
    unexpected_missing = target_keys - loaded_keys - moe_gen_keys - allowed_tied
    if unexpected_missing:
        raise ValueError(
            f"HF conversion left {len(unexpected_missing)} non-_moe_gen tensor(s) unloaded; "
            f"first up to 10: {sorted(unexpected_missing)[:10]}"
        )

    init_moe = getattr(model, "init_moe", None)
    if not callable(init_moe):
        raise TypeError("Unified OmniMoT language model does not expose init_moe()")
    init_moe()

    # CPU conversion constructs the model with to_empty(), so a gen tensor that
    # init_moe forgot would contain undefined memory. Verify every 4B gen tensor
    # against the authoritative Understanding counterpart after the copy.
    state_dict = model.state_dict()
    for gen_key in sorted(moe_gen_keys):
        und_key = gen_key.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
        if und_key not in state_dict:
            raise ValueError(f"No Understanding counterpart for unified pathway tensor {gen_key}")
        gen_tensor = state_dict[gen_key]
        und_tensor = state_dict[und_key]
        gen_local = gen_tensor.to_local() if hasattr(gen_tensor, "to_local") else gen_tensor
        und_local = und_tensor.to_local() if hasattr(und_tensor, "to_local") else und_tensor
        if gen_local.shape != und_local.shape or not torch.equal(gen_local, und_local):
            raise ValueError(f"init_moe() did not copy {und_key} exactly into {gen_key}")
    return moe_gen_keys


def _validate_outer_state_dict(state_dict: dict[str, Any]) -> None:
    unexpected = sorted(key for key in state_dict if not key.startswith(_MODEL_PREFIX))
    if unexpected:
        raise ValueError(
            "Pure-Understanding DCP unexpectedly contains non-language-model state; "
            f"first up to 10: {unexpected[:10]}"
        )
    if not state_dict:
        raise ValueError("Target model state dict is empty")


def _check_finite(state_dict: dict[str, Any]) -> None:
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.to_local() if hasattr(value, "to_local") else value
        if tensor.device.type == "meta":
            raise ValueError(f"Target tensor was not materialized: {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Target tensor contains NaN or Inf: {name}")


def _validate_dcp_metadata(model_dir: Path, expected_keys: set[str]) -> set[str]:
    metadata = FileSystemReader(model_dir).read_metadata()
    actual_keys = set(metadata.state_dict_metadata)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Saved DCP metadata does not match the target state dict: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return actual_keys


def _model_size_bytes(state_dict: dict[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state_dict.values()
        if isinstance(value, torch.Tensor)
    )


def _save_manifest(
    work_path: Path,
    *,
    source_path: Path,
    recipe_path: Path,
    source_keys: set[str],
    loaded_keys: set[str],
    initialized_moe_gen_keys: set[str],
    dcp_keys: set[str],
    source_visual_count: int,
    target_visual_count: int,
    model_size_bytes: int,
) -> None:
    manifest = {
        "format": "torch.distributed.checkpoint/model-only",
        "purpose": "Qwen3-VL-4B pure-Understanding OmniMoT warm start",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "recipe_path": str(recipe_path),
        "source_tensor_count": len(source_keys),
        "loaded_vlm_tensor_count": len(loaded_keys),
        "initialized_moe_gen_tensor_count": len(initialized_moe_gen_keys),
        "source_visual_tensor_count": source_visual_count,
        "target_visual_tensor_count": target_visual_count,
        "dcp_tensor_count": len(dcp_keys),
        "logical_model_size_bytes": model_size_bytes,
        "contains_optimizer": False,
        "contains_scheduler": False,
        "contains_trainer": False,
        "contains_dataloader": False,
        "contains_generation_heads": False,
        "contains_unified_moe_gen_pathway": True,
        "contains_action_heads": False,
        "contains_sound_heads": False,
        "contains_ema": False,
    }
    (work_path / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def convert_qwen3vl_understanding_to_dcp(args: Args) -> Path:
    source_path = args.source_path.resolve(strict=True)
    recipe_path = args.recipe_path.resolve(strict=True)
    output_path = args.output_path.expanduser().absolute()
    if not source_path.is_dir():
        raise NotADirectoryError(source_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = output_path.parent / f".{output_path.name}.inprogress-{uuid.uuid4().hex[:8]}"
    work_path.mkdir()
    print(f"Staging conversion in {work_path}")

    source_keys = _source_tensor_keys(source_path)
    print(f"Source metadata: {len(source_keys)} tensors")

    print("Building CPU pure-Understanding OmniMoT target...")
    config = _build_target_config(source_path, recipe_path)
    target = Cosmos3OmniModel(config)
    language_model = target.model.net.language_model
    target_vlm_keys = set(language_model.state_dict())

    print("Loading Qwen language and visual weights...")
    loaded_keys = load_vlm_model(
        model=language_model,
        checkpoint_path=str(source_path),
        credential_path=None,
        parallel_dims=None,
        optional_missing_patterns=[r".*_moe_gen.*"],
    )
    initialized_moe_gen_keys = _initialize_moe_gen_pathway(language_model, loaded_keys)
    materialized_vlm_keys = loaded_keys | initialized_moe_gen_keys
    source_visual_count, target_visual_count = _validate_loaded_keys(
        target_vlm_keys,
        materialized_vlm_keys,
        source_keys,
        tied_embeddings=bool(getattr(language_model.config, "tie_word_embeddings", False)),
    )

    state_dict = get_model_state_dict(target.model)
    _validate_outer_state_dict(state_dict)
    if args.check_finite:
        print("Checking target tensors for NaN and Inf...")
        _check_finite(state_dict)

    model_size = _model_size_bytes(state_dict)
    free_bytes = shutil.disk_usage(output_path.parent).free
    if free_bytes < model_size:
        raise OSError(
            f"Insufficient free space under {output_path.parent}: "
            f"need at least {model_size} bytes, have {free_bytes}"
        )

    model_dir = work_path / "model"
    thread_count = max(1, math.ceil(model_size / _MAX_SHARD_SIZE))
    print(f"Saving {len(state_dict)} DCP tensors with {thread_count} writer thread(s)...")
    dcp.save(
        state_dict=state_dict,
        storage_writer=FileSystemWriter(model_dir, thread_count=thread_count),
        planner=CustomSavePlanner(),
    )
    config.save_pretrained(model_dir)
    dcp_keys = _validate_dcp_metadata(model_dir, set(state_dict))
    _save_manifest(
        work_path,
        source_path=source_path,
        recipe_path=recipe_path,
        source_keys=source_keys,
        loaded_keys=loaded_keys,
        initialized_moe_gen_keys=initialized_moe_gen_keys,
        dcp_keys=dcp_keys,
        source_visual_count=source_visual_count,
        target_visual_count=target_visual_count,
        model_size_bytes=model_size,
    )

    work_path.rename(output_path)
    print(f"Saved validated model-only DCP to {output_path}")
    return output_path


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    convert_qwen3vl_understanding_to_dcp(args)


if __name__ == "__main__":
    main()
