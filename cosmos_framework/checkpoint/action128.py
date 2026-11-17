"""Resolve Action128 initialization from checkpoint metadata before any state load."""

import copy
import json

from cosmos_framework.model.generator.mot.action128 import action128_contract_digest

ACTION128_FRESH_KEYS = ("action2llm", "llm2action", "action_normalization.")


def preserve_source_normalization(saved, current):
    """Retain old affine transforms by name; all routing changes still fail closed."""
    result = copy.deepcopy(current)
    old, new = saved["spec"], result["spec"]
    domains = new["registry"]["domains"]
    for index, domain in enumerate(old["registry"]["domains"]):
        if domain not in domains:
            raise ValueError(f"Cannot preserve normalization for removed domain: {domain}")
        target = domains.index(domain)
        old_digest, new_digest = old["source_digests"][index], new["source_digests"][target]
        # The normalization digest changes with the affine. Keep every other
        # source identity field visible to the strict extension validator.
        if isinstance(old_digest, dict) and isinstance(new_digest, dict):
            new_digest["normalization"] = old_digest["normalization"]
        elif old_digest != new_digest:
            raise ValueError(f"Cannot distinguish normalization from source identity: {domain}")
        new["transforms"][target] = copy.deepcopy(old["transforms"][index])
    validate_contract_extension(saved, result)
    return result


def read_local_source_contract(path):
    """Read immutable source metadata during config assembly, before model init."""
    from types import SimpleNamespace

    from torch.distributed.checkpoint import FileSystemReader

    source = SimpleNamespace(path=str(path), backend_key=None)
    reader = SimpleNamespace(get_storage_reader=lambda path, source: FileSystemReader(path))
    return read_source_contract(reader, source)


def action128_skip_keys(metadata, configured_skips, *, mode="auto", load_training_state=False):
    if mode not in {"auto", "reset_heads", "extend", "warm_start"}:
        raise ValueError(f"Unknown action128_load_mode: {mode}")
    keys = set(metadata)
    required = {
        "net.action_normalization." + name for name in ("contract_digest", "normalization_mask", "scale", "bias")
    }
    has_contract = bool(required & keys)
    if mode in {"extend", "warm_start"} and not has_contract:
        raise ValueError(f"Action128 {mode} requires a trained Action128 checkpoint, not a legacy base")
    if has_contract and not required <= keys:
        raise ValueError("Incomplete Action128 checkpoint contract; refusing to treat it as a legacy base")
    head_keys = {f"net.{head}.{name}.weight" for head in ("action2llm", "llm2action") for name in ("fc", "bias")}
    if has_contract and not head_keys <= keys:
        raise ValueError("Action128 checkpoint is missing trained projection parameters")
    if mode == "warm_start":
        if load_training_state or configured_skips:
            raise ValueError("Action128 warm_start requires model-only loading and no configured skipped keys")
        return ["action_normalization."]
    reset = mode == "reset_heads" or not has_contract
    skips = list(configured_skips)
    if load_training_state and (reset or skips):
        raise ValueError("Action128 full-state loading requires a complete compatible checkpoint and no skipped keys")
    if reset:
        skips = list(dict.fromkeys([*skips, *ACTION128_FRESH_KEYS]))
    elif any(pattern in key for pattern in skips for key in required | head_keys):
        raise ValueError("Use action128_load_mode=reset_heads explicitly to replace trained Action128 projections")
    return skips


def checkpoint_contract(model):
    """JSON metadata is stored with trainer state, not in the model's parameter namespace."""
    normalizer = model.net.action_normalization
    return {
        "version": 1,
        "spec": normalizer.spec,
        "vision_spatial_position_mode": normalizer.vision_spatial_position_mode,
    }


def validate_contract_extension(saved, current, *, allow_normalization_migration=False):
    """Compare by domain name: sorted normalization row IDs are allowed to move."""
    if saved.get("version") != 1 or current.get("version") != 1:
        raise ValueError("Unsupported Action128 checkpoint contract version")
    if saved["vision_spatial_position_mode"] != current["vision_spatial_position_mode"]:
        raise ValueError("Action128 extend cannot change spatial RoPE mode")
    old, new = saved["spec"], current["spec"]
    a, b = old["projection_routing"], new["projection_routing"]
    from cosmos_framework.data.generator.action.projection_routing import (
        PROJECTION_HEAD_IDS,
        PROJECTION_ROUTING_VERSION,
        PUBLISHED_PROJECTION_FAMILY_IDS,
    )

    registered = {**PROJECTION_HEAD_IDS, **PUBLISHED_PROJECTION_FAMILY_IDS}
    published_warm_start = (
        allow_normalization_migration
        and a["version"] in {"action128-projection-routing-v1", PROJECTION_ROUTING_VERSION}
        and b["version"] == PROJECTION_ROUTING_VERSION
        and all(registered.get(key) == value for key, value in b["keys"].items())
    )
    if (
        (a["version"] != b["version"] and not published_warm_start)
        or a["num_slots"] != b["num_slots"]
        or b["num_slots"] != 64
    ):
        raise ValueError("Action128 extend cannot change projection version or slot count")
    if any(b["keys"].get(key) != value for key, value in a["keys"].items()):
        raise ValueError("Action128 extend cannot remove or renumber existing heads")
    additions = set(b["keys"]) - set(a["keys"])
    if not published_warm_start and any(b["keys"][key] <= max(a["keys"].values()) for key in additions):
        raise ValueError("Action128 new heads must be appended after existing head IDs")

    def records(spec):
        registry = spec["registry"]
        domains = registry["domains"]
        if len(domains) != len(set(domains)):
            raise ValueError("Duplicate Action128 normalization domain")
        return {
            domain: (registry["embodiments"][embodiment], head, transform, digest)
            for domain, embodiment, head, transform, digest in zip(
                domains,
                registry["domain_embodiments"],
                registry["domain_projection_ids"],
                spec["transforms"],
                spec["source_digests"],
                strict=True,
            )
        }

    if old["registry"]["version"] != new["registry"]["version"]:
        raise ValueError("Action128 extend cannot change registry version")
    before, after = records(old), records(new)
    if allow_normalization_migration:
        return saved != current
    for domain, record in before.items():
        if after.get(domain) != record:
            raise ValueError(f"Action128 extend changed or removed old domain/normalization: {domain}")
    return saved != current


def validate_checkpoint_digest(contract, digests):
    """Bind human-readable metadata to the actual model (and EMA) checkpoint."""
    expected = list(action128_contract_digest(contract["spec"], contract["vision_spatial_position_mode"]))
    for name, tensor in digests.items():
        if tensor.tolist() != expected:
            raise ValueError(f"Action128 source contract does not match checkpoint tensor {name}")


def read_source_contract(checkpointer, source):
    """Prefer immutable metadata, then model-only or legacy source configuration."""
    import os

    import torch.distributed.checkpoint as dcp
    import yaml

    from cosmos_framework.utils.easy_io import easy_io

    trainer_path = os.path.join(source.path, "trainer")
    if easy_io.exists(os.path.join(trainer_path, ".metadata"), backend_key=source.backend_key):
        reader = checkpointer.get_storage_reader(trainer_path, source)
        if "action128_contract" in reader.read_metadata().state_dict_metadata:
            state = {"action128_contract": ""}
            dcp.load(state, storage_reader=reader, no_dist=True)
            return json.loads(state["action128_contract"])
    # Model-only initialization keeps the original config beside model/. Training
    # writes it at <run>/config.yaml, two levels above checkpoints/iter_*.
    config_paths = (
        os.path.join(source.path, "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(source.path.rstrip("/"))), "config.yaml"),
    )
    path = next((path for path in config_paths if easy_io.exists(path, backend_key=source.backend_key)), None)
    if path is None:
        raise ValueError("Action128 source needs saved contract metadata or the source's original config.yaml")
    config = yaml.safe_load(easy_io.load(path, file_format="txt", backend_key=source.backend_key))
    model = config["model"]["config"]
    return {
        "version": 1,
        "spec": model["action128_spec"],
        "vision_spatial_position_mode": model.get("diffusion_expert_config", {}).get(
            "vision_spatial_position_mode", "canvas"
        ),
    }
