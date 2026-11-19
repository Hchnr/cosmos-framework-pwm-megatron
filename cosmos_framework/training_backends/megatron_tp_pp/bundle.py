# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded runtime import of the independently verified canonical bundle."""

import hashlib
import json
from pathlib import Path

import torch

from .optimizer import adam_parameter_states, initialize_adam_state
from .parameter_map import stage_parameter_manifest
from .source import release_file_cache


class CanonicalBundle:
    def __init__(self, path, topology):
        self.path = Path(path)
        if not (self.path / ".save_complete").is_file():
            raise ValueError("Canonical import requires a complete conversion")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.normalization_extension = False
        if (
            self.manifest["format"] != "cosmos_tp_pp_canonical"
            or self.manifest["format_version"] != 1
            or not self.manifest["complete"]
            or self.manifest["tp"] != topology.tensor_model_parallel_size
            or self.manifest["pp"] != topology.pipeline_model_parallel_size
        ):
            raise ValueError("Incompatible canonical bundle format/topology")

    def release_import_cache(self):
        """Advise away clean, immutable import pages after every peer copied them.

        This does not unmap live tensors or modify files. Call once on rank zero
        after a WORLD barrier; model and Adam state already own CUDA storage.
        """
        paths = {shard["path"] for entry in self.manifest["parameters"].values() for shard in entry["shards"]}
        resolved = []
        for relative in paths:
            path = (self.path / relative).resolve()
            if not path.is_relative_to(self.path.resolve()):
                raise ValueError("Bundle shard escapes its checkpoint directory")
            resolved.append(path)
        release_file_cache(resolved)

    def validate_action128_extension(self, current_contract):
        from cosmos_framework.checkpoint.action128 import validate_checkpoint_digest, validate_contract_extension

        from .source import CanonicalDCPReader

        reader = CanonicalDCPReader(self.path / "logical_state" / "trainer")
        saved = json.loads(reader.read("action128_contract"))
        changed = validate_contract_extension(saved, current_contract)
        name = "net.action_normalization.contract_digest"
        validate_checkpoint_digest(saved, {name: self.read(name, 0)["state"]["model"]})
        self.normalization_extension = changed
        return saved, changed

    def read(self, name, tensor_rank):
        entry = self.manifest["parameters"][name]
        rank = tensor_rank if entry["axis"] is not None else 0
        shard = next(item for item in entry["shards"] if item["tensor_rank"] == rank)
        target = (self.path / shard["path"]).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError("Bundle shard escapes its checkpoint directory")
        result = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
        if result["canonical_name"] != name or tuple(result["state"]["model"].shape) != tuple(shard["shape"]):
            raise ValueError(f"Bundle identity/shape mismatch: {name}")
        return result

    def _stage_manifest(self, stage):
        manifest = stage_parameter_manifest(stage)
        expected = {
            name
            for name, entry in self.manifest["parameters"].items()
            if entry["pipeline_rank"] == next(iter(manifest.values())).pipeline_rank
        }
        actual = {entry.canonical_name for entry in manifest.values()}
        if actual != expected:
            raise ValueError(f"Stage import coverage mismatch: missing={expected - actual}, extra={actual - expected}")
        for entry in manifest.values():
            source = self.manifest["parameters"][entry.canonical_name]
            extension_buffer = self.normalization_extension and entry.canonical_name.startswith(
                "net.action_normalization."
            )
            if entry.axis != source["axis"] or not extension_buffer and list(entry.shape) != source["global_shape"]:
                raise ValueError(f"Canonical layout mismatch: {entry.canonical_name}")
            if bool(source["optimizer_fields"]) != entry.trainable:
                raise ValueError(f"Source trainability mismatch: {entry.canonical_name}")
        return manifest

    @torch.no_grad()
    def load_model(self, stage):
        manifest = self._stage_manifest(stage)
        state = dict(stage.named_parameters()) | dict(stage.named_buffers())
        for name, entry in manifest.items():
            if self.normalization_extension and entry.canonical_name.startswith("net.action_normalization."):
                continue  # Native extend retains the validated, expanded config buffers.
            value = self.read(entry.canonical_name, entry.tensor_rank)
            state[name].copy_(value["state"]["model"])
        normalization = getattr(stage.input_adapter, "action_normalization", None)
        if normalization is not None:
            if self.normalization_extension:
                normalization.reset_contract()
            normalization.validate_contract()
            normalization._reset_transforms()
        return manifest

    def optimizer_overrides(self, stage, config):
        """Separate all source LR/WD/step identities before MCore flattens params."""
        from megatron.core.optimizer.optimizer_config import ParamKey

        parameters, overrides = dict(stage.named_parameters()), {}
        for name, entry in self._stage_manifest(stage).items():
            if not entry.trainable:
                continue
            hyper = self.read(entry.canonical_name, entry.tensor_rank)["hyperparameters"]
            if not hyper["bias_correction"] or tuple(hyper["betas"]) != (config.adam_beta1, config.adam_beta2):
                raise ValueError(f"Source Adam math mismatch: {entry.canonical_name}")
            if hyper["eps"] != config.adam_eps:
                raise ValueError(f"Source epsilon mismatch: {entry.canonical_name}")
            signature = (float(hyper["initial_lr"]), float(hyper["weight_decay"]), int(hyper["step"].item()))
            attribute = "_cosmos_import_" + hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
            setattr(parameters[name], attribute, True)
            overrides[ParamKey(attr=attribute)] = {
                "max_lr": signature[0],
                "min_lr": 0.0,
                "wd_mult": signature[1] / config.weight_decay if config.weight_decay else 0.0,
            }
            if not config.weight_decay and signature[1]:
                raise ValueError("A nonzero source weight decay requires a nonzero base weight decay")
        return overrides

    def load_optimizer(self, stage, optimizer):
        manifest, groups = self._stage_manifest(stage), {}
        for entry in adam_parameter_states(stage, optimizer):
            layout = manifest[entry.name]
            saved = self.read(layout.canonical_name, layout.tensor_rank)
            hyper, values = saved["hyperparameters"], saved["state"]
            signature = (
                float(hyper["lr"]),
                float(hyper["initial_lr"]),
                float(hyper["weight_decay"]),
                tuple(hyper["betas"]),
                float(hyper["eps"]),
                int(hyper["step"].item()),
            )
            if groups.setdefault(id(entry.group), signature) != signature:
                raise ValueError("MCore merged incompatible source optimizer groups")
            entry.group.update(lr=signature[0], initial_lr=signature[1], weight_decay=signature[2])
            initialize_adam_state(
                entry,
                step=signature[-1],
                master=values["master_param"].to(entry.master.device),
                exp_avg=values["exp_avg"].to(entry.master.device),
                exp_avg_sq=values["exp_avg_sq"].to(entry.master.device),
            )
        return manifest
