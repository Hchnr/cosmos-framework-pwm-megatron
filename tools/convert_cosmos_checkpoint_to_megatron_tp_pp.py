"""Convert full Cosmos DCP state to bounded canonical TP/PP tensor bundles.

This is an import format, distinct from MCore's runtime restart format. Every
model tensor (including frozen values/buffers), true FP32 master/moments and
per-parameter hyperparameters is accounted for. Original trainer, scheduler and
all logical loader/RNG state are carried separately without rank renumbering.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import canonical_parameter_owner
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader, canonical_tp_axis


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def convert(source, destination, *, tp=2, pp=2, layers=36, only_layer=None, dry_run=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and output must be disjoint trees")
    if not (source / ".save_complete").is_file():
        raise ValueError("Source checkpoint is incomplete")
    if tp not in (1, 2, 4, 8) or not 1 <= pp <= layers:
        raise ValueError("Unsupported TP/PP topology")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    model = CanonicalDCPReader(source / "model")
    optim = CanonicalDCPReader(source / "optim")
    entries, covered_optim = {}, set()
    for name, item in model.metadata.items():
        if not isinstance(item, TensorStorageMetadata):
            raise ValueError(f"Unregistered non-tensor model state: {name}")
        owner = canonical_parameter_owner(name, layers, pp)
        axis = canonical_tp_axis(name)
        if axis is not None and item.size[axis] % tp:
            raise ValueError(f"TP cannot divide {name}")
        optimizer_fields = [k for k in optim.metadata if k.startswith(f"state.{name}.")]
        hyperparameter_fields = [k for k in optim.metadata if k.startswith(f"param_groups.{name}.")]
        if optimizer_fields:
            required = {f"state.{name}.{k}" for k in ("master_param", "exp_avg", "exp_avg_sq")}
            if set(optimizer_fields) != required or not hyperparameter_fields:
                raise ValueError(f"Incomplete or unknown optimizer state: {name}")
            if any(
                optim.metadata[k].properties.dtype != torch.float32 or optim.metadata[k].size != item.size
                for k in required
            ):
                raise ValueError(f"Incompatible FP32 optimizer state: {name}")
        covered_optim.update(optimizer_fields + hyperparameter_fields)
        entries[name] = {
            "global_shape": list(item.size),
            "dtype": str(item.properties.dtype),
            "axis": axis,
            "pipeline_rank": owner,
            "optimizer_fields": optimizer_fields,
            "hyperparameter_fields": hyperparameter_fields,
            "shards": [],
        }
    if covered_optim != set(optim.metadata):
        raise ValueError(f"Unmapped optimizer state: {sorted(set(optim.metadata) - covered_optim)[:10]}")
    manifest = {
        "format": "cosmos_tp_pp_canonical",
        "format_version": 1,
        "source": str(source),
        "tp": tp,
        "pp": pp,
        "num_layers": layers,
        "complete": False,
        "parameters": entries,
    }
    if dry_run:
        return manifest
    destination.mkdir(parents=True)
    for index, (name, entry) in enumerate(entries.items()):
        if only_layer is not None and f".layers.{only_layer}." not in name:
            continue
        hyperparameters = {
            k.removeprefix(f"param_groups.{name}."): optim.read(k) for k in entry["hyperparameter_fields"]
        }
        if hyperparameters:
            if not hyperparameters.get("bias_correction", False):
                raise ValueError(f"Bias correction is required: {name}")
            if set(hyperparameters) != {"lr", "initial_lr", "bias_correction", "betas", "eps", "weight_decay", "step"}:
                raise ValueError(f"Unknown optimizer hyperparameters: {name}: {hyperparameters.keys()}")
        axis = entry["axis"]
        for rank in range(tp):
            # Replicated values are stored once; target TP ranks refer to the
            # same file. Sliced matrices are read independently from source DCP.
            if axis is None and rank:
                continue
            state = {"model": model.read(name, axis=axis, rank=rank, size=tp)}
            for key in entry["optimizer_fields"]:
                state[key.rsplit(".", 1)[-1]] = optim.read(key, axis=axis, rank=rank, size=tp)
            relative = Path(f"pp{entry['pipeline_rank']:02d}") / f"{index:05d}.tp{rank:02d}.pt"
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            checksums = {k: digest(v) for k, v in state.items()}
            torch.save({"canonical_name": name, "state": state, "hyperparameters": hyperparameters}, path)
            # Read-back verification covers actual storage bytes, not just shapes.
            restored = torch.load(path, weights_only=False, map_location="cpu")
            if any(digest(restored["state"][k]) != checksum for k, checksum in checksums.items()):
                raise ValueError(f"Failed read-back verification: {name}")
            entry["shards"].append(
                {
                    "tensor_rank": rank,
                    "path": str(relative),
                    "shape": list(state["model"].shape),
                    "offset": 0 if axis is None else rank * state["model"].shape[axis],
                    "sha256": checksums,
                }
            )
            del state, restored
        print(json.dumps({"converted": name, "index": index + 1, "total": len(entries)}), flush=True)
    if only_layer is None:
        # Preserve DCP bytes for non-model components. Their metadata embeds
        # logical RNG rank identities, so they must not be rewritten as physical
        # TP/PP ranks or replaced with a seed. Loader contains all 192 streams.
        for component in ("scheduler", "trainer", "dataloader"):
            shutil.copytree(source / component, destination / "logical_state" / component)
        manifest["complete"] = True
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["complete"]:
        (destination / ".save_complete").touch()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--only-layer", type=int, help="Diagnostic partial conversion; never publishes completion")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = convert(
        args.source,
        args.destination,
        tp=args.tp,
        pp=args.pp,
        layers=args.layers,
        only_layer=args.only_layer,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "parameters"}))


if __name__ == "__main__":
    main()
