"""Compare converted bundles against original canonical source values exactly."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader


def verify(bundle, *, allow_partial=False):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["format"] != "cosmos_tp_pp_canonical" or manifest["format_version"] != 1:
        raise ValueError("Unsupported canonical conversion format")
    if not manifest["complete"] and not allow_partial:
        raise ValueError("Conversion is partial")
    if manifest["complete"] and not (bundle / ".save_complete").is_file():
        raise ValueError("Converted checkpoint has no completion marker")
    source = Path(manifest["source"])
    model, optim = CanonicalDCPReader(source / "model"), CanonicalDCPReader(source / "optim")
    if set(manifest["parameters"]) != set(model.metadata):
        raise ValueError("Canonical model coverage differs from the source")
    checked = {}
    for name, entry in manifest["parameters"].items():
        if not entry["shards"]:
            if not allow_partial:
                raise ValueError(f"Missing shards for {name}")
            continue
        parts = [
            (shard, torch.load(bundle / shard["path"], weights_only=False, map_location="cpu"))
            for shard in entry["shards"]
        ]
        axis = entry["axis"]
        if len(parts) != (1 if axis is None else manifest["tp"]):
            raise ValueError(f"Missing or duplicate TP shards for {name}")
        if sum(math.prod(shard["shape"]) for shard, _ in parts) != math.prod(entry["global_shape"]):
            raise ValueError(f"Incomplete element coverage for {name}")
        fields = ["model"] + [key.rsplit(".", 1)[-1] for key in entry["optimizer_fields"]]
        for field in fields:
            original = model.read(name) if field == "model" else optim.read(f"state.{name}.{field}")
            for index, (shard, state) in enumerate(parts):
                tensor = state["state"][field]
                if shard["tensor_rank"] != index or tuple(tensor.shape) != tuple(shard["shape"]):
                    raise ValueError(f"Invalid rank/shape metadata: {name}")
                if state["canonical_name"] != name:
                    raise ValueError(f"Wrong parameter identity in {shard['path']}")
                if axis is not None and shard["offset"] != index * tensor.shape[axis]:
                    raise ValueError(f"Non-contiguous or overlapping shard: {name}")
                expected = original if axis is None else original.narrow(axis, shard["offset"], tensor.shape[axis])
                if tensor.dtype != expected.dtype or not torch.equal(tensor, expected):
                    raise ValueError(f"Source values differ: {name}/{field}/TP{index}")
            del original
        for key in entry["hyperparameter_fields"]:
            field = key.rsplit(".", 1)[-1]
            expected = optim.read(key)
            for _, state in parts:
                actual = state["hyperparameters"][field]
                if isinstance(actual, torch.Tensor):
                    if not torch.equal(actual, expected):
                        raise ValueError(f"Hyperparameter differs: {name}/{field}")
                elif actual != expected:
                    raise ValueError(f"Hyperparameter differs: {name}/{field}")
        checked[name] = {"fields": fields, "tp_shards": len(parts)}
        del parts
        print(json.dumps({"verified": name, "checked": len(checked)}), flush=True)
    if manifest["complete"]:
        for component in ("trainer", "scheduler", "dataloader"):
            for path in (source / component).rglob("*"):
                if path.is_file():
                    copy = bundle / "logical_state" / component / path.relative_to(source / component)
                    # These components are small; verify that all bytes, including
                    # RNG and worker execution state, survived migration.
                    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(copy.read_bytes()).digest():
                        raise ValueError(f"Logical state copy differs: {path}")
    return {
        "passed": True,
        "source": str(source),
        "bundle": str(bundle),
        "full_conversion": manifest["complete"],
        "checked_parameters": checked,
        "training_resume_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.bundle, allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checked_parameters"}))


if __name__ == "__main__":
    main()
