"""Reassign PP ownership of an immutable canonical bundle without changing TP.

Payloads are hard-linked, never rewritten. Both bundles must remain immutable;
runtime training checkpoints must be written to separate job directories.
Run the independent source verifier on the result before accepting conversion.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import canonical_parameter_owner


def repartition(source, destination, pp):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Bundle directories must be disjoint")
    if not (source / ".save_complete").is_file():
        raise ValueError("Source canonical bundle is incomplete")
    original = (source / "manifest.json").read_bytes()
    manifest = json.loads(original)
    if manifest["format"] != "cosmos_tp_pp_canonical" or manifest["format_version"] != 1 or not manifest["complete"]:
        raise ValueError("Unsupported or incomplete canonical bundle")
    if not 1 <= pp <= manifest["num_layers"]:
        raise ValueError("Invalid pipeline stage count")
    result = copy.deepcopy(manifest)
    result.update(pp=pp, complete=False)
    result["repartitioned_from"] = {
        "path": str(source),
        "pp": manifest["pp"],
        "manifest_sha256": hashlib.sha256(original).hexdigest(),
    }
    destination.mkdir(parents=True, exist_ok=False)
    checks = []

    def link(relative, target):
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("Source payload missing or escapes canonical bundle")
        before = path.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        after = target.stat()
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise ValueError("Linked payload identity differs")
        checks.append((path, before.st_size, before.st_mtime_ns))

    for index, (name, entry) in enumerate(result["parameters"].items()):
        owner = canonical_parameter_owner(name, result["num_layers"], pp)
        entry["pipeline_rank"] = owner
        if not entry["shards"]:
            raise ValueError(f"Incomplete canonical parameter: {name}")
        for shard in entry["shards"]:
            relative = Path(f"pp{owner:02d}") / f"{index:05d}.tp{shard['tensor_rank']:02d}.pt"
            link(Path(shard["path"]), destination / relative)
            shard["path"] = str(relative)
    for component in ("scheduler", "trainer", "dataloader"):
        directory = source / "logical_state" / component
        if not directory.is_dir():
            raise ValueError(f"Missing logical state: {component}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Logical state must contain ordinary immutable files")
            if path.is_file():
                relative = path.relative_to(source)
                link(relative, destination / relative)
    if (source / "manifest.json").read_bytes() != original:
        raise ValueError("Source manifest changed during repartition")
    for path, size, modified in checks:
        current = path.stat()
        if current.st_size != size or current.st_mtime_ns != modified:
            raise ValueError("Source payload changed during repartition")
    result["complete"] = True
    with (destination / "manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    (destination / ".save_complete").touch(exist_ok=False)
    return {
        "completed": True,
        "scope": "immutable hard-linked payloads with PP ownership reassigned; independent source verification required",
        "source": str(source),
        "destination": str(destination),
        "tp": result["tp"],
        "pp": pp,
        "parameters": len(result["parameters"]),
        "linked_files": len(checks),
        "independent_source_verification_passed": False,
        "training_resume_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = repartition(args.source, args.destination, args.pp)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
