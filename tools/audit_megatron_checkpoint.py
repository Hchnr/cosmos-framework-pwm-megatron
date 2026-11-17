"""Audit DCP chunk coverage without materializing model or optimizer weights."""

import argparse
import itertools
import json
import math
import pickle
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def audit(root, *, flat=False):
    if flat:
        with (root / ".metadata").open("rb") as handle:
            metadata = pickle.load(handle)
        grouped = {}
        for name, item in metadata.state_dict_metadata.items():
            grouped.setdefault(name.split(".", 1)[0], {})[name] = item
        metadata_components = grouped.items()
    else:
        if not (root / ".save_complete").is_file():
            raise ValueError(f"Checkpoint has no completion marker: {root}")
        metadata_components = []
        for component in ("model", "optim", "scheduler", "trainer"):
            with (root / component / ".metadata").open("rb") as handle:
                metadata_components.append((component, pickle.load(handle).state_dict_metadata))
    report = {"passed": True, "checkpoint": str(root), "flat_probe_layout": flat, "components": {}}
    for component, state_metadata in metadata_components:
        tensors = elements = stored_bytes = 0
        for name, item in state_metadata.items():
            if not isinstance(item, TensorStorageMetadata):
                continue
            tensors += 1
            count = math.prod(item.size)
            elements += count
            stored_bytes += count * torch.empty((), dtype=item.properties.dtype).element_size()
            chunks = [chunk for chunk in item.chunks if math.prod(chunk.sizes)]
            if sum(math.prod(chunk.sizes) for chunk in chunks) != count:
                raise ValueError(f"Incomplete or duplicate coverage: {component}/{name}")
            for chunk in chunks:
                if any(
                    o < 0 or n < 0 or o + n > size
                    for o, n, size in zip(chunk.offsets, chunk.sizes, item.size, strict=True)
                ):
                    raise ValueError(f"Out-of-bounds chunk: {component}/{name}")
            for left, right in itertools.combinations(chunks, 2):
                if all(
                    max(a, b) < min(a + n, b + m)
                    for a, n, b, m in zip(left.offsets, left.sizes, right.offsets, right.sizes, strict=True)
                ):
                    raise ValueError(f"Overlapping chunks: {component}/{name}")
        report["components"][component] = {"tensors": tensors, "elements": elements, "bytes": stored_bytes}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flat", action="store_true", help="Audit a flat model/optimizer probe DCP layout")
    args = parser.parse_args()
    report = audit(args.checkpoint, flat=args.flat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
