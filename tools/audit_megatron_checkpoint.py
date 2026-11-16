"""Audit DCP chunk coverage without materializing model or optimizer weights."""

import argparse
import itertools
import json
import math
import pickle
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def audit(root):
    if not (root / ".save_complete").is_file():
        raise ValueError(f"Checkpoint has no completion marker: {root}")
    report = {"passed": True, "checkpoint": str(root), "components": {}}
    for component in ("model", "optim", "scheduler", "trainer"):
        with (root / component / ".metadata").open("rb") as handle:
            metadata = pickle.load(handle)
        tensors = elements = stored_bytes = 0
        for name, item in metadata.state_dict_metadata.items():
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
    args = parser.parse_args()
    report = audit(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
