"""Check tensor inputs captured by full-model forward hooks on each rank/microbatch."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.source.glob("input_rank_*_micro_*.pt"))
    if not files or [p.name for p in files] != sorted(p.name for p in args.candidate.glob("input_rank_*_micro_*.pt")):
        raise ValueError("Input capture files are absent or differ")
    tensors = 0
    differences = []
    for path in files:
        source = torch.load(path, map_location="cpu", weights_only=True)
        candidate = torch.load(args.candidate / path.name, map_location="cpu", weights_only=True)
        if not source or source.keys() != candidate.keys():
            raise ValueError(f"Input tensor paths are absent or differ: {path.name}")
        for name, expected in source.items():
            actual = candidate[name]
            tensors += 1
            if not torch.equal(expected, actual):
                differences.append(
                    {
                        "file": path.name,
                        "name": name,
                        "source_shape": list(expected.shape),
                        "candidate_shape": list(actual.shape),
                        "max_absolute_difference": (expected.float() - actual.float()).abs().max().item()
                        if expected.shape == actual.shape
                        else None,
                    }
                )
    report = {"bitwise_equal": not differences, "files": len(files), "tensors": tensors, "differences": differences}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
