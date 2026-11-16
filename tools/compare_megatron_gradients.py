"""Compare full logical gradients reconstructed from unequal per-rank shards."""

import argparse
import json
import math
from pathlib import Path

import torch


def load(root):
    return [
        torch.load(path, map_location="cpu", mmap=True, weights_only=False) for path in sorted(root.glob("rank_*.pt"))
    ]


def logical(states, name):
    entries = [state["gradients"][name] for state in states]
    dim = entries[0]["shard_dim"]
    if dim is None:
        return entries[0]["local"].float()
    entries.sort(key=lambda entry: entry["chunks"][0].offsets[dim])
    value = torch.cat([entry["local"] for entry in entries], dim=dim).float()
    if list(value.shape) != entries[0]["global_shape"]:
        raise ValueError(f"Gradient coverage mismatch for {name}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, candidate = load(args.source), load(args.candidate)
    if not source or len(source) != len(candidate):
        raise ValueError("Missing or mismatched rank files")
    names = sorted(source[0]["gradients"])
    if any(sorted(state["gradients"]) != names for state in source + candidate):
        raise ValueError("Gradient names differ across runs/ranks")
    parameters = []
    a2 = b2 = dot = diff2 = 0.0
    for name in names:
        a, b = logical(source, name), logical(candidate, name)
        if a.shape != b.shape:
            raise ValueError(f"Gradient shape mismatch: {name}")
        aa, bb = a.square().sum().item(), b.square().sum().item()
        ab = (a * b).sum().item()
        delta = (a - b).square().sum().item()
        a2 += aa
        b2 += bb
        dot += ab
        diff2 += delta
        parameters.append(
            {
                "name": name,
                "source_norm": math.sqrt(aa),
                "candidate_norm": math.sqrt(bb),
                "difference_squared": delta,
                "relative_l2_error": math.sqrt(delta / aa) if aa else None,
                "cosine": ab / math.sqrt(aa * bb) if aa * bb else None,
                "max_absolute_error": (a - b).abs().max().item() if a.numel() else 0.0,
            }
        )
    report = {
        "audit_completed": True,
        "source": str(args.source),
        "candidate": str(args.candidate),
        "parameters": len(parameters),
        "source_norm": math.sqrt(a2),
        "candidate_norm": math.sqrt(b2),
        "relative_l2_error": math.sqrt(diff2 / a2),
        "cosine": dot / math.sqrt(a2 * b2),
        "largest_differences": sorted(parameters, key=lambda item: item["difference_squared"], reverse=True)[:30],
        "per_parameter": parameters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_parameter", "largest_differences")}))


if __name__ == "__main__":
    main()
