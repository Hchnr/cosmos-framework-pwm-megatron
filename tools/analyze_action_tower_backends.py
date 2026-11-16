#!/usr/bin/env python3
"""Locate first-layer divergence in fixed-input Nano backend/projection ablations."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="Contains two.pt and the four backend/projection captures")
    args = parser.parse_args()
    root = Path(args.directory)
    baseline = torch.load(root / "two.pt", map_location="cpu", weights_only=False)
    route = baseline["routing"][0]
    gen = torch.cat([route["action_sequence_indexes"], route["video_sequence_indexes"]]).sort().values
    ai = torch.where(torch.isin(gen, route["action_sequence_indexes"]))[0]
    vi = torch.where(torch.isin(gen, route["video_sequence_indexes"]))[0]
    nu = len(route["und_sequence_indexes"])
    rows = []

    def check(name, stage, branch, x, y):
        assert x.shape == y.shape and x.dtype == y.dtype
        delta = (x.float() - y.float()).abs()
        rows.append(
            {
                "variant": name,
                "stage": stage,
                "branch": branch,
                "shape": list(x.shape),
                "bitwise_equal": torch.equal(
                    x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8)
                ),
                "max_abs": delta.max().item(),
                "different_elements": int((x != y).sum()),
            }
        )

    kernels = {"two": baseline["native_backend_calls"]}
    for name in ("nano_full", "nano_routed", "flex_full", "flex_routed"):
        candidate = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=False)
        assert candidate["input_sha256"] == baseline["input_sha256"]
        kernels[name] = candidate["native_backend_calls"]
        for stage in ("input_layernorm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm"):
            for branch, indexes in (("UND", None), ("GEN_VIDEO", vi), ("GEN_ACTION", ai)):
                stem = stage if stage == "input_layernorm" else "self_attn." + stage
                source = stem + ("_moe_gen" if indexes is not None else "")
                target = stem + (
                    "_moe_gen_action" if branch == "GEN_ACTION" else "_moe_gen" if indexes is not None else ""
                )
                for io in ("in", "out"):
                    x, y = baseline["structures"][source][io], candidate["structures"][target][io]
                    if indexes is not None:
                        if len(y) == len(x):
                            y = y[indexes]
                        x = x[indexes]
                    check(name, f"{stage}.{io}", branch, x, y)
        for stage in ("q", "k", "v", "out"):
            x, y = baseline["attention"][stage], candidate["attention"][stage]
            for branch, indexes in (("UND", torch.arange(nu)), ("GEN_VIDEO", nu + vi), ("GEN_ACTION", nu + ai)):
                check(name, f"attention.{stage}", branch, x[indexes], y[indexes])
    report = {
        "input_sha256": baseline["input_sha256"],
        "native_kernel_calls_first_layer": kernels,
        "first_difference": {
            name: next((row for row in rows if row["variant"] == name and not row["bitwise_equal"]), None)
            for name in kernels
            if name != "two"
        },
        "metrics": rows,
    }
    (root / "first_divergence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
