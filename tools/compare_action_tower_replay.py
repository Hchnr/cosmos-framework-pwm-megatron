#!/usr/bin/env python3
"""Compare fixed-input replay or fixed-preparation sampler artifacts."""

import argparse
import dataclasses
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    a, b = [torch.load(path, map_location="cpu", weights_only=False) for path in (args.baseline, args.candidate)]
    assert a["input_sha256"] == b["input_sha256"], "Different explicit inputs"

    def exact_input(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        elif dataclasses.is_dataclass(x):
            for field in dataclasses.fields(x):
                exact_input(getattr(x, field.name), getattr(y, field.name))
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for xi, yi in zip(x, y, strict=True):
                exact_input(xi, yi)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x:
                exact_input(x[key], y[key])
        else:
            assert x == y

    if "sampling_inputs" in a:
        # Each independent sample starts from identical noise/condition/IDs;
        # later inputs intentionally follow each model's own sampler trajectory.
        fixture = torch.load(a["args"]["inputs"], map_location="cpu", weights_only=False)
        cursor = 0
        for capture in fixture["captures"]:
            exact_input(a["sampling_inputs"][cursor], b["sampling_inputs"][cursor])
            cursor += len(capture["packed_steps"])
    for key in ("precision", "checkpoint", "config_base", "sample"):
        assert a["args"].get(key) == b["args"].get(key), key
    rows = []

    def check(x, y, path):
        if isinstance(x, torch.Tensor):
            assert x.shape == y.shape, path
            assert x.dtype == y.dtype, path
            d = (x.float() - y.float()).abs()
            close = torch.isclose(x, y, atol=args.atol, rtol=args.rtol)
            bitwise = torch.equal(
                x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8)
            )
            rows.append(
                {
                    "path": path,
                    "max_abs": d.max().item() if d.numel() else 0,
                    "mean_abs": d.mean().item() if d.numel() else 0,
                    "passed": bitwise if args.atol == args.rtol == 0 else bool(close.all()),
                    "bitwise_equal": bitwise,
                    "mismatched": int((~close).sum()),
                    "elements": x.numel(),
                }
            )
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for key in x:
                check(x[key], y[key], f"{path}/{key}")
        elif isinstance(x, list):
            assert len(x) == len(y), path
            for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
                check(xi, yi, f"{path}/{i}")

    check(a["outputs"], b["outputs"], "outputs")
    if "attention" in a and "attention" in b:
        check(a["attention"], b["attention"], "first_attention")
    if "sample_results" in a:
        check(a["sample_results"], b["sample_results"], "sample_results")
    if "traces" in a:
        check(a["traces"], b["traces"], "layers")
        for x, y in zip(a["routing"], b["routing"], strict=True):
            assert x["state_tokens"] == y["state_tokens"] == 0
            for key in (
                "action_sequence_indexes",
                "video_sequence_indexes",
                "und_sequence_indexes",
                "action_timesteps",
            ):
                torch.testing.assert_close(x[key], y[key], atol=0, rtol=0)
        route = a["routing"][0]
        gen = torch.cat([route["action_sequence_indexes"], route["video_sequence_indexes"]]).sort().values
        ai = torch.where(torch.isin(gen, route["action_sequence_indexes"]))[0]
        vi = torch.where(torch.isin(gen, route["video_sequence_indexes"]))[0]
        for key, value in b["structures"].items():
            source = key if key in a["structures"] else key.replace("_moe_gen_action", "_moe_gen")
            original = a["structures"][source]
            indexes = ai if "_moe_gen_action" in key else vi if "_moe_gen" in key else None
            if indexes is not None:
                original = {k: v[indexes] if v.shape[0] != value[k].shape[0] else v for k, v in original.items()}
            check(original, value, f"first_layer_structure/{key}")
    groups = {}
    for group in ("outputs", "sample_results", "layers", "first_layer_structure", "first_attention"):
        subset = [row for row in rows if row["path"].startswith(group + "/")]
        if subset:
            groups[group] = {
                "passed": all(row["passed"] for row in subset),
                "max_abs": max(row["max_abs"] for row in subset),
                "failed_tensors": sum(not row["passed"] for row in subset),
                "checked_tensors": len(subset),
            }
    action_predictions = [row for row in rows if "/preds_action/" in row["path"]]
    final_actions = [row for row in rows if row["path"].startswith("sample_results/") and "/action/" in row["path"]]
    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_diagnostic_only": a.get("diagnostic_only", False),
        "candidate_diagnostic_only": b.get("diagnostic_only", False),
        "input_sha256": a["input_sha256"],
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": all(row["passed"] for row in rows),
        "summary": groups,
        "metrics": rows,
        "action_predictions_passed": all(row["passed"] for row in action_predictions) if action_predictions else None,
        "final_action_passed": all(row["passed"] for row in final_actions) if final_actions else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
