"""Apply explicit numerical gates to independently recorded input/gradient audits."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--relative-l2-limit", type=float, default=0.02)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    gradients = json.loads(args.gradients.read_text())
    passed = (
        inputs["bitwise_equal"]
        and inputs["tensors"] > 0
        and gradients["relative_l2_error"] < args.relative_l2_limit
        and gradients["cosine"] > args.minimum_cosine
    )
    result = {
        "passed": passed,
        "inputs": str(args.inputs),
        "gradients": str(args.gradients),
        "input_tensor_count": inputs["tensors"],
        "inputs_bitwise_equal": inputs["bitwise_equal"],
        "gradient_relative_l2_error": gradients["relative_l2_error"],
        "relative_l2_limit": args.relative_l2_limit,
        "gradient_cosine": gradients["cosine"],
        "minimum_cosine": args.minimum_cosine,
        "scope": gradients.get("scope", "See source audit"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
