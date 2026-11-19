"""Verify streamed per-parameter CUDA norms reproduce the native norm bits."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    names = json.loads((args.reference / "manifest.json").read_text())["names"]
    records = torch.load(args.reference / "outputs.pt", map_location="cpu", weights_only=False)
    rows = []
    for update, record in enumerate(records):
        grads = []
        for index in range(len(names)):
            state = torch.load(
                args.reference / f"update{update}.parameter{index:04d}.pt",
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            if state["gradient"] is None:
                raise ValueError("This qualification requires the complete used-parameter fixture")
            grads.append(state["gradient"].cuda())
        together = torch._foreach_norm(grads, 2)
        streamed = [torch._foreach_norm([gradient], 2)[0] for gradient in grads]
        mismatches = [
            names[index] for index, (a, b) in enumerate(zip(together, streamed, strict=True)) if not torch.equal(a, b)
        ]
        expected = torch.as_tensor(record["norm"], dtype=torch.float32, device="cuda")
        native = torch.linalg.vector_norm(torch.stack(together), 2)
        candidate = torch.linalg.vector_norm(torch.stack(streamed), 2)
        passed = not mismatches and torch.equal(expected, native) and torch.equal(expected, candidate)
        rows.append(
            dict(
                update=update,
                passed=passed,
                expected=expected.item(),
                all_together=native.item(),
                streamed=candidate.item(),
                parameter_norm_mismatches=mismatches,
            )
        )
        del grads, together, streamed
    result = dict(passed=all(row["passed"] for row in rows), scope=__doc__, rows=rows, production_acceptance=False)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    if not result["passed"]:
        raise SystemExit("Streamed native norm qualification failed")


if __name__ == "__main__":
    main()
