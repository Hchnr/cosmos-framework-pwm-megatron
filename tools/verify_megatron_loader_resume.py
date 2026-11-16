"""Compare real-catalog content, cursors and CPU-worker RNG across restart."""

import argparse
import json
from pathlib import Path


def read(prefix, rank):
    path = prefix.with_name(f"{prefix.stem}_rank_{rank}.json")
    return [{key: value for key, value in item.items() if key != "iteration"} for item in json.loads(path.read_text())]


def main():
    parser = argparse.ArgumentParser()
    for name in ("first", "resumed", "loader-resumed", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    differences = []
    revisited = set()
    counts = []
    for rank in range(args.ranks):
        first, resumed, loader_resumed, reference = [
            read(prefix, rank) for prefix in (args.first, args.resumed, args.loader_resumed, args.reference)
        ]
        if not first or not resumed or not loader_resumed:
            raise ValueError(f"Empty observations on rank {rank}")
        offset = len(first)
        for name, actual, expected in (
            ("initial", first, reference[:offset]),
            ("full_model_restart", resumed, reference[offset : offset + len(resumed)]),
            ("dataflow_restart", loader_resumed, reference[offset : offset + len(loader_resumed)]),
        ):
            if actual != expected:
                differences.append({"rank": rank, "phase": name, "actual": actual, "expected": expected})
        revisited.update({item["dataset"] for item in first} & {item["dataset"] for item in loader_resumed})
        counts.append(
            {
                "rank": rank,
                "initial": len(first),
                "full_model_restart": len(resumed),
                "dataflow_restart": len(loader_resumed),
                "reference": len(reference),
            }
        )
    report = {
        "passed": not differences and bool(revisited),
        "ranks": args.ranks,
        "counts": counts,
        "revisited_streams": sorted(revisited),
        "compared_fields": [
            "dataset",
            "worker",
            "epoch",
            "index",
            "global_id",
            "source",
            "resume_rng_sha256",
            "content_sha256",
        ],
        "differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "differences"}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
