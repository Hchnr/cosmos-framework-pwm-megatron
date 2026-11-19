"""Check restored compute parameters against rounded FP32 optimizer masters."""

import argparse
import json
from pathlib import Path

import torch


def audit(directory, *, verify_source_model=False):
    if not (directory / ".numerical_complete").is_file():
        raise ValueError("Numerical export must be complete before auditing")
    manifest = json.loads((directory / "manifest.json").read_text())
    rank_files = sorted(directory.glob("manifest.rank*.json"))
    if len(rank_files) != manifest["world_size"]:
        raise ValueError("Missing rank manifests")
    rows = [row for path in rank_files for row in json.loads(path.read_text())]
    if len(rows) != manifest["optimizer_fragments"]:
        raise ValueError("Incomplete optimizer fragment coverage")
    source = None
    if verify_source_model:
        from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader

        checkpoint = Path(manifest["source_runtime_checkpoint"])
        if not (checkpoint / ".save_complete").is_file():
            raise ValueError("Source runtime checkpoint must be complete")
        source = CanonicalDCPReader(checkpoint)
    reports = []
    for index, row in enumerate(rows):
        path = (directory / row["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Fragment escapes observation directory")
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model = saved["state"]["model"].reshape(-1)
        master = saved["state"]["master_param"].reshape(-1)
        if master.dtype != torch.float32 or model.shape != master.shape:
            raise ValueError("Invalid model/master shape or master dtype")
        mismatches, nonfinite, maximum = 0, 0, 0.0
        for start in range(0, model.numel(), 1048576):
            actual, expected = model[start : start + 1048576], master[start : start + 1048576].to(model.dtype)
            mismatches += int(torch.count_nonzero(actual != expected))
            nonfinite += int(torch.count_nonzero(~torch.isfinite(actual) | ~torch.isfinite(expected)))
            maximum = max(maximum, float((actual.float() - expected.float()).abs().max()))
        reports.append(
            {
                "name": row["layout"]["canonical_name"],
                "tensor_rank": row["layout"]["tensor_rank"],
                "start": row["start"],
                "end": row["end"],
                "dtype": str(model.dtype),
                "mismatched_elements": mismatches,
                "nonfinite_elements": nonfinite,
                "maximum_absolute_difference": maximum,
                "passed": mismatches == 0 and nonfinite == 0,
            }
        )
        if source is not None:
            layout = row["layout"]
            restored_from = source.read(
                layout["canonical_name"],
                axis=layout["axis"],
                rank=layout["tensor_rank"],
                size=manifest["topology"]["tensor_model_parallel_size"],
            ).reshape(-1)[row["start"] : row["end"]]
            reports[-1]["source_model_exact"] = model.dtype == restored_from.dtype and torch.equal(model, restored_from)
            del restored_from
        del saved, model, master
        if index % 100 == 0:
            print(json.dumps({"audited_fragments": index + 1}), flush=True)
    return {
        "passed": all(row["passed"] for row in reports),
        "scope": "restored compute parameters equal FP32 optimizer masters rounded to compute dtype",
        "iteration": manifest["iteration"],
        "source_model_verified": verify_source_model,
        "source_model_difference_fragments": (
            sum(not row["source_model_exact"] for row in reports) if verify_source_model else None
        ),
        "fragments": len(reports),
        "failed_fragments": sum(not row["passed"] for row in reports),
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-source-model", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit(args.observation, verify_source_model=args.verify_source_model)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "reports"}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
