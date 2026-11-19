"""Compare canonical TP/PP observations with a native FSDP numerical snapshot."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.training_backends.megatron_tp_pp.numerical_state import validate_fragment_coverage
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader, release_file_cache
from tools.compare_tp_pp_performance import compare as compare_inputs
from tools.compare_tp_pp_runtime_checkpoints import tensor_difference


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def compare(observation, native):
    if not (observation / ".numerical_complete").is_file() or not (native / ".math_complete").is_file():
        raise ValueError("Both numerical observations must be complete")
    manifest = json.loads((observation / "manifest.json").read_text())
    reference = json.loads((native / "manifest.json").read_text())
    if manifest["format"] != "cosmos_tp_pp_numerical_observation" or manifest["iteration"] != reference["iteration"]:
        raise ValueError("Mismatched numerical formats or update windows")
    model, optimizer = CanonicalDCPReader(native / "model"), CanonicalDCPReader(native / "optim")
    active = {
        name.removeprefix("state.").removesuffix(".master_param"): {
            "optimizer_fields": [name],
            "global_shape": list(metadata.size),
        }
        for name, metadata in optimizer.metadata.items()
        if name.startswith("state.") and name.endswith(".master_param")
    }
    rank_files = sorted(observation.glob("manifest.rank*.json"))
    if len(rank_files) != manifest["world_size"]:
        raise ValueError("Numerical observation lost rank manifests")
    records = [row for path in rank_files for row in json.loads(path.read_text())]
    if len(records) != manifest["optimizer_fragments"] or len(active) != manifest["canonical_optimizer_parameters"]:
        raise ValueError("Numerical observation manifest counts differ from actual coverage")
    validate_fragment_coverage(records, active, manifest["topology"]["tensor_model_parallel_size"])
    tensor_reports, object_reports = [], []
    for index, row in enumerate(records):
        path = (observation / row["path"]).resolve()
        if not path.is_relative_to(observation.resolve()):
            raise ValueError("Numerical fragment escapes its observation directory")
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if any(
            plain(saved[field]) != plain(row[field])
            for field in ("layout", "start", "end", "step", "hyperparameters", "path")
        ):
            raise ValueError("Numerical payload/manifest identity mismatch")
        if set(saved["state"]) != {"model", "master_param", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Numerical observation is missing model/master/moments")
        layout, name = row["layout"], row["layout"]["canonical_name"]
        for field, actual in saved["state"].items():
            if field not in ("model", "master_param", "exp_avg", "exp_avg_sq"):
                raise ValueError("Unexpected numerical state field")
            reader, key = (model, name) if field == "model" else (optimizer, f"state.{name}.{field}")
            expected = (
                reader.read(
                    key,
                    axis=layout["axis"],
                    rank=layout["tensor_rank"],
                    size=manifest["topology"]["tensor_model_parallel_size"],
                )
                .reshape(-1)[row["start"] : row["end"]]
                .view_as(actual)
            )
            difference = tensor_difference(
                expected,
                actual,
                relative_tolerance=1e-4,
                absolute_tolerance=2e-6,
                exact=False,
                local_windows=False,
            )
            tensor_reports.append(
                {
                    "name": name,
                    "field": field,
                    "tensor_rank": layout["tensor_rank"],
                    "start": row["start"],
                    "end": row["end"],
                    **difference,
                }
            )
            del expected
        native_step = plain(optimizer.read(f"param_groups.{name}.step"))
        if isinstance(native_step, list) and len(native_step) == 1:
            native_step = native_step[0]
        differences = ["step"] if int(native_step) != row["step"] else []
        for field, actual in row["hyperparameters"].items():
            expected = plain(optimizer.read(f"param_groups.{name}.{field}"))
            if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, list):
                expected = expected[0]
            if expected != actual:
                differences.append(field)
        object_reports.append(
            {"name": name, "tensor_rank": layout["tensor_rank"], "passed": not differences, "differences": differences}
        )
        del saved
        release_file_cache([path])
        if index % 100 == 0:
            print(json.dumps({"native_comparison_fragments": index + 1}), flush=True)
    frozen_reports = []
    source = manifest.get("source_runtime_checkpoint")
    if source is None or not (Path(source) / ".save_complete").is_file():
        raise ValueError("Full model comparison requires the completed source runtime checkpoint")
    runtime = CanonicalDCPReader(Path(source))
    source_manifest = json.loads((Path(source) / "manifest.json").read_text())
    if source_manifest["iteration"] != manifest["iteration"]:
        raise ValueError("Runtime checkpoint and numerical observation iterations differ")
    lineage = compare_inputs(
        Path(source).parents[1],
        native.parent,
        warmup_updates=0,
        logical_dp=manifest["topology"]["logical_data_parallel_size"],
        reference_microsteps=manifest["topology"]["reference_micro_steps_per_update"],
    )
    if {name for name in runtime.metadata if name.startswith("net.")} != set(model.metadata):
        raise ValueError("Native and TP/PP model key coverage differs")
    for name in model.metadata:
        if name in active:
            continue
        a, b = model.read(name), runtime.read(name)
        difference = tensor_difference(a, b, relative_tolerance=1e-4, absolute_tolerance=2e-6, exact=True)
        frozen_reports.append({"name": name, **difference})
        del a, b
    release_file_cache(
        path for root in (native / "model", native / "optim", Path(source)) for path in root.glob("*.distcp")
    )
    return {
        "passed": all(row["passed"] for row in tensor_reports + object_reports + frozen_reports)
        and lineage["inputs_matched"]
        and lineage["bf16_loss_trajectory_passed"],
        "scope": "actual production inputs: complete active model/master/moments/step/hyperparameters and frozen model",
        "iteration": manifest["iteration"],
        "canonical_optimizer_parameters": len(active),
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 2e-6,
        "tensor_reports": tensor_reports,
        "optimizer_metadata_reports": object_reports,
        "frozen_model_reports": frozen_reports,
        "input_lineage_matched": lineage["inputs_matched"],
        "bf16_loss_trajectory_passed": lineage["bf16_loss_trajectory_passed"],
        "gradient_comparison_performed": False,
        "scheduler_and_data_comparison_performed": False,
        "production_576_slot_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.observation, args.native)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("reports")}))
    if not result["passed"]:
        raise SystemExit("Native full-state numerical gate failed; complete per-parameter observations retained")


if __name__ == "__main__":
    main()
