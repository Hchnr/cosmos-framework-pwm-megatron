"""Read-only P0 inventory of the actual full-state Cosmos DCP source."""

import argparse
import json
import pickle
import re
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def audit(root, logical_dp=192):
    root = Path(root)
    if not (root / ".save_complete").is_file():
        raise ValueError("Source checkpoint has no completion marker")
    metadata = {
        c: FileSystemReader(root / c).read_metadata().state_dict_metadata
        for c in ("model", "optim", "scheduler", "trainer")
    }
    model, optim, trainer = metadata["model"], metadata["optim"], metadata["trainer"]
    parameters = {}
    state_keys = ("master_param", "exp_avg", "exp_avg_sq")
    for name, item in model.items():
        if not isinstance(item, TensorStorageMetadata):
            continue
        fields = {field: optim.get(f"state.{name}.{field}") for field in state_keys}
        trained = any(v is not None for v in fields.values())
        if trained and any(not isinstance(v, TensorStorageMetadata) or v.size != item.size for v in fields.values()):
            raise ValueError(f"Incomplete master/moments for {name}")
        if trained and any(v.properties.dtype != torch.float32 for v in fields.values()):
            raise ValueError(f"Non-FP32 optimizer state for {name}")
        parameters[name] = {
            "shape": list(item.size),
            "dtype": str(item.properties.dtype),
            "optimizer_state_present": trained,
        }
    rng_ranks = sorted({int(m[1]) for k in trainer if (m := re.match(r"rng_state_(\d+)\.torch$", k))})
    loader_files = sorted((root / "dataloader").glob("rank_*.pkl"))
    loader = []
    for rank in range(logical_dp):
        path = root / "dataloader" / f"rank_{rank}.pkl"
        if not path.exists():
            loader.append({"rank": rank, "missing": True})
            continue
        with path.open("rb") as handle:
            state = pickle.load(handle)
        execution = state.get("__execution__", {})
        topology = execution.get("topology", {})
        active = topology.get("active_workers", [])
        rng = execution.get("rng", {})
        names = state.get("__schedule__", {}).get("names", [])
        required_rng = {(source, worker) for source, name in enumerate(names) for worker in state.get(name, {})}
        loader.append(
            {
                "rank": rank,
                "global_id": state.get("global_id"),
                "schedule_sources": len(state.get("__schedule__", {}).get("names", [])),
                "execution_version": execution.get("version"),
                "active_source_workers": sum(len(workers) for workers in active),
                "consumed_source_workers": len(required_rng),
                "saved_worker_rng": len(rng),
                "missing_worker_rng": sorted(required_rng - rng.keys()),
                "execution_keys": list(execution),
            }
        )
    complete_rng = rng_ranks == list(range(logical_dp))
    complete_loader = len(loader_files) == logical_dp and all(
        not row.get("missing") and row["execution_version"] is not None and not row["missing_worker_rng"]
        for row in loader
    )
    return {
        "source": str(root.resolve()),
        "source_read_only": True,
        "metadata_keys": {c: len(m) for c, m in metadata.items()},
        "parameters": parameters,
        "optimizer_parameter_count": sum(p["optimizer_state_present"] for p in parameters.values()),
        "non_tensor_optimizer_keys": [k for k, v in optim.items() if not isinstance(v, TensorStorageMetadata)],
        "scheduler_keys": list(metadata["scheduler"]),
        "trainer_non_rng_keys": [k for k in trainer if not k.startswith("rng_state_")],
        "rank_rng_coverage": complete_rng,
        "rank_rng_count": len(rng_ranks),
        "loader_state_coverage": complete_loader,
        "loader": loader,
        "strict_resume_ready": False,
        "strict_resume_note": "Metadata coverage is necessary only. Worker payload and logical-stream replay must be validated before strict-resume acceptance.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("parameters", "loader", "non_tensor_optimizer_keys")})
    )


if __name__ == "__main__":
    main()
