"""Bounded canonical observations of actual loaded model/master/Adam fragments.

This is a diagnostic export, never a replacement for the full resumable
checkpoint. It reads the live optimizer after its ordinary checkpoint restore.
"""

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .optimizer import adam_parameter_states
from .parameter_map import stage_parameter_manifest


def _write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_fragment_coverage(records, parameters, tensor_parallel_size):
    grouped = {}
    for record in records:
        grouped.setdefault(record["layout"]["canonical_name"], []).append(record)
    expected = {name for name, entry in parameters.items() if entry["optimizer_fields"]}
    if set(grouped) != expected:
        raise ValueError("Numerical export does not cover all canonical optimizer parameters")
    for name, pieces in grouped.items():
        for piece in pieces:
            layout = piece["layout"]
            if list(layout["shape"]) != list(parameters[name]["global_shape"]):
                raise ValueError(f"Canonical optimizer shape mismatch: {name}")
            expected_shape = list(layout["shape"])
            if layout["axis"] is not None:
                expected_shape[layout["axis"]] //= tensor_parallel_size
            if list(layout["local_shape"]) != expected_shape:
                raise ValueError(f"Local TP optimizer shape mismatch: {name}")
        for tensor_rank in range(tensor_parallel_size):
            local = [piece for piece in pieces if piece["layout"]["tensor_rank"] == tensor_rank]
            if not local:
                raise ValueError(f"Missing tensor-parallel replica/shard: {name}")
            cursor = 0
            for piece in sorted(local, key=lambda piece: piece["start"]):
                if piece["start"] != cursor or piece["end"] <= cursor:
                    raise ValueError(f"Overlapping or missing DP optimizer fragment: {name}")
                cursor = piece["end"]
            if cursor != math.prod(local[0]["layout"]["local_shape"]):
                raise ValueError(f"Incomplete local TP optimizer parameter: {name}")


@torch.no_grad()
def export_numerical_state(driver, directory, canonical_parameters, *, source_checkpoint=None):
    root = Path(directory)
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    target = root / f"rank{rank:04d}"
    target.mkdir()
    layouts = stage_parameter_manifest(driver.stage)
    records = []
    for index, entry in enumerate(adam_parameter_states(driver.stage, driver.optimizer)):
        layout = layouts[entry.name]
        values = {
            "model": entry.select(entry.model_parameter.detach()).cpu(),
            "master_param": entry.master.detach().cpu(),
            "exp_avg": entry.state["exp_avg"].detach().cpu(),
            "exp_avg_sq": entry.state["exp_avg_sq"].detach().cpu(),
        }
        for field, value in values.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite numerical export: {layout.canonical_name}/{field}")
            if field != "model" and value.dtype != torch.float32:
                raise ValueError("Numerical export requires true FP32 master/moments")
        relative = Path(target.name) / f"parameter{index:04d}.pt"
        record = {
            "layout": layout.to_dict(),
            "start": entry.start,
            "end": entry.end,
            "step": entry.step,
            "path": str(relative),
            "hyperparameters": {
                key: (list(entry.group[key]) if key == "betas" else float(entry.group[key]))
                for key in ("lr", "initial_lr", "weight_decay", "betas", "eps")
            },
        }
        record["hyperparameters"]["bias_correction"] = bool(entry.group.get("bias_correction", True))
        with (root / relative).open("xb") as handle:
            torch.save({**record, "state": values}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        records.append(record)
        del values
        if index % 100 == 0:
            print(json.dumps({"numerical_export_rank": rank, "parameters_written": index + 1}), flush=True)
    _write_json(root / f"manifest.rank{rank}.json", records)
    group = dist.new_group(backend="gloo")
    try:
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(records, gathered, dst=0, group=group)
        if rank == 0:
            complete = [row for part in gathered for row in part]
            validate_fragment_coverage(complete, canonical_parameters, driver.topology.tensor_model_parallel_size)
            _write_json(
                root / "manifest.json",
                {
                    "format": "cosmos_tp_pp_numerical_observation",
                    "version": 1,
                    "iteration": driver.iteration,
                    "topology": asdict(driver.topology),
                    "world_size": world,
                    "canonical_optimizer_parameters": len({row["layout"]["canonical_name"] for row in complete}),
                    "optimizer_fragments": len(complete),
                    "resumable_training_checkpoint": False,
                    "source_runtime_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
                },
            )
            with (root / ".numerical_complete").open("x") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group(group)
