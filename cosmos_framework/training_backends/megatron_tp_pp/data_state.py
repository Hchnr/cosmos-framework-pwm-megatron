# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Import native loader/training RNG identities, including baseline extension."""

import json
import pickle

from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback

from .rng import LogicalTrainingRNG
from .source import CanonicalDCPReader


def import_logical_state(template, bundle, topology, logical_ranks, saved_contract, contract_changed):
    reader = CanonicalDCPReader(bundle.path / "logical_state" / "trainer")
    precision = json.loads(reader.read("training_precision"))
    if precision.get("enabled"):
        raise ValueError("The TP/PP recipe does not migrate FP8 training state")
    rng = LogicalTrainingRNG.from_source(reader, logical_ranks)
    states = {}
    callback = JointCosmosDataLoaderStateCallback(template)
    for rank in logical_ranks:
        with (bundle.path / "logical_state" / "dataloader" / f"rank_{rank}.pkl").open("rb") as handle:
            saved = pickle.load(handle)
        added = callback.validate_action128_extension(saved, saved_contract["spec"])
        if not contract_changed and not added:
            states[rank] = saved
            continue
        # This is exactly the original append-only extension boundary. Only
        # that boundary resets data streams; a TP/PP topology change never does.
        active = [
            stream.dataset._distributor.active_worker_ids(
                rank, topology.logical_data_parallel_size, max(1, stream.num_workers)
            )
            for stream in template._loaders
        ]
        if any(not workers for workers in active):
            raise ValueError(f"Logical rank {rank} has an empty source shard")
        states[rank] = {
            "global_id": 0,
            "__schedule__": template.schedule_identity,
            "__fingerprints__": template.dataset_fingerprints,
            "__execution__": {
                "version": 1,
                "topology": {
                    "logical_workers": [stream.num_workers for stream in template._loaders],
                    "active_workers": active,
                    "dp": [[rank, topology.logical_data_parallel_size] for _ in template._loaders],
                },
                "next_workers": [0] * len(template._loaders),
                "base_seeds": rng.draw_loader_base_seeds(rank, len(template._loaders)),
                "rng": {},
            },
        }
    return (
        states,
        rng,
        {
            "source_trainer_iteration": reader.read("iteration"),
            "source_training_precision": precision,
            "action128_data_extension_reset": bool(contract_changed or added),
        },
    )
