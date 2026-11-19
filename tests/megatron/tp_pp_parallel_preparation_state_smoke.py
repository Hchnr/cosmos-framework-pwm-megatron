"""Four CPU ranks exercise real Gloo checkpoint gather/repartition boundaries."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_framework.training_backends.megatron_tp_pp.parallel_preparation import (
    ParallelPreparation,
    partition_logical_ranks,
)
from cosmos_framework.training_backends.megatron_tp_pp.planning import batch_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("Run this boundary smoke with four CPU ranks")
    runtime = ParallelPreparation.__new__(ParallelPreparation)
    # Noncontiguous model groups catch accidental WORLD-rank partitioning.
    for members in ([0, 2], [1, 3]):
        group = dist.new_group(members, backend="gloo")
        if rank in members:
            runtime.members, runtime.cpu_group = members, group
    runtime.index = runtime.members.index(rank)
    runtime.owner, runtime.source = runtime.index == 0, runtime.members[0]
    runtime.logical_ranks = tuple(range(rank % 2, 8, 2))
    runtime.assigned = partition_logical_ranks(runtime.logical_ranks, 2, runtime.index)
    complete = {
        "loader": {r: {"cursor": 6, "worker_rng": torch.arange(17) + r} for r in runtime.logical_ranks},
        "rng": {r: {"state": torch.arange(23, dtype=torch.uint8) + r} for r in runtime.logical_ranks},
    }
    local = {field: {r: complete[field][r] for r in runtime.assigned} for field in complete}
    runtime.loader = SimpleNamespace(logical_state_dict=lambda: local["loader"])
    runtime.rng = SimpleNamespace(state_dict=lambda: local["rng"])
    merged = runtime.checkpoint_state()
    if runtime.owner:
        assert batch_fingerprint(merged) == batch_fingerprint(complete)
        restored = {"logical_loader": merged["loader"], "logical_training_rng": merged["rng"]}
    else:
        assert merged is None
        restored = {"logical_loader": None, "logical_training_rng": None}
    repartitioned = runtime.restart_partition(restored)
    assert batch_fingerprint(repartitioned) == batch_fingerprint(local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = args.output.with_name(args.output.stem + f".rank{rank}.json")
    with target.open("x") as handle:
        json.dump(
            {
                "passed": True,
                "rank": rank,
                "members": runtime.members,
                "assigned": runtime.assigned,
                "scope": "real CPU/Gloo aggregation and repartition; no production loader or GPU acceptance",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
