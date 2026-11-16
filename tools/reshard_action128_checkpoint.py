# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare a separate checkpoint with reassigned episode shards (trusted input only).

Without --output, validate and report only. Model/optimizer payloads are never
deserialized. With --output, hard-link immutable components on the same FS so
retention of the source checkpoint cannot break the prepared checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import uuid
from pathlib import Path

from cosmos_framework.data.generator.dataflow.reshard import redistribute_episode_states


def prepare(checkpoint: Path, target_dp_size: int, output: Path | None = None):
    checkpoint = checkpoint.resolve()
    if not (checkpoint / ".save_complete").is_file():
        raise ValueError("Source checkpoint has no save-complete marker")
    for component in ["model", "optim", "scheduler", "trainer"]:
        if not (checkpoint / component / ".metadata").is_file():
            raise ValueError(f"Missing completed checkpoint component: {component}")
    files = sorted((checkpoint / "dataloader").glob("rank_*.pkl"), key=lambda p: int(p.stem.removeprefix("rank_")))
    if [p.name for p in files] != [f"rank_{r}.pkl" for r in range(len(files))]:
        raise ValueError("Source rank files must be contiguous and complete")
    states, hashes = [], {}
    for path in files:
        data = path.read_bytes()
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        states.append(pickle.loads(data))  # Only trusted framework checkpoints.
    converted = redistribute_episode_states(states, target_dp_size)
    report = {
        "source_checkpoint": str(checkpoint),
        "old_dp_size": len(states),
        "target_dp_size": target_dp_size,
        "global_id": converted[0]["global_id"],
        "source_count": len(converted[0]["__fingerprints__"]),
        "original_shards_per_source": [
            sum(len(s["__execution__"]["virtual_shards"][i]) for s in converted)
            for i in range(len(converted[0]["__fingerprints__"]))
        ],
        "source_dataloader_sha256": hashes,
        "output": str(output.resolve()) if output else None,
        "scope": "Committed data cursors and logical-shard ownership only; no distributed model/optimizer restore tested. "
        "No replay within each original epoch; future epochs retain normal dataset repetition. "
        "Work consumed after this checkpoint is not represented.",
    }
    if output is None:
        return report
    output = output.resolve()
    if output.exists() or output == checkpoint or checkpoint in output.parents:
        raise ValueError("Output must be a new directory outside the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.preparing-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        for component in ["model", "optim", "scheduler", "trainer"]:
            shutil.copytree(checkpoint / component, temporary / component, copy_function=os.link)
        (temporary / "dataloader").mkdir()
        for rank, state in enumerate(converted):
            (temporary / "dataloader" / f"rank_{rank}.pkl").write_bytes(pickle.dumps(state, protocol=5))
        (temporary / "reshard_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        (temporary / ".save_complete").write_text("virtual-shard migration complete\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-dp-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare(args.checkpoint, args.target_dp_size, args.output)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_dataloader_sha256"}))


if __name__ == "__main__":
    main()
