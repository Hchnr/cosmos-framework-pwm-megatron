"""Explicitly migrate a trusted checkpoint to audited media-only publications.

This writes a separate checkpoint and new strict fingerprints. It never disables
runtime fingerprint checks, advances a cursor, resets a stream or changes model
and optimizer payloads. Run from the repository root with python -m tools....
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

import tomllib
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.checkpoint.action128 import validate_checkpoint_digest
from cosmos_framework.data.generator.action.media_migration import validate_media_migration
from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset, build_action128_spec
from cosmos_framework.data.generator.catalog import ParquetCatalogDataset


def prepare(checkpoint, recipe, pairs, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if not (checkpoint / ".save_complete").is_file() or output.exists() or checkpoint in output.parents:
        raise ValueError("Need a complete source checkpoint and a new independent output directory")
    cfg = tomllib.loads(Path(recipe).read_text())
    replacements = {str(Path(old).resolve()): str(Path(new).resolve()) for old, new in pairs}
    if len(replacements) != len(pairs) or not replacements:
        raise ValueError("Need explicit distinct old/new catalog pairs")
    paths = [s["path"] for s in cfg["dataloader_train"]["datasets"]]
    if not set(replacements).issubset(paths):
        raise ValueError("Every old catalog must occur in the unchanged source recipe")
    old_rows = [ParquetCatalogDataset(p)[0] for p in paths]
    new_rows = [ParquetCatalogDataset(replacements.get(p, p))[0] for p in paths]
    state = {"action128_contract": ""}
    dcp.load(state, storage_reader=dcp.FileSystemReader(checkpoint / "trainer"), no_dist=True)
    contract = json.loads(state["action128_contract"])
    reader = dcp.FileSystemReader(checkpoint / "model")
    keys = [k for k in reader.read_metadata().state_dict_metadata if k.endswith("action_normalization.contract_digest")]
    if not keys:
        raise ValueError("Missing checkpoint model contract digest")
    digests = {k: torch.empty(16, dtype=torch.uint8) for k in keys}
    dcp.load(digests, storage_reader=reader, no_dist=True)
    validate_checkpoint_digest(contract, digests)
    for rows in [old_rows, new_rows]:
        actual = build_action128_spec([r["data"]["root"] for r in rows])
        if actual != contract["spec"]:
            raise ValueError("Media publication changed checkpoint Action128 normalization/routing contract")
    with (checkpoint / "dataloader/rank_0.pkl").open("rb") as f:
        reference = pickle.load(f)
    if list(reference["__fingerprints__"]) != [f"{r['dataset']}:{i}" for i, r in enumerate(old_rows)]:
        raise ValueError("Recipe source order differs from checkpoint")
    changes = {}
    for i, (old, new) in enumerate(zip(old_rows, new_rows, strict=True)):
        if paths[i] not in replacements:
            continue
        if any(old[k] != new[k] for k in ["id", "type", "dataset"]):
            raise ValueError("Media migration changed catalog identity")
        old_data, new_data = dict(old["data"]), dict(new["data"])
        if {k: v for k, v in old_data.items() if k != "root"} != {k: v for k, v in new_data.items() if k != "root"}:
            raise ValueError("Media migration changed catalog processing parameters")
        datasets = []
        for data in [old_data, new_data]:
            if data.pop("adapter") != "unified_action128":
                raise ValueError("Only unified_action128 media migration is supported")
            data.pop("format", None)
            data["use_state"] = cfg["dataloader_train"]["action_use_state"]
            data["video_fps_max"] = cfg["dataloader_train"]["action_video_fps_default"]
            datasets.append(UnifiedAction128Dataset(**data, action128_spec=contract["spec"], seed=42))
        report = validate_media_migration(*datasets)
        name = f"{old['dataset']}:{i}"

        def fingerprint(dataset):
            value = dataset.fingerprint
            if cfg["dataloader_train"].get("first_frame_color_jitter", False):
                value = hashlib.sha256(f"{value}:first-frame-color-jitter-v1:42".encode()).hexdigest()
            return value

        old_fp, new_fp = map(fingerprint, datasets)
        if reference["__fingerprints__"][name] != old_fp:
            raise ValueError(f"Source publication no longer matches saved checkpoint fingerprint: {name}")
        changes[name] = {
            **report,
            "old_catalog": paths[i],
            "new_catalog": replacements[paths[i]],
            "old_fingerprint": old_fp,
            "new_fingerprint": new_fp,
        }
        print(json.dumps({"validated_source": name, **report}), flush=True)
    report = {
        "source_checkpoint": str(checkpoint),
        "output": str(output),
        "changes": changes,
        "global_id": reference["global_id"],
        "cursor_rng_execution_state_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.preparing-{uuid.uuid4().hex}")
    tmp.mkdir()
    try:
        for component in ["model", "optim", "scheduler", "trainer"]:
            shutil.copytree(checkpoint / component, tmp / component, copy_function=os.link)
        (tmp / "dataloader").mkdir()
        files = list((checkpoint / "dataloader").glob("rank_*.pkl"))
        for path in files:
            original = path.read_bytes()
            state = pickle.loads(original)
            if (
                state["__fingerprints__"] != reference["__fingerprints__"]
                or state["global_id"] != reference["global_id"]
            ):
                raise ValueError("Checkpoint rank states disagree")
            for name, change in changes.items():
                state["__fingerprints__"][name] = change["new_fingerprint"]
            (tmp / "dataloader" / path.name).write_bytes(pickle.dumps(state, protocol=5))
        # Retain the virtual-shard migration provenance in addition to media audit.
        for name in ["reshard_manifest.json"]:
            if (checkpoint / name).exists():
                shutil.copy2(checkpoint / name, tmp / name)
        report["rank_count"] = len(files)
        (tmp / "media_migration_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        (tmp / ".save_complete").write_text("Audited explicit media migration complete\n")
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-recipe", type=Path, required=True)
    parser.add_argument("--catalog-pair", nargs=2, action="append", required=True, metavar=("OLD", "NEW"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.checkpoint, args.source_recipe, args.catalog_pair, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}))


if __name__ == "__main__":
    main()
