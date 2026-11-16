"""Explicitly migrate only Bench2Dex normalization, preserving trained state.

Trusted framework checkpoints only. Immutable learned model/optimizer payloads
are hard-linked; small DCP overlays replace normalization buffers and the saved
contract. All old source fingerprints must be reproduced before new ones are
published. Source checkpoints and datasets are never modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import tomllib
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.checkpoint.action128 import validate_checkpoint_digest
from cosmos_framework.data.generator.action.pretrain_state import load_state_transforms
from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset, build_action128_spec
from cosmos_framework.data.generator.catalog import ParquetCatalogDataset
from cosmos_framework.model.generator.mot.action128 import Action128Normalization


def overlay_component(source, destination, replacements):
    """Write separate metadata and storage for a strict subset of DCP keys."""
    source, destination = Path(source), Path(destination)
    metadata = copy.deepcopy(dcp.FileSystemReader(source).read_metadata())
    if not replacements or not set(replacements) <= set(metadata.state_dict_metadata):
        raise ValueError("Overlay must replace existing checkpoint keys only")
    shutil.copytree(source, destination, copy_function=os.link)
    with tempfile.TemporaryDirectory(prefix=".normalization-overlay-", dir=destination.parent) as directory:
        patch = Path(directory)
        dcp.save(replacements, storage_writer=dcp.FileSystemWriter(patch), no_dist=True)
        patch_metadata = dcp.FileSystemReader(patch).read_metadata()
        for value in patch_metadata.storage_data.values():
            filename = value.relative_path
            target = destination / ("benchphys_" + filename)
            if not target.exists():
                os.link(patch / filename, target)
            value.relative_path = target.name
        metadata.state_dict_metadata.update(patch_metadata.state_dict_metadata)
        metadata.planner_data.update(patch_metadata.planner_data)
        metadata.storage_data = {
            index: value for index, value in metadata.storage_data.items() if index.fqn not in replacements
        }
        metadata.storage_data.update(patch_metadata.storage_data)
        # The source metadata is immutable and may already have other hardlinks.
        (destination / ".metadata").unlink()
        (destination / ".metadata").write_bytes(pickle.dumps(metadata))
    original = dcp.FileSystemReader(source).read_metadata()
    for key, value in original.state_dict_metadata.items():
        if key not in replacements and metadata.state_dict_metadata[key] != value:
            raise ValueError(f"Unrelated DCP key changed: {key}")
    for index, value in original.storage_data.items():
        if index.fqn not in replacements and metadata.storage_data[index] != value:
            raise ValueError(f"Unrelated DCP storage changed: {index.fqn}")


def fingerprint(dataset, spec, jitter):
    value = dataset.fingerprint_for_spec(spec)
    return hashlib.sha256(f"{value}:first-frame-color-jitter-v1:42".encode()).hexdigest() if jitter else value


def prepare(checkpoint, recipe, publication_root, output, report_dir):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    root, report_dir = Path(publication_root), Path(report_dir)
    if not (checkpoint / ".save_complete").is_file() or output.exists() or checkpoint in output.parents:
        raise ValueError("Need a complete source checkpoint and independent new output")
    report_dir.mkdir(parents=True, exist_ok=True)
    publication = json.loads((root / "bench2dex_physical_action_upgrade_20261102.json").read_text())
    if (
        publication.get("status") != "passed"
        or publication.get("normalization_policy") != "robot-limit-affine-no-clip-v1"
    ):
        raise ValueError("Missing passed physical-limit publication audit")
    preimage = root / publication["metadata_preimage"]
    before_action = json.loads((preimage / "meta/action_normalization.json").read_text())
    before_state = json.loads((preimage / "meta/state_normalization.json").read_text())
    after_action = json.loads((root / "meta/action_normalization.json").read_text())
    allowed_domains = set(after_action["domains"])
    if len(allowed_domains) != 12 or allowed_domains != set(before_action["domains"]):
        raise ValueError("Bench2Dex migration must retain exactly its twelve named domains")

    trainer = {"action128_contract": ""}
    dcp.load(trainer, storage_reader=dcp.FileSystemReader(checkpoint / "trainer"), no_dist=True)
    old_contract = json.loads(trainer["action128_contract"])
    old_spec = old_contract["spec"]
    model_reader = dcp.FileSystemReader(checkpoint / "model")
    model_meta = model_reader.read_metadata().state_dict_metadata
    digest_keys = [key for key in model_meta if key.endswith("action_normalization.contract_digest")]
    digests = {key: torch.empty(16, dtype=torch.uint8) for key in digest_keys}
    if not digests:
        raise ValueError("Missing saved normalization digest")
    dcp.load(digests, storage_reader=model_reader, no_dist=True)
    validate_checkpoint_digest(old_contract, digests)

    cfg = tomllib.loads(Path(recipe).read_text())
    data_cfg = cfg["dataloader_train"]
    catalog_paths = [entry["path"] for entry in data_cfg["datasets"]]
    rows = [ParquetCatalogDataset(path)[0] for path in catalog_paths]
    if len(rows) != 58 or sum(row["data"]["root"] == str(root) for row in rows) != 1:
        raise ValueError("Need the unchanged58-source recipe and the exact Bench2Dex root")
    new_spec = build_action128_spec([row["data"]["root"] for row in rows])
    if new_spec["registry"] != old_spec["registry"] or new_spec["projection_routing"] != old_spec["projection_routing"]:
        raise ValueError("Normalization migration cannot change domains, embodiments or projection routing")
    names = old_spec["registry"]["domains"]
    changed = []
    for index, domain in enumerate(names):
        differs = any(old_spec[field][index] != new_spec[field][index] for field in ("transforms", "source_digests"))
        if differs:
            if domain not in allowed_domains:
                raise ValueError(f"Unapproved normalization change outside Bench2Dex: {domain}")
            changed.append(domain)
    if set(changed) != allowed_domains:
        raise ValueError("Expected exactly the twelve approved Bench2Dex normalization changes")
    # Rebuild the publisher's preimage contract; retain current unchanged routing.
    shadow = report_dir / "bench2dex_preimage_contract"
    (shadow / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "meta/info.json", "meta/action_normalization.json", "meta/state_normalization.json"):
        shutil.copy2(preimage / name, shadow / name)
    for name in ("meta/domains.json", "meta/state_action_projection_routing.json", "meta/a1_structural_gate.json"):
        shutil.copy2(root / name, shadow / name)
    preimage_spec = build_action128_spec([str(shadow)])
    if load_state_transforms(shadow) != before_state:
        raise ValueError("Invalid publisher preimage State contract")
    for index, domain in enumerate(preimage_spec["registry"]["domains"]):
        old_index = names.index(domain)
        for field in ("transforms", "source_digests"):
            if preimage_spec[field][index] != old_spec[field][old_index]:
                raise ValueError(f"Publisher preimage differs from saved checkpoint: {domain}/{field}")

    rank_paths = sorted((checkpoint / "dataloader").glob("rank_*.pkl"), key=lambda path: int(path.stem[5:]))
    if [path.name for path in rank_paths] != [f"rank_{rank}.pkl" for rank in range(160)]:
        raise ValueError("Prepare the audited160-rank checkpoint first")
    reference = pickle.loads(rank_paths[0].read_bytes())
    if list(reference["__fingerprints__"]) != [f"{row['dataset']}:{index}" for index, row in enumerate(rows)]:
        raise ValueError("Recipe source order differs from checkpoint")
    jitter = data_cfg.get("first_frame_color_jitter", False)
    new_fingerprints = {}
    inventory = []
    for index, row in enumerate(rows):
        options = dict(row["data"])
        if options.pop("adapter") != "unified_action128":
            raise ValueError("Only the unchanged UnifiedAction128 recipe is supported")
        options.pop("format", None)
        options["use_state"] = data_cfg["action_use_state"]
        options["video_fps_max"] = data_cfg["action_video_fps_default"]
        dataset = UnifiedAction128Dataset(**options, action128_spec=new_spec, seed=42)
        name = f"{row['dataset']}:{index}"
        new_fingerprints[name] = fingerprint(dataset, new_spec, jitter)
        if str(dataset.root) == str(root):
            original_state = dataset._fingerprint_inputs["state"]["normalization"]
            dataset._fingerprint_inputs["state"]["normalization"] = before_state
            old_fingerprint = fingerprint(dataset, old_spec, jitter)
            dataset._fingerprint_inputs["state"]["normalization"] = original_state
        else:
            old_fingerprint = fingerprint(dataset, old_spec, jitter)
        if old_fingerprint != reference["__fingerprints__"][name]:
            raise ValueError(f"Unchanged ordered metadata/processing no longer reproduces saved fingerprint: {name}")
        inventory.append(
            {
                "name": name,
                "root": str(dataset.root),
                "episodes": len(dataset.episodes),
                "windows": len(dataset),
                "old_fingerprint": old_fingerprint,
                "new_fingerprint": new_fingerprints[name],
            }
        )
        print(json.dumps({"validated_source": name, "windows": len(dataset)}), flush=True)
        del dataset
    new_contract = {**old_contract, "spec": new_spec}
    normalization = Action128Normalization(new_spec, old_contract["vision_spatial_position_mode"]).state_dict()
    replacements = {}
    for key, value in model_meta.items():
        if ".action_normalization." in key:
            suffix = key.rsplit(".action_normalization.", 1)[1]
            if suffix not in normalization or tuple(value.size) != tuple(normalization[suffix].shape):
                raise ValueError(f"Unexpected normalization key or shape: {key}")
            replacements[key] = normalization[suffix].to(value.properties.dtype)
    if len(replacements) != 4 * len(digests):
        raise ValueError("Expected exactly four non-parameter buffers per normalization module")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".physical-migration-", dir=output.parent))
    try:
        overlay_component(checkpoint / "model", temporary / "model", replacements)
        overlay_component(
            checkpoint / "trainer",
            temporary / "trainer",
            {"action128_contract": json.dumps(new_contract, sort_keys=True)},
        )
        for component in ("optim", "scheduler"):
            shutil.copytree(checkpoint / component, temporary / component, copy_function=os.link)
        (temporary / "dataloader").mkdir()
        for path in rank_paths:
            state = pickle.loads(path.read_bytes())
            if (
                state["__fingerprints__"] != reference["__fingerprints__"]
                or state["global_id"] != reference["global_id"]
            ):
                raise ValueError("Saved rank source fingerprints/global cursor disagree")
            old_dump = pickle.dumps(state, protocol=5)
            original_fingerprints = state["__fingerprints__"]
            state["__fingerprints__"] = new_fingerprints
            (temporary / "dataloader" / path.name).write_bytes(pickle.dumps(state, protocol=5))
            state["__fingerprints__"] = original_fingerprints
            if pickle.dumps(state, protocol=5) != old_dump:
                raise ValueError("Migration changed a cursor, schedule or execution/RNG state")
        check = {key: torch.empty_like(value) for key, value in replacements.items()}
        dcp.load(check, storage_reader=dcp.FileSystemReader(temporary / "model"), no_dist=True)
        for key, value in replacements.items():
            if not torch.equal(check[key], value):
                raise ValueError(f"Patched normalization buffer failed readback: {key}")
        validate_checkpoint_digest(new_contract, {key: value for key, value in check.items() if key in digests})
        restored = {"action128_contract": "", "iteration": 0}
        dcp.load(restored, storage_reader=dcp.FileSystemReader(temporary / "trainer"), no_dist=True)
        if json.loads(restored["action128_contract"]) != new_contract or restored["iteration"] != 5200:
            raise ValueError("Patched trainer contract/iteration failed readback")
        if (checkpoint / "reshard_manifest.json").exists():
            shutil.copy2(checkpoint / "reshard_manifest.json", temporary / "reshard_manifest.json")
        report = {
            "state": "passed",
            "source_checkpoint": str(checkpoint),
            "output": str(output),
            "publication_root": str(root),
            "publication_completed_at": publication["completed_at"],
            "changed_domains": changed,
            "changed_model_buffers": sorted(replacements),
            "iteration": 5200,
            "data_global_id": reference["global_id"],
            "ranks": 160,
            "cursor_rng_schedule_preserved": True,
            "trained_parameters_optimizer_scheduler_preserved": True,
            "sources": inventory,
            "scope": "User-approved normalization/target-mask semantic change. All learned tensor storage, optimizer and scheduler retained; new transform constants and strict fingerprints. No numerical equivalence to the old label contract is claimed.",
        }
        (temporary / "normalization_migration.json").write_text(json.dumps(report, indent=2) + "\n")
        (temporary / ".save_complete").write_text("Audited explicit Bench2Dex physical normalization migration\n")
        temporary.rename(output)
        (report_dir / "normalization_migration.json").write_text(json.dumps(report, indent=2) + "\n")
        (report_dir / "new_action128_contract.json").write_text(json.dumps(new_contract, indent=2) + "\n")
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-recipe", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = prepare(args.checkpoint, args.source_recipe, args.publication_root, args.output, args.report_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}), flush=True)


if __name__ == "__main__":
    main()
