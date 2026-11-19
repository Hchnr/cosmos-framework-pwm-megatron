# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independently audit queued TP4/FSDP4 connection-limit benchmark results.

Recompute all slowest-rank updates and sample/token rates from rank logs, inspect
the pinned prepared input schemas, and compare all fields with both the recorded
summary and a comparator loaded explicitly from the immutable snapshot.
Numerical correctness and strict GPU exclusivity remain outside this audit.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
from pathlib import Path


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_snapshot_comparator(snapshot, manifest):
    comparator = snapshot.resolve() / "tools/compare_tp4_fsdp4_microbatch_benchmark.py"
    expected = manifest.get("files", {}).get("tools/compare_tp4_fsdp4_microbatch_benchmark.py")
    if not isinstance(expected, str) or sha256(comparator) != expected:
        raise ValueError("Snapshot comparator differs from its queued SHA")
    spec = importlib.util.spec_from_file_location("_snapshot_microbatch_comparator", comparator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, expected


def verify_baseline_files(manifest):
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty queued source and input manifest is required")
    for name, expected in files.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned source or input changed: {name}")


def verify_snapshot_files(snapshot, manifest):
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty snapshot manifest is required")
    actual = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*.py")}
    if actual != set(files):
        raise ValueError("Snapshot Python file set changed")
    for name, expected in files.items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f"Snapshot Python file changed: {name}")


def compare_values(expected, actual, label="root"):
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            raise ValueError(f"Value mismatch at {label}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if (
            not math.isfinite(expected)
            or not math.isfinite(actual)
            or not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"Numeric mismatch at {label}: {expected!r} vs {actual!r}")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(f"Fields differ at {label}")
        for key in expected:
            compare_values(expected[key], actual[key], f"{label}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"Length mismatch at {label}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_values(left, right, f"{label}[{index}]")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"Value mismatch at {label}")


def independent_side(root, backend):
    contracts = [json.loads((root / f"contract.rank{rank}.json").read_text()) for rank in range(4)]
    first = contracts[0]
    if any(contract != first for contract in contracts[1:]):
        raise ValueError("Rank contracts differ")
    expected_topology = (4, 1, 1) if backend == "tp" else (1, 1, 4)
    if first["backend"] != backend or (first["tp"], first["pp"], first["dp"]) != expected_topology:
        raise ValueError("Unexpected TP/PP/DP topology")
    if first.get("checkpoint_writes") is not False:
        raise ValueError("Expected checkpoint writes to be disabled")
    if (first["precision"], first["microsteps_per_update"], first["warmup_updates"], first["measured_updates"]) != (
        "bfloat16",
        3,
        2,
        5,
    ):
        raise ValueError("Expected BF16, three microsteps, two warmups and five measured updates")
    batches = first["per_rank_microbatches"]
    if (
        len(batches) != 3
        or [row["microstep"] for row in batches] != [0, 1, 2]
        or [row["samples"] for row in batches] != [9, 11, 13]
    ):
        raise ValueError("Per-microstep sample contract does not match 9/11/13")
    if any(row.get("payload_kind", "prepared") != "prepared" for row in batches):
        raise ValueError("Expected prepared inputs without VAE/noising")
    logs = root / "training"
    if backend == "tp":
        logs /= "cosmos3_action/action128/tp4_microbatch"
    rows = []
    for rank in range(4):
        records = [json.loads(line) for line in (logs / f"updates.rank{rank}.jsonl").read_text().splitlines()]
        if [row["iteration"] for row in records] != list(range(1201, 1208)):
            raise ValueError(f"Rank {rank} has incomplete or reordered updates")
        times = [row["update_seconds"] for row in records]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in times
        ):
            raise ValueError(f"Rank {rank} has invalid update times")
        rows.append(times)
    critical = [max(rank[index] for rank in rows) for index in range(7)]
    measured = critical[2:]
    mean = statistics.mean(measured)
    total_samples = sum(batch["samples"] for batch in batches) * first["dp"]
    return {
        "contract": first,
        "slowest_rank_update_seconds": critical,
        "measured_mean_update_seconds": mean,
        "measured_median_update_seconds": statistics.median(measured),
        "amortized_microstep_seconds": mean / 3,
        "global_samples_per_update": total_samples,
        "global_samples_per_second": total_samples / mean,
        "real_tokens_per_second": {
            key: sum(batch["tokens"][key] for batch in batches) * first["dp"] / mean
            for key in ("text", "vision", "action")
        },
    }


def independent_compare(tp_root, fsdp_root):
    tp, fsdp = independent_side(tp_root, "tp"), independent_side(fsdp_root, "fsdp")
    for key in ("per_rank_microbatches", "microsteps_per_update", "warmup_updates", "measured_updates", "precision"):
        if tp["contract"][key] != fsdp["contract"][key]:
            raise ValueError(f"TP/FSDP contract mismatch: {key}")
    return {
        "measurement_completed": True,
        "scope": "real immutable prepared input replay; training compute, communication and optimizer; media/VAE excluded",
        "per_microstep_batches_exact": True,
        "global_batch_matched": False,
        "numerical_acceptance": False,
        "tp": tp,
        "fsdp": fsdp,
        "tp_over_fsdp_sample_throughput": tp["global_samples_per_second"] / fsdp["global_samples_per_second"],
    }


def prepared_metadata(header, buffers):
    """Read only contract-relevant fields from the verified transport schema."""
    schema = json.loads(header)
    if schema.get("version") != 1:
        raise ValueError("Unexpected prepared transport schema version")
    root = dict(schema["tree"]["dict"])
    packed = root["packed_sequence"]
    if packed["class"] != "PackedSequence":
        raise ValueError("Prepared payload does not contain a PackedSequence")
    fields = packed["fields"]
    samples = root["sample_count"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples != len(fields["sample_lens"]["list"]):
        raise ValueError("Prepared sample count disagrees with packed sample lengths")

    def elements(node):
        if node is None:
            return 0
        index = node.get("tensor")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(schema["tensors"]):
            raise ValueError("Invalid tensor descriptor reference")
        spec = schema["tensors"][index]
        shape, offset, dtype = spec["shape"], spec["offset"], spec["dtype"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
            raise ValueError("Invalid tensor descriptor shape")
        count = math.prod(shape)
        buffer = buffers[dtype]
        if (
            str(buffer.dtype) != dtype
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset + count > buffer.numel()
        ):
            raise ValueError("Tensor descriptor exceeds its typed buffer")
        return count

    vision, action = fields["vision"], fields["action"]
    image = dict(root["loss_inputs"]["dict"])["is_image_batch"]
    if not isinstance(image, bool):
        raise ValueError("Prepared modality flag is not boolean")
    return {
        "samples": samples,
        "image": image,
        "tokens": {
            "text": elements(fields["text_indexes"]),
            "vision": elements(vision["fields"]["sequence_indexes"]) if vision is not None else 0,
            "action": elements(action["fields"]["sequence_indexes"]) if action is not None else 0,
        },
    }


def verify_prepared_payloads(batches, baseline_manifest):
    # Runtime PackedSequence reconstruction interprets CUDA device markers, so
    # inspect only the verified serialized schema and CPU tensor buffers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    verified = []
    for expected in batches:
        path = Path(expected["source"]).resolve()
        if str(path) not in baseline_manifest["files"]:
            raise ValueError(f"Fixture not pinned in baseline manifest: {path}")
        header, buffers = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha256(header)
        for dtype, tensor in sorted(buffers.items()):
            digest.update(dtype.encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().data)
        observed = {"payload_sha256": digest.hexdigest(), **prepared_metadata(header, buffers)}
        compare_values({key: expected[key] for key in observed}, observed, f"fixture{expected['microstep']}")
        verified.append({"source": str(path), **observed})
        del header, buffers
    return verified


def audit(sequence, snapshot, snapshot_manifest, baseline_manifest, evidence):
    summary = json.loads((sequence / "summary.json").read_text())
    if summary.get("completed") is not True or summary.get("failure") is not None:
        raise ValueError("Connection experiment did not complete")
    if (
        summary.get("full_model_numerical_acceptance") is not False
        or summary.get("performance_acceptance") is not False
    ):
        raise ValueError("Unexpected acceptance claim in diagnostic summary")
    if snapshot_manifest.get("snapshot") != str(snapshot.resolve()):
        raise ValueError("Snapshot path does not match its queued manifest")
    verify_snapshot_files(snapshot, snapshot_manifest)
    verify_baseline_files(baseline_manifest)
    comparator, comparator_sha = load_snapshot_comparator(snapshot, snapshot_manifest)
    root_path = Path(evidence["root_comparator"])
    snapshot_path = snapshot.resolve() / "tools/compare_tp4_fsdp4_microbatch_benchmark.py"
    if Path(evidence["snapshot_comparator"]).resolve() != snapshot_path:
        raise ValueError("Prelaunch evidence refers to a different snapshot")
    if evidence["snapshot_comparator_sha256"] != comparator_sha or evidence["queue_launched"] is not False:
        raise ValueError("Prelaunch comparator evidence is inconsistent")
    first_start = min(
        summary["cases"][f"conn{conn}_{mode}"]["started"]
        for conn in (1, 8)
        for mode in ("tp_shared", "tp_async", "fsdp")
    )
    if evidence["observed_unix"] >= first_start:
        raise ValueError("Comparator evidence was not recorded before performance measurements")
    results = {}
    verified_batches = None
    for connection in (1, 8):
        for mode in ("shared", "async"):
            for label in (f"conn{connection}_tp_{mode}", f"conn{connection}_fsdp"):
                if summary["cases"][label]["returncode"] != 0:
                    raise ValueError(f"Benchmark failed: {label}")
            tp, fsdp = sequence / f"conn{connection}_tp_{mode}", sequence / f"conn{connection}_fsdp"
            independent = independent_compare(tp, fsdp)
            pinned = comparator.compare(tp, fsdp)
            recorded = summary["comparisons"][f"conn{connection}_{mode}"]
            compare_values(independent, pinned, f"conn{connection}.{mode}.snapshot")
            compare_values(independent, recorded, f"conn{connection}.{mode}.recorded")
            batches = independent["tp"]["contract"]["per_rank_microbatches"]
            if verified_batches is None:
                verified_batches = batches
            elif verified_batches != batches:
                raise ValueError("Inputs differ across connection experiments")
            results[f"conn{connection}_{mode}"] = independent
    fixture = verify_prepared_payloads(verified_batches, baseline_manifest)
    verify_baseline_files(baseline_manifest)
    verify_snapshot_files(snapshot, snapshot_manifest)
    root_after = sha256(root_path)
    return {
        "passed": True,
        "scope": "offline independent rank-timing/sample/payload audit of prepared-input connection comparison",
        "snapshot_comparator_sha256": comparator_sha,
        "root_comparator_sha256_prelaunch": evidence["root_comparator_sha256"],
        "root_comparator_sha256_at_audit": root_after,
        "root_sha_equal_at_both_observations": root_after == evidence["root_comparator_sha256"] == comparator_sha,
        "root_sha_limitation": "Two point-in-time hashes cannot prove the root comparator remained unchanged throughout measurement; recorded results were independently recomputed with the pinned snapshot comparator.",
        "full_model_numerical_acceptance": False,
        "performance_acceptance": False,
        "gpu_isolation_limitation": "Original measurement used a cooperative lock and polling; transient foreign users cannot be excluded.",
        "fixture": fixture,
        "comparisons": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--prelaunch-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(
        args.sequence,
        args.snapshot,
        json.loads(args.snapshot_manifest.read_text()),
        json.loads(args.baseline_manifest.read_text()),
        json.loads(args.prelaunch_evidence.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
