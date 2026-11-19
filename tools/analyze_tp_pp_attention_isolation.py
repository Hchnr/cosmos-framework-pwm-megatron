"""Compare attention traces by rank, retaining diagnostic scope and shard identity.

The recorder stores tensors in native layout on one side and potentially
head-sharded tensors on the other. This program compares each local shard to
the matching slice of the native tensor. It never substitutes for the complete
model/state acceptance gates; sampling can miss differences elsewhere.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

FILENAME = re.compile(r"^layer(?P<layer>\d+)\.(?P<tag>[^.]+)\.(?P<kind>[^.]+)\.rank(?P<rank>\d+)\.pt$")


def load_records(directory):
    records = {}
    for path in sorted(directory.glob("*.pt")):
        match = FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected trace filename: {path}")
        record = torch.load(path, map_location="cpu", weights_only=False)
        key = (int(match["layer"]), match["tag"], match["kind"])
        if (record["layer"], record["tag"], record["kind"], record["rank"]) != (*key, int(match["rank"])):
            raise ValueError(f"Trace metadata does not match filename: {path}")
        if record["dtype"] != str(record["value"].dtype):
            raise ValueError(f"Trace dtype mismatch: {path}")
        value = record["value"].contiguous().reshape(-1).view(torch.uint8)
        if hashlib.sha256(value.numpy().tobytes()).hexdigest() != record["stored_sha256"]:
            raise ValueError(f"Trace content hash mismatch: {path}")
        records.setdefault(key, []).append(record)
    if not records:
        raise ValueError(f"No trace tensors found in {directory}")
    return records


def _sample(value, shape, scope, target_rows):
    if scope == {"kind": "full"}:
        if tuple(value.shape) != tuple(shape):
            raise ValueError("Full trace tensor shape does not match its metadata")
        if target_rows is None:
            return value
        if shape[0] <= 2 * target_rows:
            raise ValueError("First/last sample does not have disjoint rows")
        return torch.cat((value[:target_rows], value[-target_rows:]), dim=0)
    if scope.get("kind") != "first_last_dim0" or set(scope) != {"kind", "rows_each_end"}:
        raise ValueError("Unsupported trace sample scope")
    rows = scope["rows_each_end"]
    if rows < 1 or shape[0] <= 2 * rows or tuple(value.shape) != (2 * rows, *shape[1:]):
        raise ValueError("Invalid sampled trace shape")
    if target_rows is None or target_rows > rows:
        raise ValueError("Cannot recover rows absent from a trace")
    return torch.cat((value[:target_rows], value[-target_rows:]), dim=0)


def compare(reference, local):
    if reference["forward"] != local["forward"] or reference["dtype"] != local["dtype"]:
        raise ValueError("Trace forward index or dtype differs")
    global_shape, local_shape = reference["shape"], local["shape"]
    if len(global_shape) != len(local_shape):
        raise ValueError("Tensor dimensionality differs")
    size, rank = local["tp_size"], local["tp_rank"]
    if size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid TP topology in trace")
    mismatched = [i for i, (a, b) in enumerate(zip(global_shape, local_shape, strict=True)) if a != b]
    if not mismatched:
        axis = None
    elif (
        len(mismatched) == 1 and mismatched[0] != 0 and global_shape[mismatched[0]] == size * local_shape[mismatched[0]]
    ):
        axis = mismatched[0]
    else:
        raise ValueError(f"Cannot uniquely map native shape {global_shape} to TP shape {local_shape}")
    scopes = (reference["scope"], local["scope"])
    sampled_rows = [s["rows_each_end"] for s in scopes if s.get("kind") == "first_last_dim0"]
    target_rows = min(sampled_rows) if sampled_rows else None
    expected = _sample(reference["value"], global_shape, scopes[0], target_rows)
    actual = _sample(local["value"], local_shape, scopes[1], target_rows)
    if axis is not None:
        expected = expected.narrow(axis, rank * local_shape[axis], local_shape[axis])
    expected, actual = expected.double(), actual.double()
    if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
        raise ValueError("Trace includes nonfinite tensor values")
    difference = expected - actual
    return {
        "rank": local["rank"],
        "tp_rank": rank,
        "tp_size": size,
        "native_event_order": reference.get("event_order"),
        "tp_event_order": local.get("event_order"),
        "shard_axis": axis,
        "native_shape": global_shape,
        "local_shape": local_shape,
        "scope": "full" if target_rows is None else {"first_last_dim0": target_rows},
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-30)).item(),
        "maximum_absolute_error": difference.abs().max().item() if difference.numel() else 0.0,
    }


def analyze(native_directory, tp_directory):
    native, tp = load_records(native_directory), load_records(tp_directory)
    if native.keys() != tp.keys():
        raise ValueError(
            f"Trace coverage mismatch: native_only={sorted(native.keys() - tp.keys())}, tp_only={sorted(tp.keys() - native.keys())}"
        )
    rows = []
    for key in sorted(native):
        expected, candidates = native[key], tp[key]
        if len(expected) != 1 or expected[0]["tp_size"] != 1:
            raise ValueError(f"Expected one native trace for {key}")
        sizes = {record["tp_size"] for record in candidates}
        if len(sizes) != 1 or sorted(record["tp_rank"] for record in candidates) != list(range(next(iter(sizes)))):
            raise ValueError(f"Missing or duplicated TP shards for {key}")
        for candidate in sorted(candidates, key=lambda r: r["tp_rank"]):
            rows.append({"layer": key[0], "tag": key[1], "kind": key[2], **compare(expected[0], candidate)})
    return {
        "scope": "sampled operator diagnostics only; event order is rank-local, not a cross-rank causal order",
        "mathematical_acceptance": False,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--tp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.native, args.tp)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
