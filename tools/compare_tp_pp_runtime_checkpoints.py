"""Compare complete same-topology runtime states with bounded arithmetic workspaces."""

import argparse
import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader, release_file_cache


class _ArchiveSlice(io.RawIOBase):
    def __init__(self, descriptor, offset, length):
        self.descriptor, self.offset, self.length, self.position = descriptor, offset, length, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        base = 0 if whence == os.SEEK_SET else self.position if whence == os.SEEK_CUR else self.length
        self.position = base + offset
        if not 0 <= self.position <= self.length:
            raise ValueError("Tensor archive seek outside its DCP storage item")
        return self.position

    def read(self, size=-1):
        available = self.length - self.position
        data = os.pread(self.descriptor, available if size < 0 else min(size, available), self.offset + self.position)
        self.position += len(data)
        return data


class PayloadHashes:
    """Stream exact tensor payloads without materializing multi-GB Adam buckets.

    Matching pickle metadata plus storage bytes proves identical logical values.
    All floating storage is also checked for finiteness; unsupported or nonfinite
    payloads fall back to the normal tensor comparison.
    """

    def __init__(self, root, reader):
        self.root, self.files, self.items = root, {}, {}
        for index, location in reader.reader.read_metadata().storage_data.items():
            self.items.setdefault(index.fqn, []).append((tuple(index.offset or ()), location))

    def digest(self, name, dtype):
        floating = {torch.float64: "f8", torch.float32: "f4", torch.float16: "f2", torch.bfloat16: "u2"}
        if dtype not in floating and dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ):
            return None
        result = hashlib.sha256()
        for offset, location in sorted(self.items.get(name, []), key=lambda item: item[0]):
            path = self.root / location.relative_path
            if path not in self.files:
                self.files[path] = os.open(path, os.O_RDONLY)
            view = _ArchiveSlice(self.files[path], location.offset, location.length)
            with zipfile.ZipFile(view) as archive:
                entries = {entry.filename.split("/", 1)[-1]: entry for entry in archive.infolist()}
                if "data.pkl" not in entries or not any(key.startswith("data/") for key in entries):
                    return None
                byteorder = archive.read(entries["byteorder"]).decode() if "byteorder" in entries else "little"
                result.update(repr(offset).encode())
                for key, entry in sorted(entries.items()):
                    if key != "data.pkl" and key != "byteorder" and not key.startswith("data/"):
                        continue
                    result.update(key.encode() + b"\0" + str(entry.file_size).encode() + b"\0")
                    with archive.open(entry) as stream:
                        while data := stream.read(4 << 20):
                            result.update(data)
                            if key.startswith("data/") and dtype in floating:
                                numeric = np.dtype(floating[dtype]).newbyteorder("<" if byteorder == "little" else ">")
                                if len(data) % numeric.itemsize:
                                    return None
                                values = np.frombuffer(data, dtype=numeric)
                                finite = (
                                    not np.any((values & 0x7F80) == 0x7F80)
                                    if dtype == torch.bfloat16
                                    else np.isfinite(values).all()
                                )
                                if not finite:
                                    return None
        return result.hexdigest() if name in self.items else None

    def close(self):
        for descriptor in self.files.values():
            os.close(descriptor)


def object_differences(left, right, prefix=""):
    if isinstance(left, torch.Tensor):
        return [] if isinstance(right, torch.Tensor) and torch.equal(left, right) else [prefix]
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [prefix + ".keys"]
        return [item for key in left for item in object_differences(left[key], right[key], f"{prefix}.{key}")]
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [prefix + ".length"]
        return [
            item
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for item in object_differences(a, b, f"{prefix}.{index}")
        ]
    return [] if left == right else [prefix]


def tensor_difference(left, right, *, relative_tolerance, absolute_tolerance, exact, local_windows=True):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Checkpoint tensor shape/dtype mismatch")
    if exact or not left.is_floating_point():
        return {"passed": torch.equal(left, right), "exact_required": True}
    a, b = left.reshape(-1), right.reshape(-1)
    aa = bb = dot = delta = maximum = 0.0
    worst_chunk = 0.0
    passed = True
    for begin in range(0, a.numel(), 1 << 20):
        x, y = a[begin : begin + (1 << 20)].double(), b[begin : begin + (1 << 20)].double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            return {"passed": False, "nonfinite": True}
        xx, yy, xy, dd = (
            x.square().sum().item(),
            y.square().sum().item(),
            (x * y).sum().item(),
            (x - y).square().sum().item(),
        )
        error = (x - y).abs().max().item()
        relative = math.sqrt(dd / xx) if xx else (0.0 if not dd else float("inf"))
        # Check local windows as well as the whole bucket: a large unrelated
        # matrix must not hide a mismatch in an optimizer bucket subregion.
        if local_windows:
            passed &= error <= absolute_tolerance if xx <= 1e-16 else relative <= relative_tolerance
        worst_chunk = max(worst_chunk, relative)
        aa, bb, dot, delta = aa + xx, bb + yy, dot + xy, delta + dd
        maximum = max(maximum, error)
    relative_l2 = math.sqrt(delta / aa) if aa else 0.0
    if not local_windows:
        passed = maximum <= absolute_tolerance if aa <= 1e-16 else relative_l2 <= relative_tolerance
    return {
        "passed": passed,
        "relative_l2": relative_l2,
        "cosine": dot / math.sqrt(aa * bb) if aa * bb else 1.0,
        "maximum_absolute_error": maximum,
        "worst_chunk_relative_l2": worst_chunk,
        "local_windows_checked": local_windows,
    }


def compare(source, candidate, bundle, *, relative_tolerance=1e-4, absolute_tolerance=2e-6):
    for path in (source, candidate):
        if not (path / ".save_complete").is_file():
            raise ValueError(f"Incomplete checkpoint: {path}")
    manifests = [json.loads((path / "manifest.json").read_text()) for path in (source, candidate)]
    if manifests[0] != manifests[1]:
        raise ValueError("Checkpoint manifests differ")
    parameters = json.loads((bundle / "manifest.json").read_text())["parameters"]
    left, right = CanonicalDCPReader(source), CanonicalDCPReader(candidate)
    if left.metadata.keys() != right.metadata.keys():
        raise ValueError("Checkpoint key coverage differs")
    reports, objects = [], []
    payloads = [PayloadHashes(root, reader) for root, reader in ((source, left), (candidate, right))]
    try:
        for name, metadata in left.metadata.items():
            if isinstance(metadata, TensorStorageMetadata):
                if name not in parameters and not name.startswith("optimizer."):
                    raise ValueError(f"Unregistered model/optimizer tensor: {name}")
                other = right.metadata[name]
                if metadata.size != other.size or metadata.properties.dtype != other.properties.dtype:
                    raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
                hashes = (
                    [payload.digest(name, metadata.properties.dtype) for payload in payloads]
                    if metadata.chunks == other.chunks
                    else [None, None]
                )
                if hashes[0] is not None and hashes[0] == hashes[1]:
                    reports.append({"name": name, "passed": True, "exact_payload_equal": True})
                    continue
            a, b = left.read(name), right.read(name)
            if isinstance(metadata, TensorStorageMetadata):
                if name not in parameters and not name.startswith("optimizer."):
                    raise ValueError(f"Unregistered model/optimizer tensor: {name}")
                frozen = name in parameters and not parameters[name]["optimizer_fields"]
                reports.append(
                    {
                        "name": name,
                        **tensor_difference(
                            a,
                            b,
                            relative_tolerance=relative_tolerance,
                            absolute_tolerance=absolute_tolerance,
                            exact=frozen,
                        ),
                    }
                )
            else:
                if name.startswith("rank_state/"):
                    # Physical compiler/probe RNG is recorded separately from
                    # the strict data/noising and model-parallel RNG contracts.
                    a, b = a[0].copy(), b[0].copy()
                    a.pop("physical_rng")
                    b.pop("physical_rng")
                differences = object_differences(a, b, name)
                objects.append({"name": name, "passed": not differences, "differences": differences})
            del a, b
    finally:
        for payload in payloads:
            payload.close()
        release_file_cache(path for root in (source, candidate) for path in root.glob("*.distcp"))
    return {
        "passed": all(row["passed"] for row in reports + objects),
        "scope": "same-topology complete model/master/moments/step/scheduler/logical state comparison",
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "tensors": reports,
        "objects": objects,
        "production_576_slot_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.source, args.candidate, args.bundle)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("tensors", "objects")}))
    if not report["passed"]:
        raise AssertionError("Runtime checkpoint numerical/state comparison failed; see report")


if __name__ == "__main__":
    main()
