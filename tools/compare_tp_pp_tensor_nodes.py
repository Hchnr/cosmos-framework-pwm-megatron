# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare CPU node tensors described by JSON manifests and raw little-endian files.

Manifest v1 example::

    {"format": "cosmos_tp_pp_tensor_nodes", "version": 1, "byte_order": "little",
     "tensors": [{"name": "layer0.gate", "path": "gate.raw", "shape": [12, 4],
                  "dtype": "float32", "sha256": "<64 lowercase hex digits>"}]}

Each raw path is relative to its manifest directory and contains exactly one
contiguous tensor. No pickle, torch.load, checkpoint, or executable format is
accepted. This is a node report, not a training correctness certification.
Only names present in at least one manifest are checked; a separate expected
node contract is required to detect a name missing from both manifests. The
byte limit applies to each raw tensor, not total peak memory: both inputs and
intermediate copies may coexist.

Run with ``python -m tools.compare_tp_pp_tensor_nodes --reference ref.json
--candidate candidate.json --output report.json --relative-l2 1e-4
--zero-max-abs 2e-6 --min-cosine 0.999999``. A nonzero-reference tensor must
meet both relative L2 and cosine limits; a zero-reference tensor uses the
absolute limit. ``--require-bitwise`` adds a byte-equality requirement.
"""

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path, PurePosixPath

import torch

from tools.tp_pp_tensor_metrics import compare_tensor_metrics

FORMAT = "cosmos_tp_pp_tensor_nodes"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64}
SIZES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read_fd(fd, limit):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("expected a regular file within the size limit")
    chunks = []
    remaining = before.st_size
    while remaining:
        part = os.read(fd, min(1 << 20, remaining))
        if not part:
            raise ValueError("file truncated while reading")
        chunks.append(part)
        remaining -= len(part)
    if _identity(before) != _identity(os.fstat(fd)):
        raise ValueError("file changed while reading")
    return b"".join(chunks), before


def _manifest(path):
    path = Path(path).absolute()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        raw, before = _read_fd(fd, 8 << 20)
        if _identity(before) != _identity(os.stat(path, follow_symlinks=False)):
            raise ValueError("manifest path changed while reading")
    finally:
        os.close(fd)
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if (
        not isinstance(data, dict)
        or data.get("format") != FORMAT
        or type(data.get("version")) is not int
        or data["version"] != 1
        or data.get("byte_order") != "little"
        or not isinstance(data.get("tensors"), list)
    ):
        raise ValueError("invalid tensor node manifest header")
    if sys.byteorder != "little":
        raise ValueError("little-endian host required")
    records, paths = {}, set()
    for item in data["tensors"]:
        if not isinstance(item, dict) or set(item) != {"name", "path", "shape", "dtype", "sha256"}:
            raise ValueError("invalid tensor record fields")
        name, rel, shape, dtype, sha = (item[key] for key in ("name", "path", "shape", "dtype", "sha256"))
        if not isinstance(name, str) or not name or name in records:
            raise ValueError("tensor names must be nonempty and unique")
        if not isinstance(rel, str) or not rel or "\\" in rel:
            raise ValueError("invalid relative tensor path")
        parts = PurePosixPath(rel)
        if parts.is_absolute() or any(part in (".", "..") for part in parts.parts) or str(parts) != rel:
            raise ValueError("tensor path must be a normalized relative path")
        if rel in paths:
            raise ValueError("tensor paths must be unique")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("tensor shape must contain nonnegative integers")
        if (
            dtype not in DTYPES
            or not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("invalid tensor dtype or SHA-256")
        records[name] = item
        paths.add(rel)
    if not records:
        raise ValueError("empty tensor manifest")
    return path.parent, records, hashlib.sha256(raw).hexdigest()


def _tensor(root, record, max_bytes):
    parts = PurePosixPath(record["path"]).parts
    fds = []
    try:
        directory = os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(directory)
        for part in parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            fds.append(directory)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
        fds.append(fd)
        expected = math.prod(record["shape"]) * SIZES[record["dtype"]]
        if expected > max_bytes:
            raise ValueError("tensor exceeds configured byte limit")
        data, before = _read_fd(fd, max_bytes)
        if len(data) != expected or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("tensor byte count or SHA-256 mismatch")
        if _identity(before) != _identity(os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)):
            raise ValueError("tensor path changed while reading")
        dtype = DTYPES[record["dtype"]]
        if not data:
            return torch.empty(record["shape"], dtype=dtype)
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(record["shape"])
    finally:
        for fd in reversed(fds):
            os.close(fd)


def compare_manifests(
    reference, candidate, *, relative_l2, zero_max_abs, min_cosine, require_bitwise=False, max_tensor_bytes=256 << 20
):
    """Return a fail-closed, JSON-compatible report using caller-specified limits."""
    limits = (relative_l2, zero_max_abs, min_cosine)
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in limits)
        or relative_l2 < 0
        or zero_max_abs < 0
        or not -1 <= min_cosine <= 1
        or type(require_bitwise) is not bool
        or type(max_tensor_bytes) is not int
        or max_tensor_bytes <= 0
    ):
        raise ValueError("invalid explicit comparison limits")
    report = {
        "format": "cosmos_tp_pp_node_comparison",
        "version": 1,
        "passed": False,
        "scope": "listed CPU node tensors only; not training correctness",
        "limits": {
            "relative_l2": relative_l2,
            "zero_reference_max_abs": zero_max_abs,
            "minimum_cosine": min_cosine,
            "require_bitwise": require_bitwise,
            "max_tensor_bytes": max_tensor_bytes,
        },
        "missing": [],
        "extra": [],
        "nodes": [],
        "unpaired_integrity": [],
        "errors": [],
    }
    try:
        root_a, records_a, sha_a = _manifest(reference)
        root_b, records_b, sha_b = _manifest(candidate)
        report["manifest_sha256"] = {"reference": sha_a, "candidate": sha_b}
        report["missing"] = sorted(set(records_a) - set(records_b))
        report["extra"] = sorted(set(records_b) - set(records_a))
        for side, root, records, names in (
            ("reference", root_a, records_a, report["missing"]),
            ("candidate", root_b, records_b, report["extra"]),
        ):
            for name in names:
                try:
                    _tensor(root, records[name], max_tensor_bytes)
                    report["unpaired_integrity"].append({"side": side, "name": name, "passed": True})
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    report["unpaired_integrity"].append(
                        {"side": side, "name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
                    )
        for name in sorted(set(records_a) & set(records_b)):
            left, right = records_a[name], records_b[name]
            try:
                a, b = _tensor(root_a, left, max_tensor_bytes), _tensor(root_b, right, max_tensor_bytes)
                if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
                    raise ValueError("shape or dtype mismatch")
                metrics = compare_tensor_metrics(a, b)
                if metrics["status"] != "ok":
                    passed = False
                elif metrics["reference_norm_zero"]:
                    passed = metrics["max_abs"] is not None and metrics["max_abs"] <= zero_max_abs
                else:
                    passed = (
                        metrics["relative_l2"] is not None
                        and metrics["relative_l2"] <= relative_l2
                        and metrics["cosine"] is not None
                        and metrics["cosine"] >= min_cosine
                    )
                if require_bitwise:
                    passed = passed and metrics["bitwise_equal"]
                report["nodes"].append(
                    {
                        "name": name,
                        "passed": bool(passed),
                        "metrics": metrics,
                        "reference_sha256": left["sha256"],
                        "candidate_sha256": right["sha256"],
                    }
                )
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                report["nodes"].append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
        report["passed"] = (
            not report["missing"] and not report["extra"] and all(row["passed"] for row in report["nodes"])
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-l2", type=float, required=True)
    parser.add_argument("--zero-max-abs", type=float, required=True)
    parser.add_argument("--min-cosine", type=float, required=True)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--max-tensor-bytes", type=int, default=256 << 20)
    args = parser.parse_args(argv)
    try:
        report = compare_manifests(
            args.reference,
            args.candidate,
            relative_l2=args.relative_l2,
            zero_max_abs=args.zero_max_abs,
            min_cosine=args.min_cosine,
            require_bitwise=args.require_bitwise,
            max_tensor_bytes=args.max_tensor_bytes,
        )
    except ValueError as exc:
        report = {
            "format": "cosmos_tp_pp_node_comparison",
            "version": 1,
            "passed": False,
            "errors": [f"ValueError: {exc}"],
        }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
