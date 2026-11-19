"""Measure CPU serialization/readback of one immutable real prepared payload.

Uses the production buffered temporary-file/link publication protocol. Excludes
CUDA synchronization, device transfer, encoding, and cold-cache guarantees.
"""

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch


def fingerprint(payload):
    header, buffers = payload
    digest = hashlib.sha256(header)
    for dtype, value in sorted(buffers.items()):
        if value.device.type != "cpu":
            raise ValueError("Spool benchmark requires CPU-only encoded buffers")
        digest.update(dtype.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().data)
    return digest.hexdigest()


def publish(payload, target):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--shared-directory", type=Path, required=True)
    parser.add_argument("--local-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or args.output.exists():
        raise ValueError("Require positive repetitions and a new report")
    source = args.source.resolve()
    initial = source.stat()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    expected = fingerprint(payload)
    rows = []
    for label, directory in (("shared", args.shared_directory), ("local", args.local_directory)):
        directory.mkdir(parents=True, exist_ok=False)
        writes, reads = [], []
        for iteration in range(args.repetitions + 1):
            target = directory / f"sample{iteration:03d}.pt"
            started = time.perf_counter()
            publish(payload, target)
            write_seconds = time.perf_counter() - started
            started = time.perf_counter()
            restored = torch.load(target, map_location="cpu", weights_only=False)
            read_seconds = time.perf_counter() - started
            if fingerprint(restored) != expected:
                raise ValueError("Prepared payload changed during spool I/O")
            del restored
            if iteration:
                writes.append(write_seconds)
                reads.append(read_seconds)
            target.unlink()
        rows.append(
            {
                "storage": label,
                "directory": str(directory.resolve()),
                "write_seconds": writes,
                "read_seconds": reads,
                "median_write_seconds": statistics.median(writes),
                "median_read_seconds": statistics.median(reads),
                "all_readback_payloads_exact": True,
            }
        )
    final = source.stat()
    if (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns):
        raise ValueError("Published source payload changed during benchmark")
    report = {
        "passed": True,
        "scope": "single real encoded CPU payload, buffered publication and warm readback only; concurrent diagnostics, not production throughput",
        "source": str(source),
        "payload_sha256": expected,
        "source_file_bytes": initial.st_size,
        "source_unchanged": True,
        "cuda_initialized": torch.cuda.is_initialized(),
        "rows": rows,
        "production_performance_accepted": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
