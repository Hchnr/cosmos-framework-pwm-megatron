"""Measure buffer-copy service time and exclusive non-NCCL occupancy."""

import argparse
import collections
import json
import re
import sqlite3
import statistics
from pathlib import Path

from analyze_megatron_fsdp_nsys import seconds


def is_buffer_copy(name):
    return (
        "split_with_sizes_copy_out_contiguous_no_cast_kernel" in name
        or "chunk_cat_cuda_kernel" in name
        or ("multi_tensor_apply_kernel" in name and "Copy<c10::BFloat16, c10::BFloat16>" in name)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observations = []
    kernels = collections.defaultdict(lambda: [0, 0.0])
    for database in sorted(args.root.glob("*/rank[0-3].sqlite")):
        db = sqlite3.connect(database)
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        ranges = [
            (a, b, n or names.get(i, ""))
            for a, b, n, i in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end>start")
        ]
        updates = [(a, b, n) for a, b, n in ranges if re.fullmatch(r"attribution.update.\d+", n)]
        assert len(updates) == 2
        events = [
            (a, b, names[n]) for a, b, n in db.execute("SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
        ]
        for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
            events.extend((a, b, table) for a, b in db.execute(f"SELECT start,end FROM {table}"))
        for a, b, label in updates:
            work = [
                (max(x, a), min(y, b), name) for x, y, name in events if y > a and x < b and "nccl" not in name.lower()
            ]
            all_active = seconds([(x, y) for x, y, _ in work])
            other_active = seconds([(x, y) for x, y, name in work if not is_buffer_copy(name)])
            service = 0.0
            for x, y, name in work:
                if is_buffer_copy(name):
                    kernels[(database.parent.name, name)][0] += 1
                    kernels[(database.parent.name, name)][1] += (y - x) / 1e9
                    service += (y - x) / 1e9
            observations.append(
                {
                    "backend": database.parent.name,
                    "rank": int(database.stem[-1]),
                    "update": label,
                    "all_non_nccl_active_seconds": all_active,
                    "other_non_nccl_active_seconds": other_active,
                    "buffer_copy_only_seconds": all_active - other_active,
                    "buffer_copy_service_seconds": service,
                }
            )
    summary = {}
    for backend in sorted({r["backend"] for r in observations}):
        rows = [r for r in observations if r["backend"] == backend]
        summary[backend] = {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k.endswith("_seconds")}
    args.output.write_text(
        json.dumps(
            {
                "observations": observations,
                "summary": summary,
                "kernels": [
                    {"backend": b, "name": n, "total_calls": count, "total_service_seconds": service}
                    for (b, n), (count, service) in kernels.items()
                ],
                "note": "Non-NCCL occupancy exclusively covered by these copy kernels; may overlap NCCL. Observational decomposition, not an isolated wall-time effect.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
