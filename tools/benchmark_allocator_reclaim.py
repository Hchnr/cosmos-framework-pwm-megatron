"""Isolate cross-stream cache reclaim cost; this is not a training benchmark."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def counters():
    stats = torch.cuda.memory_stats()
    return {
        k: stats[k]
        for k in ("num_alloc_retries", "num_ooms", "allocated_bytes.all.current", "reserved_bytes.all.current")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-gib", type=int, nargs="+", default=[10, 20, 35])
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    default = torch.cuda.default_stream()
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    gib = 1024**3
    # Leave 2 GiB physical headroom, then request 4 GiB on another stream.
    # That forces reclamation despite a large idle cache on the default stream.
    budget = ((free - 2 * gib) // (20 * 1024**2)) * (20 * 1024**2)
    requested = 4 * gib
    rows = []
    torch.cuda.cudart().cudaProfilerStart()
    for cache_gib in args.cache_gib:
        for same_stream in (False, True):
            for repeat in range(args.repeats):
                torch.cuda.empty_cache()
                label = f"reclaim.cache_{cache_gib}GiB.{'same' if same_stream else 'other'}_stream.repeat_{repeat}"
                with torch.cuda.nvtx.range(label):
                    with torch.cuda.nvtx.range("setup_active_and_cached_memory"):
                        active = torch.empty(budget - cache_gib * gib, dtype=torch.uint8, device="cuda")
                        cached = torch.empty(cache_gib * gib, dtype=torch.uint8, device="cuda")
                        del cached
                        torch.cuda.synchronize()
                    before = counters()
                    physical_free_before = torch.cuda.mem_get_info()[0]
                    target = default if same_stream else side
                    with torch.cuda.stream(target), torch.cuda.nvtx.range("timed_4GiB_request"):
                        started = time.perf_counter_ns()
                        output = torch.empty(requested, dtype=torch.uint8, device="cuda")
                        finished = time.perf_counter_ns()
                    torch.cuda.synchronize()
                    after = counters()
                    retry_delta = after["num_alloc_retries"] - before["num_alloc_retries"]
                    assert retry_delta == (0 if same_stream else 1), (label, before, after)
                    assert after["num_ooms"] == before["num_ooms"], (label, before, after)
                    rows.append(
                        dict(
                            cache_gib=cache_gib,
                            same_stream=same_stream,
                            repeat=repeat,
                            request_seconds=(finished - started) / 1e9,
                            physical_free_before=physical_free_before,
                            before=before,
                            after=after,
                            retry_delta=retry_delta,
                        )
                    )
                    del output, active
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
    torch.cuda.cudart().cudaProfilerStop()
    summary = [
        dict(
            cache_gib=cache,
            same_stream=same,
            median_request_seconds=statistics.median(
                r["request_seconds"] for r in rows if r["cache_gib"] == cache and r["same_stream"] == same
            ),
        )
        for cache in args.cache_gib
        for same in (False, True)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                scope="synthetic allocation-only mechanism test; no training throughput claim",
                device=args.device,
                total_bytes=total,
                budget_bytes=budget,
                allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                requested_bytes=requested,
                rows=rows,
                summary=summary,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
