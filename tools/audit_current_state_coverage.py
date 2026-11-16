"""Deterministic window-uniform audit of State availability; no media/model IO."""

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tomllib


def audit(spec, samples):
    row = pq.read_table(spec["path"]).to_pylist()[0]
    data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
    root = Path(data["root"])
    columns = [
        "episode_index",
        "duration_seconds",
        "fps",
        "data_path",
        "dataset_from_index",
        "dataset_to_index",
        "data_role",
        "training_eligible",
    ]
    ep = pa.concat_tables(
        [pq.read_table(p, columns=columns) for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))]
    )
    ep = ep.filter(pc.and_(pc.equal(ep["data_role"], "training"), ep["training_eligible"]))
    stride = float(data.get("window_stride_seconds", 2.0))
    counts = np.ceil(ep["duration_seconds"].to_numpy() / stride - 1e-9).astype(np.int64)
    ends = np.cumsum(counts)
    total = int(ends[-1])
    rng = random.Random(int.from_bytes(hashlib.blake2b(spec["path"].encode(), digest_size=8).digest(), "big"))
    indexes = sorted(rng.sample(range(total), min(samples, total)))
    result = dict(
        path=spec["path"],
        ratio=spec["ratio"],
        total_windows=total,
        sampled=len(indexes),
        missing=0,
        first_window_sampled=0,
        first_window_missing=0,
        entirely_masked_episode_samples=0,
    )
    examples = []
    for flat in indexes:
        e = int(np.searchsorted(ends, flat, side="right"))
        item = ep.slice(e, 1).to_pylist()[0]
        window = flat - (int(ends[e - 1]) if e else 0)
        start = window * stride
        parts = []
        with pq.ParquetFile(root / item["data_path"]) as f:
            col = f.schema_arrow.get_field_index("index")
            lo, hi = item["dataset_from_index"], item["dataset_to_index"]
            for g in range(f.num_row_groups):
                stats = f.metadata.row_group(g).column(col).statistics
                if stats and stats.has_min_max and (stats.max < lo or stats.min >= hi):
                    continue
                t = f.read_row_group(g, columns=["index", "timestamp", "observation.state_mask"])
                t = t.filter(pc.and_(pc.greater_equal(t["index"], lo), pc.less(t["index"], hi)))
                if len(t):
                    parts.append(t)
        if not parts:
            raise ValueError(f"No numerical rows: {root}, episode={item['episode_index']}")
        t = pa.concat_tables(parts)
        times = t["timestamp"].to_numpy()
        masks = np.asarray(t["observation.state_mask"].to_pylist(), dtype=bool).any(axis=1)
        if not masks.any():
            result["entirely_masked_episode_samples"] += 1
        if window == 0:
            result["first_window_sampled"] += 1
        tolerance = max(1e-8, abs(start) * np.finfo(np.float32).eps)
        eligible = np.flatnonzero(times <= start + tolerance)
        missing = not len(eligible) or not masks[eligible[-1]]
        if missing:
            result["missing"] += 1
            result["first_window_missing"] += int(window == 0)
            if len(examples) < 5:
                examples.append(dict(episode_index=item["episode_index"], window_index=window, start=start))
    result["examples"] = examples
    result["missing_fraction"] = result["missing"] / result["sampled"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = tomllib.loads(Path(args.config).read_text())["dataloader_train"]["datasets"]
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda s: audit(s, args.samples), specs):
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("TOTAL", sum(r["missing"] for r in results), "/", sum(r["sampled"] for r in results), flush=True)


if __name__ == "__main__":
    main()
