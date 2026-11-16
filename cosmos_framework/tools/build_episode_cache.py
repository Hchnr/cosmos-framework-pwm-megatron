"""Build shared Action128 episode caches without starting a model or training.

Run with the same shared cache directory exported by the production launcher.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import tomllib

from cosmos_framework.data.generator.action.episode_metadata_cache import load_episode_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    recipe = tomllib.loads(args.recipe.read_text())
    roots = []
    for entry in recipe["dataloader_train"]["datasets"]:
        for value in pq.read_table(entry["path"], columns=["data"])["data"].to_pylist():
            config = json.loads(value)
            if config.get("adapter") == "unified_action128" and config["root"] not in roots:
                roots.append(config["root"])

    def build(root):
        start = time.monotonic()
        result = load_episode_metadata(root, cache_dir=args.cache_dir)
        record = {
            "root": root,
            "hit": result.cache_hit,
            "seconds": round(time.monotonic() - start, 3),
            "source_files": len(result.metadata),
            "episodes": len(result.episodes),
            "cache_path": str(result.cache_path),
            "cache_bytes": result.cache_path.stat().st_size,
        }
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build, root) for root in roots]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
