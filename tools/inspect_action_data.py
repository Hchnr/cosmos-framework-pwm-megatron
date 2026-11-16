#!/usr/bin/env python3
"""Inventory local LeRobot v3 data and count windows using the loader's splits.

Checks referenced files and Parquet footer row counts, not complete video decoding.
No data files or existing catalogs are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import split_episode_ids
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset


def inspect_dataset(label, adapter, root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = []
    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        columns = [
            key
            for key in pq.read_schema(path).names
            if key in ("episode_index", "length", "data/chunk_index", "data/file_index")
            or (key.startswith("videos/") and key.endswith(("/chunk_index", "/file_index")))
        ]
        episodes.extend(pq.read_table(path, columns=columns).to_pylist())
    episodes.sort(key=lambda row: row["episode_index"])
    expected = set()
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for row in episodes:
        expected.add(info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
        for camera in cameras:
            expected.add(
                info["video_path"].format(
                    video_key=camera,
                    chunk_index=row[f"videos/{camera}/chunk_index"],
                    file_index=row[f"videos/{camera}/file_index"],
                )
            )
    missing = sorted(path for path in expected if not (root / path).is_file())
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    frame_rows = sum(pq.read_metadata(path).num_rows for path in data_paths)
    counts = {}
    ids = [row["episode_index"] for row in episodes]
    for split in ("train", "val"):
        if adapter == "droid":
            assert ids == list(range(len(ids))), "DROID loader requires contiguous episode indices"
            selected = set(split_episode_ids(len(ids), seed=42, val_ratio=0.03, split=split))
        elif adapter == "libero":
            selected = LIBEROLeRobotDataset._split_episode_ids(ids, split, val_ratio=0.01, seed=0)
        else:
            raise ValueError(adapter)
        lengths = [row["length"] for row in episodes if row["episode_index"] in selected]
        counts[split] = {
            "episodes": len(lengths),
            "frames": sum(lengths),
            "windows_stride1_no_filter": {
                str(chunk): sum(max(0, length - chunk) for length in lengths) for chunk in (4, 16, 32)
            },
        }
    return {
        "label": label,
        "adapter": adapter,
        "root": str(root),
        "fps": info["fps"],
        "info_episodes": info["total_episodes"],
        "info_frames": info["total_frames"],
        "info_tasks": info["total_tasks"],
        "metadata_episodes": len(episodes),
        "metadata_frames": sum(row["length"] for row in episodes),
        "data_parquet_files": len(data_paths),
        "data_parquet_rows": frame_rows,
        "video_files": len(list((root / "videos").glob("**/*.mp4"))),
        "referenced_files": len(expected),
        "missing_referenced_files": missing,
        "camera_shapes": {key: info["features"][key]["shape"] for key in cameras},
        "split": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs=3, action="append", required=True, metavar=("LABEL", "ADAPTER", "ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for entry in args.dataset:
        result = inspect_dataset(*entry)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
