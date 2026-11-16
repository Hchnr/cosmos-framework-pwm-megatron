"""Freeze local-only DROID/LIBERO catalogs and episode splits for v0.2.1.

Creates small metadata artifacts only; never copies or downloads source media.
DROID uses the existing loader's seed42 / 3% episode split.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(root):
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    cols = ["episode_index", "length", "tasks"]
    if "episode_id" in pq.read_schema(files[0]).names:
        cols.append("episode_id")
    return sorted(pq.read_table(files, columns=cols).to_pylist(), key=lambda r: r["episode_index"])


def write_catalog(output, name, root, rows, ids, split, data):
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    payload = dict(data, root=str(root), cfg_dropout_rate=0.1 if split == "train" else 0.0)
    path = target / f"{split}.parquet"
    provenance = dict(source_path=str(root), split=split, episode_ids=ids, robot_state_input=False, version="v0.2.1")
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    id=f"{name}:{split}:v0.2.1",
                    type="action",
                    dataset=f"{name}_{split}",
                    data=json.dumps(payload),
                    provenance=json.dumps(provenance),
                )
            ]
        ),
        path,
    )
    id_set = set(ids)
    chosen = [r for r in rows if r["episode_index"] in id_set]
    chunk, stride = data["chunk_length"], data["sample_stride"]
    return dict(
        catalog=str(path),
        sha256=digest(path),
        episode_ids=ids,
        episodes=len(ids),
        frames=sum(r["length"] for r in chosen),
        windows=sum((max(0, r["length"] - chunk) + stride - 1) // stride for r in chosen),
        chunk_length=chunk,
        sample_stride=stride,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = dict(
        format="lerobot",
        use_state=False,
        mode="wam",
        resolution="256",
        append_idle_frames=False,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    result = {}
    root = Path("/share/project/eai_pwm/datasets/Cosmos3-DROID/success")
    rows = metadata(root)
    assert [r["episode_index"] for r in rows] == list(range(len(rows)))
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42)).tolist()
    n = round(len(rows) * 0.03)
    droid = dict(
        common,
        adapter="droid",
        fps=15,
        chunk_length=32,
        action_space="joint_pos",
        action_normalization=None,
        viewpoint="concat_view",
        use_success_only=True,
        use_filter_dict=False,
        use_image_augmentation=False,
        sample_stride=16,
    )
    result["droid_success"] = {
        split: write_catalog(args.output, "droid_success", root, rows, sorted(ids), split, dict(droid, split=split))
        for split, ids in [("train", order[n:]), ("val", order[:n])]
    }
    # Round-robin institutions to avoid selecting only one collection source.
    groups = defaultdict(list)
    for idx in order[:n]:
        row = rows[idx]
        groups[row["episode_id"].split("/")[0]].append(row)
    fixed = []
    while len(fixed) < 128 and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(fixed) < 128:
                row = groups[name].pop(0)
                valid = row["length"] - 32
                if valid > 0:
                    # Offsets are aligned to the catalog stride, in source frames.
                    last = (valid - 1) // 16
                    fixed.append(
                        dict(
                            episode=row["episode_index"],
                            source_episode_id=row["episode_id"],
                            task=row["tasks"],
                            offset_frames=sorted({0, (last // 2) * 16, last * 16}),
                        )
                    )
    result["droid_offline_evaluation"] = dict(
        episodes=fixed,
        policy_input="current image + text only; no state or action history",
        noise_seed=2026,
        caption="freeze one existing annotation per fixture",
        note="Selection frozen; decoded reference/policy fixtures still need generation and validation.",
    )
    for suite in ["libero_10", "libero_spatial", "libero_object", "libero_goal"]:
        root = Path("/share/project/eai_pwm/datasets/libero/原始版") / suite
        rows = metadata(root)
        groups = defaultdict(list)
        for r in rows:
            assert len(r["tasks"]) == 1
            groups[r["tasks"][0]].append(r["episode_index"])
        train, val = [], []
        for task, ids in sorted(groups.items()):
            random.Random("v0.2.1:42:" + suite + ":" + task).shuffle(ids)
            holdout = set(ids[: max(1, round(len(ids) * 0.2))])
            # Preserve the first pilot's held-out bowl episodes for regression.
            if suite == "libero_goal" and task == "put the bowl on the plate":
                previous = Path(
                    "/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_goal_bowl_plate_v1/manifest.json"
                )
                if not previous.is_file():
                    raise FileNotFoundError(previous)
                holdout = set(json.loads(previous.read_text())["splits"]["val"]["episode_ids"])
                assert holdout <= set(ids)
            val.extend(holdout)
            train.extend(set(ids) - holdout)
        data = dict(
            common,
            adapter="libero",
            split="full",
            fps=20,
            chunk_length=16,
            camera_mode="concat_view",
            action_space="frame_wise_relative",
            rotation_space="6d",
            pose_coordinate_frame="native",
            action_normalization="quantile_rot",
            sample_stride=1,
        )
        result[suite] = {
            split: write_catalog(
                args.output, suite, root, rows, sorted(ids), split, dict(data, episode_ids=sorted(ids))
            )
            for split, ids in [("train", train), ("val", val)]
        }
    for name, report in result.items():
        if "train" not in report:
            continue
        assert set(report["train"]["episode_ids"]).isdisjoint(report["val"]["episode_ids"])
        print(
            name,
            {
                s: {k: v for k, v in r.items() if k not in ["episode_ids", "catalog", "sha256"]}
                for s, r in report.items()
            },
        )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
