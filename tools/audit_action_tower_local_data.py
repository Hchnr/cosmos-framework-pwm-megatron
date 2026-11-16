"""Audit existing LeRobot metadata and referenced files without downloading data."""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq


def audit(root):
    info = json.loads((root / "meta/info.json").read_text())
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    names = pq.read_schema(files[0]).names
    columns = [n for n in names if not n.startswith("stats/")]
    rows = pq.read_table(files, columns=columns).to_pylist()
    features = {k: v for k, v in info["features"].items() if v["dtype"] == "video"}
    refs = set()
    tasks = Counter()
    for row in rows:
        tasks.update(row["tasks"])
        refs.add(info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
        for key in features:
            refs.add(
                info["video_path"].format(
                    video_key=key,
                    chunk_index=row[f"videos/{key}/chunk_index"],
                    file_index=row[f"videos/{key}/file_index"],
                )
            )

    def check(name):
        path = root / name
        return name if not path.is_file() or path.stat().st_size == 0 else None

    with ThreadPoolExecutor(max_workers=16) as pool:
        missing = [p for p in pool.map(check, sorted(refs)) if p]
    lengths = sorted(r["length"] for r in rows)
    scene_coverage = None
    task_map = Path(
        "/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
    )
    if root.name.startswith("libero_") and task_map.is_file():
        tree = ast.parse(task_map.read_text())
        mapping = ast.literal_eval(
            next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets)
            )
        )
        scene_names = mapping[root.name]
        language_scenes = {}
        for name in scene_names:
            language = re.sub(r"^.*?SCENE\d+_", "", name).replace("_", " ")
            language_scenes.setdefault(language, []).append(name)
        scene_coverage = dict(
            benchmark_tasks=len(scene_names),
            unique_benchmark_languages=len(language_scenes),
            missing_languages=sorted(set(language_scenes) - set(tasks)),
            unknown_languages=sorted(set(tasks) - set(language_scenes)),
            repeated_language_scenes={k: v for k, v in language_scenes.items() if len(v) > 1},
            note="Language coverage does not establish episode-to-scene identity or complete scene coverage.",
        )
    return dict(
        root=str(root),
        scene_coverage=scene_coverage,
        declared={k: info.get(k) for k in ["total_episodes", "total_frames", "total_tasks", "fps"]},
        episodes=len(rows),
        unique_episode_ids=len({r["episode_index"] for r in rows}),
        frames=sum(lengths),
        observed_tasks=len(tasks),
        task_episode_counts=dict(sorted(tasks.items())),
        length_min=lengths[0],
        length_median=lengths[len(lengths) // 2],
        length_max=lengths[-1],
        video_features=features,
        referenced_files=len(refs),
        missing_or_empty_files=missing,
        metadata_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root / "meta/info.json", root / "meta/tasks.parquet", *files]
        },
        audit_scope="All metadata rows and referenced-file existence/size; not full decode or numerical validation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [Path("/share/project/eai_pwm/datasets/Cosmos3-DROID") / s for s in ["success", "failure"]]
    roots += sorted(Path("/share/project/eai_pwm/datasets/libero/原始版").glob("libero_*"))
    roots += [Path("/share/project/liyue/data/Cosmos3-DROID-golden-5c11a20/success")]
    reports = []
    for root in roots:
        report = audit(root)
        reports.append(report)
        print(
            json.dumps(
                {k: report[k] for k in ["root", "episodes", "frames", "observed_tasks", "referenced_files"]}
                | {"missing_files": len(report["missing_or_empty_files"])},
                ensure_ascii=False,
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
