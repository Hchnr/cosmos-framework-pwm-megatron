#!/usr/bin/env python3
"""Package individual WorldArena runs for incremental offline downloads.

Usage examples:
    # Package the displayed VLM v2 runs matching specific training steps.
    python tools/package_worldarena_checkpoints.py --steps 10000 11000 12000

    # Package one or more displayed runs by their exact directory names.
    python tools/package_worldarena_checkpoints.py --runs RUN_DIRECTORY [RUN_DIRECTORY ...]

    # Package every run referenced by the comparison viewer, using six workers.
    python tools/package_worldarena_checkpoints.py --all --workers 6

    # Override the input and transfer directories.
    python tools/package_worldarena_checkpoints.py --root RESULTS_DIR --output TRANSFER_DIR --all

The default input is ``outputs/worldarena_eval25_480p_steps50`` and the default
output is its sibling ``outputs/worldarena_eval25_480p_steps50_transfer``. After
downloading archives from the transfer directory, run ``./extract.sh`` there and
open ``index.html`` to view the extracted checkpoints.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path


def package(root: Path, output: Path, run: str) -> dict:
    directory = root / run
    samples = sorted(directory.glob("*/sample_outputs.json"))
    if len(samples) != 25 or any(json.loads(p.read_text()).get("status") != "success" for p in samples):
        raise ValueError(f"{run}: expected 25 successful samples")
    for sample in samples:
        for name in ("vision.mp4", "sample_args.json"):
            if not (sample.parent / name).is_file():
                raise ValueError(f"{sample.parent}: missing {name}")
    destination = output / f"{run}.tar.gz"
    temporary = destination.with_suffix(".gz.partial")
    # Each archive contains exactly one run; shared viewer files live beside the archives.
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            archive.add(directory, arcname=run)
        with tarfile.open(temporary) as archive:
            videos = [m for m in archive.getmembers() if m.name.endswith("/vision.mp4")]
            if len(videos) != 25 or any(not m.name.startswith(f"{run}/") for m in videos):
                raise ValueError(f"{run}: archive content verification failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"run": run, "archive": destination.name, "bytes": destination.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("outputs/worldarena_eval25_480p_steps50"))
    parser.add_argument("--output", type=Path, help="Default: sibling directory <root>_transfer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs="+", help="Exact result directory names")
    group.add_argument("--steps", nargs="+", type=int, help="VLM v2 checkpoint training iterations")
    group.add_argument("--all", action="store_true", help="Each displayed checkpoint gets its own archive")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root.with_name(root.name + "_transfer")).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        parser.error("Display and transfer directories must be separate, non-overlapping directories")
    # Only package runs actually present in the comparison viewer, excluding obsolete duplicates.
    page = (root / "index.html").read_text()
    viewer_data = re.search(r'<script id="viewerData" type="application/json">(.*?)</script>', page, re.S)
    if viewer_data:
        displayed = [run["directory"] for run in json.loads(viewer_data[1])["runs"]]
    else:
        # Older viewers listed each video source directly in JavaScript.
        displayed = list(dict.fromkeys(re.findall(r"`([^/`]+)/\$\{name\}/vision.mp4`", page)))
    if args.steps:
        runs = []
        for step in args.steps:
            matches = [run for run in displayed if run.startswith(f"vlm_v2_t2v_iter{step}_")]
            if len(matches) != 1:
                parser.error(f"step {step}: expected one displayed VLM v2 run, found {matches}; use --runs")
            runs.extend(matches)
    else:
        runs = displayed if args.all else args.runs
    runs = list(dict.fromkeys(runs))
    if not runs or any(run not in displayed for run in runs):
        parser.error("Choose result directories present in index.html")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    # Copy all top-level support files, including the standalone viewer and audit records.
    for source in root.iterdir():
        if source.is_file() and not source.name.endswith((".tar", ".gz", ".partial")):
            shutil.copy2(source, output / source.name)
    (output / "extract.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\n'
        "if (( $# == 0 )); then\n  shopt -s nullglob\n  set -- ./*.tar.gz\nfi\n"
        'if (( $# == 0 )); then echo "No checkpoint archives found" >&2; exit 1; fi\n'
        'for archive in "$@"; do tar -xzf "$archive"; done\n'
        'echo "Open index.html in this directory and select the downloaded checkpoints."\n'
    )
    (output / "extract.sh").chmod(0o755)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda run: package(root, output, run), runs))
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    merged = {item["run"]: item for item in previous}
    merged.update({item["run"]: item for item in records})
    records = list(merged.values())
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(f"{item['sha256']}  {item['archive']}\n" for item in records))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
