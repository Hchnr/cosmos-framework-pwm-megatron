"""Audit actual inputs across four evaluations, render results and build the portable report."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/libero_bowl_plate"))
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    fixture = args.fixture_root.resolve()
    # Resolving the python symlink bypasses the virtual environment's packages.
    python = str(Path.cwd() / ".venv/bin/python")
    episodes = json.loads((fixture / "episodes.json").read_text())["episodes"]
    variants = [(s, t) for s in [0, 1000] for t in ["two", "three"]]
    manifest_path = root / "eval5_launch_manifest.json"
    while True:
        manifest = json.loads(manifest_path.read_text())
        capture_jobs = [j for j in manifest["jobs"] if j["kind"] == "capture"]
        if len(capture_jobs) == 4 and all("Saved " in Path(j["log"]).read_text() for j in capture_jobs):
            break
        if not args.wait:
            raise ValueError("Offline captures are not complete")
        time.sleep(10)
    capture_inputs = {}
    for step, tower in variants:
        path = root / f"step{step}/{tower}_tower.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert len(payload["captures"]) == 5
        capture_inputs[f"step{step}_{tower}"] = [fingerprint(c["initial"]) for c in payload["captures"]]
        assert all(c["initial"][0]["state_indexes"].numel() == 0 for c in payload["captures"])
        del payload
    baseline = capture_inputs["step0_two"]
    assert all(value == baseline for value in capture_inputs.values()), "Actual offline model inputs/noise differ"
    print("Actual offline inputs and noise are bitwise identical across all four conditions.", flush=True)
    audit = dict(
        passed=False,
        offline_inputs_identical=True,
        offline_actual_inputs=baseline,
        fixture_sha256=hashlib.sha256((fixture / "eval_batches.pt").read_bytes()).hexdigest(),
        reset_manifest_sha256=hashlib.sha256((fixture / "episodes.json").read_bytes()).hexdigest(),
        episodes=[
            {k: row[k] for k in ["trial", "episode", "demo", "window_offset", "action_match", "initial_state_sha256"]}
            for row in episodes
        ],
    )
    (root / "eval5_offline_alignment.json").write_text(json.dumps(audit, indent=2) + "\n")

    def run(command, log):
        with log.open("w") as f:
            subprocess.run(
                [python, *command],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "LD_LIBRARY_PATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
                check=True,
            )

    def render_offline(step):
        if (root / f"step{step}/visuals/metrics.json").exists():
            return
        run(
            [
                "tools/render_action_tower_comparison.py",
                "--baseline",
                str(root / f"step{step}/two_tower.pt"),
                "--candidate",
                str(root / f"step{step}/three_tower.pt"),
                "--fixture",
                str(fixture / "eval_batches.pt"),
                "--output",
                str(root / f"step{step}/visuals"),
                "--baseline-label",
                f"2 towers / step{step}",
                "--candidate-label",
                f"3 towers / step{step}",
            ],
            root / f"step{step}/render.log",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render_offline, [0, 1000]))
    print("Both five-window offline reports rendered.", flush=True)
    while not json.loads(manifest_path.read_text()).get("all_complete"):
        if not args.wait:
            raise ValueError("Closed-loop evaluation has not completed")
        pid = int((root / "eval5_coordinator.pid").read_text())
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError("Coordinator exited before completing the evaluations; inspect its log")
        time.sleep(15)
    rows = {}
    for step, tower in variants:
        files = sorted((root / f"step{step}/closed_loop_{tower}_tower").glob("shard*/trial_*.json"))
        records = [json.loads(p.read_text()) for p in files]
        assert len(records) == 5 and {r["trial"] for r in records} == set(range(5))
        rows[step, tower] = {r["trial"]: r for r in records}
    for ep in audit["episodes"]:
        trial = ep["trial"]
        records = [rows[v][trial] for v in variants]
        ref = records[0]
        for row in records:
            assert row["error"] is None
            assert row["dataset_episode"] == ep["episode"]
            assert row["initial_state_sha256"] == ep["initial_state_sha256"]
            assert row["protocol"] == ref["protocol"]
            assert row["seed"] == ref["seed"]
            assert row["requests"][0]["observation_sha256"] == ref["requests"][0]["observation_sha256"]
            np.testing.assert_array_equal(row["observations"][0]["qpos"], ref["observations"][0]["qpos"])
            assert row["steps"] == len(row["actions"]) <= 1000
            assert row["success"] or row["steps"] == 1000, "Premature unsuccessful rollout"
            for key in [
                "guidance",
                "num_steps",
                "fps",
                "seed",
                "action_chunk_size",
                "raw_action_dim",
                "action_stats_path",
            ]:
                assert row["server_info"][key] == ref["server_info"][key]
        ep["identical_first_policy_image"] = True
        ep["first_policy_image_sha256"] = ref["requests"][0]["observation_sha256"]
        ep["identical_initial_qpos"] = True
    audit.update(
        passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        successes={f"step{s}_{t}": sum(r["success"] for r in rows[s, t].values()) for s, t in variants},
    )
    (root / "eval5_alignment.json").write_text(json.dumps(audit, indent=2) + "\n")

    def render_closed(step):
        run(
            [
                "tools/summarize_libero_paired_trials.py",
                "--baseline",
                str(root / f"step{step}/closed_loop_two_tower"),
                "--candidate",
                str(root / f"step{step}/closed_loop_three_tower"),
                "--output",
                str(root / f"step{step}/closed_loop_visuals"),
                "--expected-trials",
                "5",
                "--checkpoint-step",
                str(step),
                "--render",
            ],
            root / f"step{step}/render_closed.log",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render_closed, [0, 1000]))
    run(
        [
            "tools/render_libero_eval_overview.py",
            "--source",
            str(root),
            "--output",
            str(root / "libero_bowl_plate_eval"),
        ],
        root / "eval5_render_overview.log",
    )
    (root / "eval5_complete.json").write_text(
        json.dumps(
            dict(time_utc=datetime.now(timezone.utc).isoformat(), successes=audit["successes"], errors=0), indent=2
        )
        + "\n"
    )
    print("All results rendered; cross-checkpoint alignment passed.", audit["successes"], flush=True)


if __name__ == "__main__":
    main()
