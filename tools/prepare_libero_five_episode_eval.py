"""Match five held-out LeRobot episodes to original LIBERO demonstration states."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [43, 69, 72, 80, 132]
    data = pq.read_table(args.dataset / "data/chunk-000/file-000.parquet").to_pandas()
    fixture = torch.load(args.source_fixture, map_location="cpu", weights_only=False)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = dict(batches=[], reference_batches=[], samples=[])
    rows = []
    with h5py.File(args.demonstrations) as source:
        for trial, episode in enumerate(episodes):
            actions = np.stack(data[data.episode_index == episode].action)
            matches = []
            for name, demo in source["data"].items():
                raw = demo["actions"][:].astype(np.float32)
                raw[:, -1] = (1 - raw[:, -1]) / 2
                if raw.shape == actions.shape and np.array_equal(raw, actions):
                    matches.append(name)
            if len(matches) != 1:
                raise ValueError(f"Episode {episode}: expected unique exact full-action match, got {matches}")
            demo = source["data"][matches[0]]
            state = demo["states"][0].astype(np.float64)
            # Simulator state is metadata for resetting the environment, never a policy input.
            rows.append(
                dict(
                    trial=trial,
                    episode=episode,
                    window_offset=0,
                    demo=matches[0],
                    action_rows=len(actions),
                    action_match="bitwise float32, entire episode",
                    initial_state=state.tolist(),
                    initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    model_xml=str(demo.attrs["model_file"]),
                    reference_eef_pos=demo["obs/ee_pos"][0].tolist(),
                )
            )
            index = next(
                i for i, row in enumerate(fixture["samples"]) if row["episode"] == episode and row["window_offset"] == 0
            )
            for key in selected:
                selected[key].append(copy.deepcopy(fixture[key][index]))
            for plan in selected["batches"][-1]["sequence_plan"]:
                assert not plan.robot_state_frame_indexes and not plan.condition_frame_indexes_action
    manifest = dict(
        task="put the bowl on the plate",
        episodes=rows,
        max_steps=1000,
        warmup_steps=0,
        seed=42,
        window_selection="first window of each of the first five sorted held-out episodes",
        source_hdf5=str(args.demonstrations),
        source_hdf5_sha256=hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
        robot_state_policy_input=False,
    )
    (args.output / "episodes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected["manifest"] = {k: v for k, v in manifest.items() if k != "episodes"}
    torch.save(selected, args.output / "eval_batches.pt")
    print(
        json.dumps(
            [{k: r[k] for k in ["trial", "episode", "demo", "action_rows", "action_match"]} for r in rows], indent=2
        )
    )


if __name__ == "__main__":
    main()
