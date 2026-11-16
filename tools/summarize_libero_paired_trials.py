"""Compare completed paired LIBERO trials, including time-aligned trajectories."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_trials(root):
    trials = {}
    for path in sorted(root.glob("shard*/trial_*.json")):
        row = json.loads(path.read_text())
        if row["trial"] in trials:
            raise ValueError(f"Duplicate trial: {path}")
        trials[row["trial"]] = (row, path)
    return trials


def gif_frames(path):
    with Image.open(path) as gif:
        return [
            np.asarray(frame.convert("RGB")).copy()
            for frame in ImageSequence.Iterator(gif)
            for _ in range(max(1, frame.info.get("duration", 50) // 50))
        ]


def render_trial(a, b, paths, output, checkpoint_step=700, labels=None, candidate_only=False):
    trial = a["trial"]
    observations = [a["observations"], b["observations"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = labels or [f"2 towers / step{checkpoint_step}", f"3 towers / step{checkpoint_step}"]
    rows = [a, b]
    if candidate_only:
        rows, observations, labels, paths = rows[-1:], observations[-1:], labels[-1:], paths[-1:]
    for row, obs, label in zip(rows, observations, labels):
        xyz = np.asarray([o["robot0_eef_pos"] for o in obs])
        seconds = np.arange(len(xyz)) / 20
        for d, ax in enumerate(axes.flat[:3]):
            ax.plot(seconds, xyz[:, d], label=label)
            ax.set(title=f"End-effector {'xyz'[d]} (m)", xlabel="Time (s)")
        action = np.asarray(row["actions"])
        axes.flat[3].plot(np.arange(len(action)) / 20, action[:, -1], label=label)
        axes.flat[3].set(title="Executed gripper command", xlabel="Time (s)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    episode = a.get("dataset_episode")
    title = f"Dataset Episode {episode}" if episode is not None else f"Initial state {trial}"
    fig.suptitle(
        f"{title}: actual rollout" + ("" if candidate_only else "; trajectories compared at equal environment time")
    )
    fig.tight_layout()
    fig.savefig(output / f"trial_{trial:03d}_trajectory.png", dpi=130)
    plt.close(fig)
    videos = [gif_frames(p.with_suffix(".gif")) for p in paths]
    with imageio.get_writer(
        output / f"trial_{trial:03d}.mp4",
        fps=20,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-threads", "2"],
    ) as writer:
        for t in range(max(map(len, videos))):
            panels = []
            for video, label in zip(videos, labels):
                frame = video[min(t, len(video) - 1)]
                canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 28), "#101722")
                canvas.paste(Image.fromarray(frame), (0, 28))
                ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
                panels.append(np.asarray(canvas))
            frame = np.concatenate(panels, axis=1)
            writer.append_data(frame)
            if t == 0:
                Image.fromarray(frame).save(output / f"trial_{trial:03d}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint-step", type=int, default=700)
    parser.add_argument("--baseline-label")
    parser.add_argument("--candidate-label")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [read_trials(args.baseline), read_trials(args.candidate)]
    paired = sorted(groups[0].keys() & groups[1].keys())
    rows, sections = [], []
    for trial in paired:
        (a, pa), (b, pb) = [group[trial] for group in groups]
        for key in ["guidance", "num_steps", "fps", "seed", "action_chunk_size", "raw_action_dim", "action_stats_path"]:
            assert a["server_info"][key] == b["server_info"][key], key
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert a.get("protocol") == b.get("protocol")
        assert a.get("dataset_episode") == b.get("dataset_episode")
        row = dict(
            trial=trial,
            dataset_episode=a.get("dataset_episode"),
            two_success=a["success"],
            three_success=b["success"],
            two_steps=a["steps"],
            three_steps=b["steps"],
            errors=[a["error"], b["error"]],
        )
        if not any(row["errors"]):
            assert a["requests"][0]["observation_sha256"] == b["requests"][0]["observation_sha256"], (
                "Initial policy images differ"
            )
            np.testing.assert_array_equal(a["observations"][0]["qpos"], b["observations"][0]["qpos"])
            count = min(len(a["observations"]), len(b["observations"]))
            xyz = [np.asarray([o["robot0_eef_pos"] for o in r["observations"][:count]]) for r in [a, b]]
            distance = np.linalg.norm(xyz[0] - xyz[1], axis=-1)
            row.update(
                identical_initial_policy_image=True,
                overlap_environment_steps=count - 1,
                eef_distance_mean_m=float(distance.mean()),
                eef_distance_max_m=float(distance.max()),
            )
            # Do not pad a successful short trajectory for numerical metrics.
            # World-frame object positions are also retained in the raw records.
            objects = {}
            for key in a["observations"][0]:
                if key.endswith("_pos") and not key.startswith("robot") and "_to_" not in key:
                    positions = [np.asarray([o[key] for o in r["observations"][:count]]) for r in [a, b]]
                    objects[key] = float(np.linalg.norm(positions[0] - positions[1], axis=-1).mean())
            row["object_position_mean_distance_m"] = objects
            if args.render and not (args.output / f"trial_{trial:03d}.mp4").exists():
                render_trial(
                    a,
                    b,
                    [pa, pb],
                    args.output,
                    args.checkpoint_step,
                    labels=[
                        args.baseline_label or f"2 towers / step{args.checkpoint_step}",
                        args.candidate_label or f"3 towers / step{args.checkpoint_step}",
                    ],
                )
            if (args.output / f"trial_{trial:03d}.mp4").exists():
                title = (
                    f"Dataset Episode {a['dataset_episode']} / trial {trial}"
                    if a.get("dataset_episode") is not None
                    else f"Initial state {trial}"
                )
                sections.append(
                    f'<section><h2>{title}</h2><p>Success: two={a["success"]}, three={b["success"]}; steps: {a["steps"]} / {b["steps"]}. Mean EEF separation {distance.mean():.3f} m over shared duration.</p><video controls preload="none" poster="trial_{trial:03d}.jpg" src="trial_{trial:03d}.mp4"></video><details><summary>Trajectory</summary><img src="trial_{trial:03d}_trajectory.png"></details></section>'
                )
        rows.append(row)
    mapped = bool(rows) and all(r["dataset_episode"] is not None for r in rows)
    limitation = (
        "Five held-out dataset episodes matched to original demonstrations by exact full action sequences. "
        "Simulator resets use original states[0]; offline videos use dataset observations, whose original collection "
        "has a before-action/after-action offset. Checkpoint comparisons use identical inputs within each setting. "
        "Single-task small-sample diagnostic, not evidence of general architecture superiority."
        if mapped
        else "Official initial states are not mapped to held-out dataset episodes. A small paired diagnostic, not an equivalence test; "
        "post-action observations diverge. Distances use common duration only."
    )
    report = dict(
        checkpoint_step=args.checkpoint_step,
        expected_trials=args.expected_trials,
        completed=[len(g) for g in groups],
        paired_trials=len(rows),
        successes=[sum(r[k] for r in rows) for k in ["two_success", "three_success"]],
        pairs=rows,
        limitation=limitation,
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    page = f'<!doctype html><meta charset="utf-8"><title>Step{args.checkpoint_step} paired rollouts</title><style>body{{font:16px system-ui;max-width:1050px;margin:32px auto;background:#f3f5f8}}section{{background:white;padding:24px;margin:24px 0}}video,img{{width:100%}}</style><h1>LIBERO step{args.checkpoint_step} · paired actual rollouts</h1><p>Completed pairs: {len(rows)} / {args.expected_trials}. Successes (paired only): {report["successes"]}. Same initial states and first policy images; no robot-state policy input. CPU simulation + GPU policy inference. 20 Hz, execute 8 of 16 actions then reobserve. Videos freeze the final frame when one rollout ends earlier.</p><p>{report["limitation"]}</p>'
    (args.output / "index.html").write_text(page + "\n".join(sections))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
