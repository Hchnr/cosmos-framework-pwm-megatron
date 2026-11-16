"""Render one action experiment against expert data, without a model baseline."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image
from render_action_tower_comparison import frames, labelled
from summarize_libero_paired_trials import read_trials, render_trial

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--closed-loop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#242b37",
            "axes.facecolor": "#242b37",
            "savefig.facecolor": "#242b37",
            "text.color": "#d8dfe8",
            "axes.labelcolor": "#b6c3d4",
            "xtick.color": "#b6c3d4",
            "ytick.color": "#b6c3d4",
            "axes.edgecolor": "#69778b",
            "grid.color": "#69778b",
            "axes.prop_cycle": plt.cycler(color=["#91abc4", "#c58b54", "#b7a0bf"]),
        }
    )
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    offline, simulation = args.output / "offline", args.output / "simulation"
    offline.mkdir(parents=True, exist_ok=True)
    simulation.mkdir(parents=True, exist_ok=True)
    count = len(payload["captures"])
    metrics = []
    for capture, reference, sample in zip(
        payload["captures"], fixture["reference_batches"][:count], fixture["samples"][:count], strict=True
    ):
        h, w = map(int, reference["image_size"][0].flatten()[-2:].tolist())
        expert = frames(reference["video"][0][0], h, w)
        generated = frames(capture["result"]["decoded_video"][0], h, w, True)
        assert expert.shape == generated.shape
        name = f"episode_{sample['episode']}_offset_{sample['window_offset']}"
        montage = [
            np.concatenate((labelled(a, "Expert reference"), labelled(b, args.label)), axis=1)
            for a, b in zip(expert, generated, strict=True)
        ]
        imageio.mimwrite(
            offline / f"{name}.mp4",
            montage,
            fps=8,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
            ffmpeg_params=["-threads", "2"],
        )
        Image.fromarray(montage[0]).save(offline / f"{name}_poster.jpg")
        truth = reference["action"][0][0].float()[:, :10]
        predicted = capture["result"]["action"][0].float().reshape(16, 10)
        assert torch.isfinite(predicted).all()
        fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True)
        channels = ["dx", "dy", "dz", *[f"rot6d[{i}]" for i in range(6)], "gripper"]
        for dim, ax in enumerate(axes.flat):
            ax.plot(truth[:, dim], label="Expert reference", color="#aeb8c7")
            ax.plot(predicted[:, dim], label=args.label, color="#c58b54")
            ax.set_title(channels[dim])
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Episode {sample['episode']} · normalized action channels")
        fig.supxlabel("Action step (20 Hz)")
        fig.tight_layout()
        fig.savefig(offline / f"{name}_actions.png", dpi=130)
        plt.close(fig)
        metrics.append(
            dict(
                sample=sample,
                candidate=dict(
                    normalized_action_mse=float((predicted - truth).square().mean()),
                    future_video_mae_0_255=float(np.abs(generated[1:].astype(float) - expert[1:].astype(float)).mean()),
                ),
            )
        )
    (offline / "metrics.json").write_text(json.dumps(dict(samples=metrics), indent=2) + "\n")
    trials = read_trials(args.closed_loop)
    assert set(trials) == set(range(count))
    rows = []
    for trial, (row, path) in sorted(trials.items()):
        render_trial(
            row, row, [path, path], simulation, args.step, labels=[args.label, args.label], candidate_only=True
        )
        rows.append(dict(trial=trial, episode=row["dataset_episode"], success=row["success"], steps=row["steps"]))
    (simulation / "metrics.json").write_text(json.dumps(dict(trials=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
