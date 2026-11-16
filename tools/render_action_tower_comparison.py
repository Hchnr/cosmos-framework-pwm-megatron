#!/usr/bin/env python3
"""Render actual decoded video and action predictions against held-out expert data.

Consumes trusted local captures from validate_action_tower.py. No model runs or
weight changes here. Metrics are descriptive, not closed-loop success rates.
"""

import argparse
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageSequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frames(tensor, height, width, normalized=False):
    value = tensor.detach().float().squeeze(0)
    if normalized:
        value = (value + 1) * 127.5
    return value[:, :, :height, :width].clamp(0, 255).byte().permute(1, 2, 3, 0).numpy()


def labelled(frame, label):
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 28), "#101722")
    canvas.paste(Image.fromarray(frame), (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
    return np.asarray(canvas)


def render_closed_loop(baseline_dir, candidate_dir, output):
    """Append one matched simulator trial; keep it distinct from WAM forecasts."""
    summaries = [json.loads((p / "summary.json").read_text()) for p in [baseline_dir, candidate_dir]]
    for key in ["task_suite", "selected_task_ids", "action_space", "rotation_space", "action_dim"]:
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"Closed-loop setting mismatch: {key}")
    videos, episodes, durations = [], [], []
    for directory, summary in zip([baseline_dir, candidate_dir], summaries):
        if summary["total_episodes"] != 1 or len(summary["task_results"]) != 1:
            raise ValueError("Closed-loop visualization expects exactly one trial per model")
        task = summary["task_results"][0]
        episode = task["episode_results"][0]
        if episode["error"]:
            raise ValueError(f"Incomplete closed-loop evaluation: {episode['error']}")
        path = directory / "gifs" / f"task_{task['task_id']:03d}" / f"episode_{episode['episode']:03d}.gif"
        with Image.open(path) as source:
            frames_and_times = [
                (np.asarray(f.convert("RGB")), f.info["duration"]) for f in ImageSequence.Iterator(source)
            ]
        videos.append([f for f, _ in frames_and_times])
        durations.append([dt for _, dt in frames_and_times])
        episodes.append(episode)
    # GIF may merge repeated frames. Expand their recorded durations so real
    # simulator time remains aligned, including the warm-up steps.
    tick_ms = min(dt for times in durations for dt in times)
    expanded = []
    for video, times in zip(videos, durations):
        if any(dt % tick_ms for dt in times):
            raise ValueError("GIF frame durations must share an integer time grid")
        expanded.append([frame for frame, dt in zip(video, times) for _ in range(dt // tick_ms)])
    montage = [
        np.concatenate(
            [
                labelled(v[min(t, len(v) - 1)], label)
                for v, label in zip(expanded, ["Nano / actual rollout", "3 towers / actual rollout"])
            ],
            axis=1,
        )
        for t in range(max(map(len, expanded)))
    ]
    imageio.mimwrite(
        output / "closed_loop.mp4",
        montage,
        fps=1000 / tick_ms,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    Image.fromarray(np.concatenate([montage[t] for t in np.linspace(0, len(montage) - 1, 5, dtype=int)], axis=0)).save(
        output / "closed_loop.jpg"
    )
    report = dict(
        baseline=episodes[0],
        candidate=episodes[1],
        source_directories=[str(baseline_dir), str(candidate_dir)],
        limitation="One initial state per model; diagnostic rollout, not a reliable success-rate estimate.",
    )
    section = f"""<section><h2>闭环仿真：实际执行录像</h2>
<p>下方是模型动作驱动 MuJoCo 后得到的环境画面，不是生成视频。左：原始 Nano；右：未训练三塔。</p>
<video controls muted playsinline src="closed_loop.mp4" poster="closed_loop.jpg"></video>
<p>Nano: success={episodes[0]["success"]}, {episodes[0]["steps"]} steps；三塔: success={episodes[1]["success"]}, {episodes[1]["steps"]} steps。</p>
<p>每组仅一个初始状态，仅检查执行链路与直观行为，不能估计可靠成功率。20 Hz，预测16步、执行前8步后重新观测；若一组提前结束则定格末帧。</p></section>"""
    return report, section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Original Nano (2 towers)")
    parser.add_argument("--candidate-label", default="3 towers / before training")
    parser.add_argument("--samples", type=int, help="Compare only this many leading frozen fixture samples")
    parser.add_argument("--closed-loop-baseline", type=Path)
    parser.add_argument("--closed-loop-candidate", type=Path)
    args = parser.parse_args()
    if bool(args.closed_loop_baseline) != bool(args.closed_loop_candidate):
        parser.error("Both closed-loop directories must be supplied together")
    two = torch.load(args.baseline, map_location="cpu", weights_only=False)
    three = torch.load(args.candidate, map_location="cpu", weights_only=False)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    if args.samples is not None:
        assert 0 < args.samples <= min(len(two["captures"]), len(three["captures"]), len(fixture["samples"]))
        two["captures"] = two["captures"][: args.samples]
        three["captures"] = three["captures"][: args.samples]
        fixture["reference_batches"] = fixture["reference_batches"][: args.samples]
        fixture["samples"] = fixture["samples"][: args.samples]
    assert two["batch_file_sha256"] == three["batch_file_sha256"]
    assert not two["diagnostic_only"] and not three["diagnostic_only"]
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, sections = [], []
    for i, (baseline, candidate, reference, sample) in enumerate(
        zip(two["captures"], three["captures"], fixture["reference_batches"], fixture["samples"], strict=True)
    ):
        for key in baseline["initial"][0]:
            torch.testing.assert_close(baseline["initial"][0][key], candidate["initial"][0][key], atol=0, rtol=0)
        h, w = map(int, reference["image_size"][0].flatten()[-2:].tolist())
        videos = [frames(reference["video"][0][0], h, w)]
        videos += [frames(c["result"]["decoded_video"][0], h, w, True) for c in [baseline, candidate]]
        assert len({v.shape for v in videos}) == 1
        labels = ["Expert reference", args.baseline_label, args.candidate_label]
        montage = [
            np.concatenate([labelled(v[t], label) for v, label in zip(videos, labels)], axis=1)
            for t in range(len(videos[0]))
        ]
        name = f"episode_{sample['episode']}_offset_{sample['window_offset']}"
        imageio.mimwrite(
            args.output / f"{name}.mp4",
            montage,
            fps=8,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        imageio.mimsave(args.output / f"{name}.gif", montage, duration=125, loop=0)
        Image.fromarray(montage[0]).save(args.output / f"{name}_poster.jpg")
        Image.fromarray(np.concatenate([montage[t] for t in [0, 4, 8, 12, 16]], axis=0)).save(
            args.output / f"{name}.jpg"
        )
        truth = reference["action"][0][0].float()[:, :10]
        predicted = [c["result"]["action"][0].float().reshape(16, 10) for c in [baseline, candidate]]
        assert all(torch.isfinite(x).all() for x in [truth, *predicted])
        fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True)
        names = ["dx", "dy", "dz", "rot6d[0]", "rot6d[1]", "rot6d[2]", "rot6d[3]", "rot6d[4]", "rot6d[5]", "gripper"]
        for dimension, ax in enumerate(axes.flat):
            for value, label, color in zip([truth, *predicted], labels, ["black", "#1677b8", "#df6a25"]):
                ax.plot(np.arange(16), value[:, dimension].numpy(), label=label, color=color, linewidth=1.6)
            ax.set_title(names[dimension], fontsize=10)
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Episode {sample['episode']}, offset {sample['window_offset']} — normalized action channels")
        fig.supxlabel("Action step (20 Hz)")
        fig.tight_layout()
        fig.savefig(args.output / f"{name}_actions.png", dpi=130)
        plt.close(fig)
        np.savez(
            args.output / f"{name}_actions.npz",
            expert=truth.numpy(),
            nano=predicted[0].numpy(),
            three_tower=predicted[1].numpy(),
        )
        row = dict(
            sample=sample,
            identical_initial_inputs=True,
            action_pair_max_abs=float((predicted[0] - predicted[1]).abs().max()),
            action_pair_rmse=float((predicted[0] - predicted[1]).square().mean().sqrt()),
        )
        for label, value, video in zip(["nano", "three_tower"], predicted, videos[1:]):
            row[label] = dict(
                normalized_action_mse=float((value - truth).square().mean()),
                normalized_translation_mse=float((value[:, :3] - truth[:, :3]).square().mean()),
                normalized_rotation6d_mse=float((value[:, 3:9] - truth[:, 3:9]).square().mean()),
                normalized_gripper_mse=float((value[:, 9] - truth[:, 9]).square().mean()),
                future_video_mae_0_255=float(np.abs(video[1:].astype(float) - videos[0][1:].astype(float)).mean()),
            )
        metrics.append(row)
        sections.append(f'''<section><h2>Episode {sample["episode"]} · window {sample["window_offset"]}</h2>
<p>{html.escape(sample["caption"])} · seed {two["args"]["seed"] + i}</p>
<video controls loop muted playsinline preload="none" src="{name}.mp4" poster="{name}_poster.jpg"></video>
<p>Left: expert reference · Middle: {html.escape(args.baseline_label)} · Right: {html.escape(args.candidate_label)}. Playback 8 fps (2.5× slower).</p>
<details><summary>Action curves and sampled frames</summary><img src="{name}_actions.png"><img src="{name}.jpg"></details>
<p>Normalized action MSE: {html.escape(args.baseline_label)} {row["nano"]["normalized_action_mse"]:.5f}; {html.escape(args.candidate_label)} {row["three_tower"]["normalized_action_mse"]:.5f}.</p></section>''')
    report = dict(
        samples=metrics,
        batch_sha256=two["batch_file_sha256"],
        sampling={k: two["args"][k] for k in ["steps", "shift", "guidance", "seed"]},
        candidate_trainability=three["trainability"],
        copied_tensors=three["copied_parameter_tensors"],
        limitation="Selected held-out windows; sampled predictions are not simulator rollouts or success rates. Native BF16 is not required to be bitwise equal.",
        labels=dict(baseline=args.baseline_label, candidate=args.candidate_label),
    )
    if args.closed_loop_baseline:
        report["closed_loop"], section = render_closed_loop(
            args.closed_loop_baseline, args.closed_loop_candidate, args.output
        )
        sections.append(section)
        report["limitation"] = (
            "Three offline windows and one simulator initial state per model; no training or reliable success-rate estimate. Native BF16 is not bitwise equal."
        )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    page = """<!doctype html><html><meta charset="utf-8"><title>LIBERO — action tower comparison</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1080px;background:#f3f5f8;color:#18202b}section{background:white;padding:24px;margin:24px 0;border-radius:12px}video,img{width:100%;height:auto}p{line-height:1.6}summary{cursor:pointer}</style>
<h1>LIBERO · Action tower comparison</h1><p>任务：把碗放到盘子上。留出 episode 的固定窗口。输入仅当前图像和指令；未来示范视频与动作只作参考，不输入模型。</p>
<p>这是离线 WAM 预测，可视化包含真实 VAE 解码视频及归一化动作曲线，<b>不是仿真执行录像，不代表闭环成功率</b>。两组使用相同输入、初始噪声及采样参数，三塔使用 native+routed 正式路径。</p>
<p>35 steps · UniPC · shift 5 · guidance 1.5 · BF16 · chunk16 / 20 Hz · no robot state</p>"""
    (args.output / "index.html").write_text(page + "\n".join(sections) + "</html>\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
