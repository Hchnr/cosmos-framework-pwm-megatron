"""Audit aligned final evaluations and render a portable 1B comparison report."""

import argparse
import hashlib
import html
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch
from finalize_libero_five_episode_eval import fingerprint
from summarize_libero_paired_trials import read_trials

from cosmos_framework.model.generator.mot.small_action_tower import action_shape

PRESETS = ("depth_1b", "width_heads_1b", "width_head_dim_1b", "width_only_1b")
LABELS = dict(zip(PRESETS, ("A · 减层数", "B · 减 head 数", "C · 减 head dim", "D · 仅减 hidden / MLP")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/action_tower_1b"))
    p.add_argument("--checkpoint-step", type=int, choices=range(200, 1001, 200), default=1000)
    p.add_argument("--episodes", type=int, choices=range(1, 6), default=2)
    p.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    args = p.parse_args()
    suffix = "" if args.checkpoint_step == 1000 else f"_step{args.checkpoint_step}"
    root = args.root.resolve()
    repo = Path.cwd()
    baseline = repo / "outputs/libero_bowl_plate/step1000"
    fixture = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_bowl_plate_eval5")
    output = root / ("visuals" + suffix)
    output.mkdir(parents=True, exist_ok=True)
    py = str(repo / ".venv/bin/python")
    base = torch.load(baseline / "two_tower.pt", map_location="cpu", weights_only=False)
    initial = [fingerprint(c["initial"]) for c in base["captures"][: args.episodes]]
    base_trials = read_trials(baseline / "closed_loop_two_tower")
    base_trials = {i: base_trials[i] for i in range(args.episodes)}
    report = dict(
        checkpoint_step=args.checkpoint_step,
        episodes=args.episodes,
        fixture_sha256=hashlib.sha256((fixture / "eval_batches.pt").read_bytes()).hexdigest(),
        experiments={},
    )
    cards = []
    for preset in args.presets:
        dest = root / ("eval" + suffix) / preset
        capture = torch.load(dest / "capture.pt", map_location="cpu", weights_only=False)
        shape = action_shape(preset)
        routing = capture["routing_audit"]
        expected_queries = {
            f"layer{layer}.{role}"
            for layer in range(36)
            for role in ("UND", "GEN_VIDEO", "GEN_ACTION")
            if role != "GEN_ACTION" or layer in shape.layers
        }
        assert routing.keys() == expected_queries, (preset, "Incomplete query-routing audit")
        for key, value in routing.items():
            width = shape.hidden if key.endswith(".GEN_ACTION") else 4096
            assert value["input_shape"][1] == width, (preset, key, value)
        checkpoint_audit = json.loads((root / f"checkpoint_audit_{preset}{suffix}.json").read_text())
        assert [fingerprint(c["initial"]) for c in capture["captures"]] == initial, "Offline input/noise mismatch"
        assert all(c["initial"][0]["state_indexes"].numel() == 0 for c in capture["captures"])
        trials = read_trials(dest / "closed_loop")
        assert set(trials) == set(range(args.episodes))
        for trial, (candidate, _) in trials.items():
            ref = base_trials[trial][0]
            for key in ("dataset_episode", "initial_state_sha256", "protocol", "seed"):
                assert candidate[key] == ref[key], (preset, trial, key)
            assert candidate["error"] is None, candidate["error"]
            assert candidate["requests"][0]["observation_sha256"] == ref["requests"][0]["observation_sha256"]
            np.testing.assert_array_equal(candidate["observations"][0]["qpos"], ref["observations"][0]["qpos"])
            for key in (
                "guidance",
                "num_steps",
                "fps",
                "seed",
                "action_chunk_size",
                "raw_action_dim",
                "action_stats_path",
            ):
                assert candidate["server_info"][key] == ref["server_info"][key], key
            assert candidate["steps"] == len(candidate["actions"]) <= 1000
            assert candidate["success"] or candidate["steps"] == 1000
            assert np.isfinite(np.asarray(candidate["actions"])).all()
        od = output / preset / "offline"
        cd = output / preset / "simulation"
        commands = [
            [
                "tools/render_small_action_candidate.py",
                "--capture",
                str(dest / "capture.pt"),
                "--fixture",
                str(fixture / "eval_batches.pt"),
                "--closed-loop",
                str(dest / "closed_loop"),
                "--output",
                str(output / preset),
                "--label",
                preset + f" / step{args.checkpoint_step}",
                "--step",
                str(args.checkpoint_step),
            ]
        ]
        for j, cmd in enumerate(commands):
            with (dest / f"render_{j}.log").open("w") as f:
                subprocess.run(
                    [py, *cmd],
                    env={
                        **os.environ,
                        "LD_LIBRARY_PATH": "",
                        "PYTHONPATH": str(repo),
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        om = json.loads((od / "metrics.json").read_text())
        row = dict(
            successes=sum(x["success"] for x, _ in trials.values()),
            steps=[trials[i][0]["steps"] for i in range(args.episodes)],
            action_mse=float(np.mean([x["candidate"]["normalized_action_mse"] for x in om["samples"]])),
            video_mae=float(np.mean([x["candidate"]["future_video_mae_0_255"] for x in om["samples"]])),
            inputs_aligned=True,
            query_routing_verified=True,
            action_parameters=checkpoint_audit["action_parameters_including_adapters"],
            adapter_parameters=checkpoint_audit["adapter_parameters"],
            action_shape=shape.__dict__,
        )
        report["experiments"][preset] = row
        for i, sample in enumerate(om["samples"]):
            ep = sample["sample"]["episode"]
            prefix = f"{preset}/offline/episode_{ep}_offset_0"
            cards.append(
                dict(
                    preset=preset,
                    episode=ep,
                    kind="视频生成",
                    src=prefix + ".mp4",
                    poster=prefix + "_poster.jpg",
                    image=prefix + "_actions.png",
                    caption=f"专家参考 / {preset}；action MSE {sample['candidate']['normalized_action_mse']:.5f}",
                )
            )
            prefix = f"{preset}/simulation/trial_{i:03d}"
            cards.append(
                dict(
                    preset=preset,
                    episode=ep,
                    kind="仿真",
                    src=prefix + ".mp4",
                    poster=prefix + ".jpg",
                    image=prefix + "_trajectory.png",
                    caption=f"{preset}；{'成功' if trials[i][0]['success'] else '未完成'}，{trials[i][0]['steps']} 步",
                )
            )
        del capture
    # Keep only portable presentation assets; do not package arrays/checkpoints.
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in (".npz", ".gif"):
            path.unlink()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / ("evaluation_summary" + suffix + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    sections = []
    for c in cards:
        sections.append(
            f'<section data-step="{args.checkpoint_step}" data-preset="{c["preset"]}" data-episode="{c["episode"]}" data-kind="{c["kind"]}"><h2>Episode {c["episode"]} · {LABELS[c["preset"]]} · {c["kind"]}</h2><p>{html.escape(c["caption"])}</p><video controls loop muted playsinline preload="metadata" poster="{c["poster"]}" src="{c["src"]}"></video><details><summary>动作轨迹</summary><img loading="lazy" src="{c["image"]}"></details></section>'
        )
    controls = ""
    for key, values in [
        ("episode", [43, 69, 72, 80, 132][: args.episodes]),
        ("kind", ["视频生成", "仿真"]),
        ("step", [args.checkpoint_step]),
        ("preset", args.presets),
    ]:
        controls += (
            f'<div class="filters" data-key="{key}"><b>{dict(episode="Episode", kind="类型", step="Step", preset="实验")[key]}</b>'
            + "".join(
                f'<label><input type="checkbox" value="{value}" checked>{LABELS.get(value, value)}</label>'
                for value in values
            )
            + "</div>"
        )
    table = (
        "<table><tr><th>实验</th><th>Action 参数</th><th>成功</th><th>Action MSE（归一化）</th><th>Video MAE（0–255）</th></tr>"
        + "".join(
            f"<tr><td>{LABELS[k]}</td><td>{v['action_parameters'] / 1e9:.3f}B</td><td>{v['successes']}/{args.episodes}</td><td>{v['action_mse']:.5f}</td><td>{v['video_mae']:.4f}</td></tr>"
            for k, v in report["experiments"].items()
        )
        + "</table>"
    )
    page = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIBERO · Action 1B</title><style>
body{margin:0;background:#181c24;color:#d8dfe8;font:15px system-ui}header{position:sticky;top:0;background:#252b38f5;backdrop-filter:blur(14px);padding:14px 4vw;z-index:2;border-bottom:1px solid #424e63}.filters{display:flex;gap:14px;flex-wrap:wrap;margin:9px 0}label,button{cursor:pointer}button{background:#52627c;color:#fff;border:0;padding:9px 16px;border-radius:7px;margin:5px}main{max-width:1250px;margin:auto;padding:24px}section{background:#242b37;border:1px solid #394354;border-radius:14px;padding:20px;margin:24px 0}video,img{width:100%;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #414c60}h1{font-size:24px}h2{font-size:19px}p{color:#b6c3d4;line-height:1.7}[hidden]{display:none!important}
</style><header>"""
        + controls
        + """<button id="play">当前视频从头同步播放</button><button id="pause">暂停</button><button id="reset">全选 / 重置筛选</button><span id="visible-count"></span><span>step1000 · GBS16 · 无 robot state</span></header><main><h1>LIBERO · 约 1B Action Tower</h1><p>任务：把碗放到盘子上。展示 ABCD 四种约1B的Action结构。各组使用相同留出 Episode；仅单任务5个Episode，不代表泛化能力。</p><p>离线：每个 Episode 一个固定窗口，专家示范与本组预测并排，8 fps（2.5倍慢放）。仿真：本组实际执行录像，20 fps，成功或1000个环境步停止。生成视频不代表任务执行成功。</p>"""
        + '<div style="overflow:auto">'
        + table
        + "</div>"
        + '<p id="empty-state" hidden>没有匹配的结果，请调整筛选或点击全选。</p>'
        + "".join(sections)
        + """</main><script>
const cards=[...document.querySelectorAll('section')];
function filter(){const chosen={};document.querySelectorAll('.filters').forEach(g=>chosen[g.dataset.key]=[...g.querySelectorAll('input:checked')].map(x=>x.value));cards.forEach(c=>{c.hidden=Object.entries(chosen).some(([k,v])=>!v.includes(c.dataset[k]));if(c.hidden)c.querySelector('video').pause()});const n=cards.filter(c=>!c.hidden).length;document.querySelector('#visible-count').textContent=`${n} 项结果 · `;document.querySelector('#empty-state').hidden=n>0}
document.querySelectorAll('input').forEach(x=>x.onchange=filter);document.querySelector('#reset').onclick=()=>{document.querySelectorAll('.filters input').forEach(x=>x.checked=true);filter()};
document.querySelector('#play').onclick=()=>{const videos=cards.filter(c=>!c.hidden).map(c=>c.querySelector('video'));videos.forEach(v=>{v.pause();v.currentTime=0});videos.forEach(v=>v.play().catch(()=>{}))};document.querySelector('#pause').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());filter();
</script></html>"""
    )
    page = (
        page.replace("step1000 · GBS16", f"step{args.checkpoint_step} · GBS16")
        .replace("（5/5）", f"（本次选定{args.episodes}个Episode）")
        .replace("仅单任务5个Episode", f"仅单任务{args.episodes}个Episode")
        .replace("展示 ABCD 四种约1B的Action结构", "展示选定实验组的约1B Action结构")
    )
    page = page.replace(
        "</style>",
        """
.filters{align-items:center;gap:8px}.filters b{min-width:65px;font-size:13px;color:#aebacd}
.filters label{border:1px solid #4a5262;border-radius:20px;padding:7px 12px;color:#9caabd;background:#242a35}
.filters label:has(input:checked){background:#45576b;border-color:#8394a8;color:#f0f3f6}
.filters input{position:absolute;opacity:0;width:1px;height:1px}.filters label:focus-within{outline:2px solid #b7b4d0}
section p{overflow-wrap:anywhere}summary{cursor:pointer;padding:12px 0}td{white-space:nowrap}
@media(max-width:600px){header{padding:10px 12px}main{padding:12px}section{padding:12px}.filters label{font-size:12px;padding:6px 9px}}
</style>""",
    )
    (output / "index.html").write_text(page)
    # Require every declared asset, including trajectory plots, to exist.
    for card in cards:
        for key in ("src", "image", "poster"):
            assert (output / card[key]).is_file(), card[key]
    with tarfile.open(root / ("visuals" + suffix + ".tar.gz"), "w:gz") as tf:
        tf.add(output, arcname=f"action_tower_1b_step{args.checkpoint_step}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
