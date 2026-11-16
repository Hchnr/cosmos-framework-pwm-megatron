"""Build a portable, presentation-only LIBERO step0/step1000 evaluation report.

No model dependencies: copies only HTML, metrics JSON, MP4, JPEG and PNG from
the existing visualizations. Original captures, logs and checkpoints stay put.
"""

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def video(path, poster):
    ratio = "512/284" if "closed_loop" in path else "960/188"
    return (
        f'<video controls muted loop playsinline preload="none" src="{path}" '
        f'poster="{poster}" style="aspect-ratio:{ratio}">浏览器不支持视频，请<a href="{path}">下载 MP4</a>。</video>'
    )


def picture(path, description):
    return (
        f'<a href="{path}" target="_blank" rel="noopener"><img loading="lazy" '
        f'src="{path}" alt="{html.escape(description)}"></a>'
    )


def status(success):
    return f'<span class="badge {"ok" if success else "fail"}">{"成功" if success else "未完成"}</span>'


def offline_card(row, folder, label):
    sample = row["sample"]
    stem = f"episode_{sample['episode']}_offset_{sample['window_offset']}"
    a, b = row["nano"], row["three_tower"]
    return f"""<article class="card result-card" data-step="{label.removeprefix("Step")}">
      <h3>{label} · Episode {sample["episode"]} / offset {sample["window_offset"]}</h3>
      {video(f"{folder}/{stem}.mp4", f"{folder}/{stem}_poster.jpg")}
      <p class="caption">左：专家参考 · 中：双塔 · 右：三塔 · 8 fps 慢放</p>
      {
        table(
            ["指标", "双塔", "三塔"],
            [
                ["Action MSE", f"{a['normalized_action_mse']:.6f}", f"{b['normalized_action_mse']:.6f}"],
                ["Video MAE", f"{a['future_video_mae_0_255']:.3f}", f"{b['future_video_mae_0_255']:.3f}"],
            ],
        )
    }
      <h4>归一化动作曲线</h4>{picture(f"{folder}/{stem}_actions.png", "专家、双塔与三塔的十维动作曲线")}
      <h4>预测视频关键帧</h4>{picture(f"{folder}/{stem}.jpg", "专家与两模型的预测视频关键帧")}
    </article>"""


CSS = """
:root{color-scheme:light;--ink:#152e40;--muted:#576c7b;--line:#dce5ec;--blue:#146b93}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:78px}
body{margin:0;background:#f3f6f8;color:var(--ink);font:15px/1.65 system-ui,-apple-system,sans-serif}
header{background:#132e40;color:white;padding:42px max(24px,calc((100vw - 1440px)/2))}
header p{color:#c1d6e2;max-width:980px}.eyebrow{letter-spacing:.16em;font-size:12px;color:#8dc9e3}
h1{font-size:clamp(26px,4vw,42px);margin:8px 0}h2{font-size:25px;margin:0 0 8px}h3{font-size:17px;margin:0 0 14px}
h4{font-size:13px;color:var(--muted);margin:20px 0 8px}p{margin:10px 0}
nav{position:sticky;top:0;z-index:2;display:flex;gap:8px;overflow:auto;background:#fffffff5;
border-bottom:1px solid var(--line);padding:12px max(20px,calc((100vw - 1440px)/2));backdrop-filter:blur(8px)}
nav a{white-space:nowrap;text-decoration:none;padding:4px 12px;border-radius:20px;background:#edf3f7}
a{color:var(--blue)}main{max-width:1480px;margin:auto;padding:28px 20px 70px}
section{margin-bottom:44px}.intro{max-width:1100px;color:var(--muted);margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.stats{display:grid;
grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:26px}
.stat,.card,.note{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;min-width:0}
.stat strong{display:block;font-size:30px;line-height:1.4}.stat span,.caption{color:var(--muted);font-size:13px}
.note{border-left:4px solid #d7a340;margin:18px 0}.card.attention{border-color:#d3a357}
video,img{display:block;width:100%;height:auto;border-radius:7px}video{background:#111;aspect-ratio:2/1}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}
table{width:100%;border-collapse:collapse;background:white;font-size:13px;text-align:left}
th,td{padding:10px 13px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#eaf1f5}
#protocol td{white-space:normal}#protocol td:first-child{min-width:110px;width:18%}
tr:last-child td{border-bottom:0}.badge{display:inline-block;border-radius:20px;padding:2px 9px;font-size:12px}
.ok{color:#166249;background:#e0f3e9}.fail{color:#9a4812;background:#fff0dc}.anchor-row{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}
.anchor-row a{padding:3px 10px;background:#e4edf3;border-radius:5px;text-decoration:none}
.episode{margin-top:26px}.episode .grid{grid-template-columns:repeat(3,minmax(0,1fr))}
code{overflow-wrap:anywhere;white-space:normal}footer{border-top:1px solid var(--line);padding-top:20px;color:var(--muted)}
@media(max-width:1050px){.episode .grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){.grid,.episode .grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}
header{padding:28px 20px}.card{padding:14px}main{padding:20px 12px}.stat strong{font-size:25px}}
"""


CSS += """
:root{color-scheme:dark;--ink:#e0e5e9;--muted:#a2adb8;--line:#35414e;--blue:#a7baca}
body{background:#171c23}header{background:linear-gradient(125deg,#263442,#302f3e);padding-top:32px;padding-bottom:28px}
header p{color:#aebbc6}.eyebrow{color:#a9c2c1}h1{font-weight:550;letter-spacing:-.025em}
nav{display:block;max-height:65vh;overflow:auto;background:#202730f5;z-index:10;padding:14px max(20px,calc((100vw - 1440px)/2));box-shadow:0 8px 25px #0003}
.toolbar-head,.filter-row,.playback{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.toolbar-head{margin-bottom:12px}.toolbar-title{font-weight:650}.playback{margin-left:auto}
#selection-count,.toolbar-hint{font-size:12px;color:var(--muted)}.toolbar-hint{margin:10px 0 0}
.filter-row{margin-top:7px}.filter-label{width:88px;color:var(--muted);font-size:12px}
button{font:inherit;font-size:13px;cursor:pointer;color:#aeb8c3;background:#252e39;border:1px solid #46515f;border-radius:7px;padding:6px 13px;transition:background .15s,border-color .15s}
button:hover{background:#394555}button[aria-pressed=true]{color:#e9f3f0;background:#3c5658;border-color:#759799}
button[aria-pressed=false]{opacity:.7}button:focus-visible,a:focus-visible{outline:2px solid #b4d4d0;outline-offset:3px}
button.primary{background:#57627e;color:#fff;border-color:#8290ab}button:disabled{opacity:.6;cursor:wait}
.stat,.card,.note{background:#222b35;border-color:var(--line);border-radius:12px}.card{box-shadow:0 6px 22px #0002}
.note{border-left-color:#a89776;color:#b9c2cc}table{background:#222b35}th{background:#2c3744}
.ok{color:#b9d4c7;background:#354d46}.fail{color:#dcc8a4;background:#51483b}
.anchor-row a{background:#303c49}.caption{color:#a6b4c2}h2{font-weight:550}h3{color:#c4cfd9}
img{background:#e9ebee}.result-kind{margin-top:22px}.grid:has(>.card:not([hidden]):only-child){grid-template-columns:1fr}
[hidden]{display:none!important}#overview{margin-bottom:28px}#protocol{padding-top:24px;border-top:1px solid var(--line)}
@media(max-width:700px){nav{padding:10px 12px}.toolbar-head{gap:6px}.playback{margin-left:0;width:100%}button{padding:5px 10px}.filter-label{width:76px}.toolbar-hint{font-size:11px}}
"""

SCRIPT = """
const filterButtons = [...document.querySelectorAll('[data-filter]')];
const videos = [...document.querySelectorAll('video')];
let playRun = 0;
const sync = document.getElementById('sync-play');
function visibleVideos() { return videos.filter(v => !v.closest('[hidden]')); }
function cancelPlayback() {
  playRun++;
  videos.forEach(v => v.pause());
  sync.disabled = false;
  sync.textContent = '↻ 从头同步播放';
}
function applyFilters() {
  cancelPlayback();
  const selected = key => new Set(filterButtons.filter(b => b.dataset.filter === key && b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.value));
  const episodes = selected('episode'), kinds = selected('kind'), steps = selected('step');
  document.querySelectorAll('.episode-result').forEach(section => {
    section.hidden = !episodes.has(section.dataset.episode) || !kinds.size || !steps.size;
    section.querySelectorAll('.result-kind').forEach(group => {
      group.hidden = !kinds.has(group.dataset.kind);
      group.querySelectorAll('.result-card').forEach(card => { card.hidden = !steps.has(card.dataset.step); });
    });
  });
  const count = visibleVideos().length;
  document.getElementById('selection-count').textContent = `${count} 个对比视频`;
  document.getElementById('empty-state').hidden = count > 0;
}
filterButtons.forEach(b => b.addEventListener('click', () => {
  b.setAttribute('aria-pressed', b.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
  applyFilters();
}));
document.getElementById('reset-filters').onclick = () => {
  filterButtons.forEach(b => b.setAttribute('aria-pressed', 'true'));
  applyFilters();
};
document.getElementById('pause-all').onclick = cancelPlayback;
function readyAtStart(v) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('视频加载超时')), 30000);
    function finish(error) {
      clearTimeout(timeout);
      ['loadeddata','canplay','seeked','error'].forEach(e => v.removeEventListener(e, check));
      error ? reject(error) : resolve();
    }
    function check() {
      if (v.error) finish(new Error('视频加载失败'));
      else if (v.readyState >= 2 && !v.seeking && v.currentTime < .05) finish();
    }
    ['loadeddata','canplay','seeked','error'].forEach(e => v.addEventListener(e, check));
    v.preload = 'auto';
    if (v.readyState === 0) v.load();
    v.currentTime = 0;
    check();
  });
}
sync.onclick = async () => {
  cancelPlayback();
  const run = playRun;
  const selected = visibleVideos();
  if (!selected.length) return;
  sync.disabled = true;
  sync.textContent = '加载视频…';
  try {
    await Promise.all(selected.map(readyAtStart));
    if (run !== playRun) return;
    await Promise.all(selected.map(v => v.play()));
    if (run === playRun) sync.textContent = '↻ 从头同步播放';
  } catch (error) {
    if (run === playRun) {
      selected.forEach(v => v.pause());
      sync.textContent = '加载或播放失败，点击重试';
    }
  } finally { if (run === playRun) sync.disabled = false; }
};
new ResizeObserver(() => {
  document.documentElement.style.scrollPaddingTop = `${document.querySelector('nav').offsetHeight + 12}px`;
}).observe(document.querySelector('nav'));
applyFilters();
"""


def build(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError("Output must be a separate presentation directory")
    audit = read(source / "eval5_alignment.json")
    if not audit["passed"]:
        raise ValueError("Cross-checkpoint input alignment must pass first")
    offline = {step: read(source / f"step{step}/visuals/metrics.json")["samples"] for step in [0, 1000]}
    closed = {step: read(source / f"step{step}/closed_loop_visuals/metrics.json") for step in [0, 1000]}
    episodes = [r["episode"] for r in audit["episodes"]]
    if len(episodes) != 5 or len(set(episodes)) != 5:
        raise ValueError("Expected exactly five distinct Episodes")
    for step in [0, 1000]:
        if [r["sample"]["episode"] for r in offline[step]] != episodes:
            raise ValueError("Offline Episode order differs from frozen manifest")
        c = closed[step]
        if c["completed"] != [5, 5] or c["paired_trials"] != 5 or any(any(p["errors"]) for p in c["pairs"]):
            raise ValueError("Closed-loop evaluations are not complete and error-free")
    output.mkdir(parents=True, exist_ok=True)
    for step in [0, 1000]:
        for kind in ["visuals", "closed_loop_visuals"]:
            folder = f"step{step}/{kind}"
            target = output / folder
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted((source / folder).iterdir()):
                if path.suffix in {".html", ".json", ".mp4", ".jpg", ".png"} and path.is_file():
                    shutil.copy2(path, target / path.name)
    shutil.copy2(source / "eval5_alignment.json", output / "alignment.json")
    sections = ['<section id="overview"><h2>同一组 5 个 Episode · 四组对照</h2>']
    rows = []
    for step in [0, 1000]:
        for i, (key, label) in enumerate([("nano", "双塔"), ("three_tower", "三塔")]):
            count = closed[step]["successes"][i]
            rows.append(
                [
                    f"Step{step}",
                    label,
                    f"{count}/5 ({count / 5:.0%})",
                    f"{mean(r[key]['normalized_action_mse'] for r in offline[step]):.6f}",
                    f"{mean(r[key]['future_video_mae_0_255'] for r in offline[step]):.3f}",
                ]
            )
    sections.append(table(["Checkpoint", "结构", "闭环成功", "Action MSE · 5窗口", "Video MAE · 5窗口"], rows))
    sections.append(
        '<div class="note">所有 Checkpoint 共用同一组 Episode、离线窗口和仿真起始状态，'
        "已核验离线实际输入/噪声，以及闭环首次策略图像和初始位置。"
        "每个 Episode 闭环跑到成功或 1000 步，未完成记为失败。"
        "5 次单任务评测用于检查训练效果，不能证明结构优劣。</div>"
    )
    sections.append(
        '<div class="anchor-row">'
        + "".join(f'<a href="#episode-{e}">Episode {e}</a>' for e in episodes)
        + "</div></section>"
    )
    for index, episode in enumerate(episodes):
        trial = audit["episodes"][index]["trial"]
        sections.append(
            f'<section class="episode-result" data-episode="{episode}" id="episode-{episode}"><h2>Episode {episode}</h2>'
            f'<p class="intro">固定窗口 offset 0 · 原始示范 {html.escape(audit["episodes"][index]["demo"])} · '
            f"仿真 seed {42 + trial} · 每个闭环视频均为左双塔、右三塔。</p>"
            '<div class="result-kind" data-kind="simulation"><h3>闭环仿真 · 实际执行动作</h3><div class="grid">'
        )
        for step in [0, 1000]:
            row = next(r for r in closed[step]["pairs"] if r["dataset_episode"] == episode)
            stem = f"step{step}/closed_loop_visuals/trial_{trial:03d}"
            sections.append(f"""<article class="card result-card" data-step="{step}"><h3>Step{step} · 实际执行</h3>
              <p>双塔 {status(row["two_success"])} {row["two_steps"]} 步 ·
              三塔 {status(row["three_success"])} {row["three_steps"]} 步</p>
              {video(stem + ".mp4", stem + ".jpg")}
              <p class="caption">左：双塔 · 右：三塔 · 20 fps · 先结束的一侧定格末帧</p>
              <h4>执行轨迹 · 末端 XYZ 与夹爪动作</h4>
              {picture(stem + "_trajectory.png", f"Episode {episode} Step{step} 执行轨迹")}
              <p class="caption">双塔 / 三塔共同持续时段的平均 EEF 距离：{row["eef_distance_mean_m"]:.4f} m</p></article>""")
        sections.append(
            '</div></div><div class="result-kind" data-kind="generation"><h3 style="margin-top:24px">离线视频生成 · 同一固定窗口</h3><div class="grid">'
        )
        for step in [0, 1000]:
            sections.append(offline_card(offline[step][index], f"step{step}/visuals", f"Step{step}"))
        sections.append("</div></div></section>")
    sections.append(
        '<section id="protocol"><h2>输入对齐与评测设置</h2>'
        + table(
            ["项目", "设置 / 核验"],
            [
                ["Episode 选择", "前5个按编号排序的留出Episode：43、69、72、80、132；不按效果筛选"],
                ["数据与仿真对应", "完整action序列在夹爪转换后FP32逐元素匹配原始HDF5示范，冻结对应场景XML及states[0]"],
                ["离线输入", "每个Episode首个窗口；同一冻结batch；实际文本/位置/视频和action初始噪声跨四组逐元素相同"],
                ["闭环输入", "同一初始状态、环境seed42+trial、相机与提示词；四组首次策略图像hash及初始qpos一致"],
                ["模型条件", "TI2VA · 当前双视图 + 文本 · 无robot state、无历史action；专家未来视频/动作仅作参考"],
                ["采样", "BF16 / native+routed · UniPC 35步 · shift5 · guidance1.5 · 无EMA"],
                ["闭环执行", "20 Hz · chunk16执行前8步再观测 · 成功或1000个策略动作 · 从示范状态开始，无额外warm-up"],
                ["离线展示", "17帧对应0.8秒预测跨度，8 fps慢放约2秒；视频是模型生成，不是仿真执行"],
                ["效率", "8卡独立模型任务 + 多进程CPU仿真；闭环通过batch1 action接口跳过无用视频解码，保留联合扩散"],
            ],
        )
        + "<p>离线 Action MSE 为归一化后10维、16步均方误差；Video MAE 为未来16帧像素误差（0–255）。"
        "离线误差不是任务成功率。闭环后续观测随各自执行动作变化，不要求跨模型一致。</p>"
        "<p>原始HDF5的 states 与 obs 存在动作前/后的时间差；因此离线与仿真的首帧不宣称逐像素相同。"
        "四个模型条件在每种评测内部共用完全相同的输入。这里只测单个任务的5个Episode，未重复训练种子。</p>"
        "<p>本目录仅含展示视频、图片和小型指标文件。所有资源为相对路径，下载整目录后直接打开 index.html，"
        "无需模型、服务或网络。视频按需播放并循环，图片延迟加载；顶部支持多选筛选与从头同步播放。</p>"
        '<p><a href="alignment.json">跨Checkpoint输入对齐证据</a> · '
        '<a href="step0/visuals/metrics.json">Step0离线指标</a> · '
        '<a href="step1000/visuals/metrics.json">Step1000离线指标</a> · '
        '<a href="step0/closed_loop_visuals/metrics.json">Step0闭环指标</a> · '
        '<a href="step1000/closed_loop_visuals/metrics.json">Step1000闭环指标</a></p></section>'
    )

    def filter_row(key, title, choices):
        buttons = "".join(
            f'<button type="button" data-filter="{key}" data-value="{value}" aria-pressed="true">{label}</button>'
            for value, label in choices
        )
        return f'<div class="filter-row"><span class="filter-label">{title}</span>{buttons}</div>'

    nav = (
        '<div class="toolbar-head"><span class="toolbar-title">筛选对照</span>'
        '<span id="selection-count" role="status" aria-live="polite"></span>'
        '<div class="playback"><button id="sync-play" class="primary">↻ 从头同步播放</button>'
        '<button id="pause-all">暂停全部</button><button id="reset-filters">重置筛选</button></div></div>'
        + filter_row("episode", "Episode", [(str(e), str(e)) for e in episodes])
        + filter_row("kind", "评测类型", [("simulation", "仿真"), ("generation", "视频生成")])
        + filter_row("step", "Checkpoint", [("0", "Step 0"), ("1000", "Step 1000")])
        + '<p class="toolbar-hint">同一行可多选，不同行交叉筛选 · 播放仅作用于筛选后的结果 · 视频循环播放</p>'
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>LIBERO · 5 Episode · Step0 / Step1000</title><style>{CSS}</style></head><body>
      <header><div class="eyebrow">COSMOS3 / LIBERO / BOWL → PLATE</div>
      <h1>同一组 5 个 Episode · 训练前后对照</h1>
      <p>Step0 → Step1000 · 双塔 / 三塔 · 每个Episode一个视频窗口、一次完整闭环，成功或1000步结束。</p>
      <p>四组评测全部完成 · 实际输入对齐核验通过 · 无robot state输入</p></header>
      <nav aria-label="多选结果筛选">{nav}</nav><main><p id="empty-state" class="note" hidden>暂无符合条件的结果，请至少选择一个 Episode、评测类型和 Checkpoint。</p>{"".join(sections)}
      <footer>5个Episode · 10个离线对比视频 · 10个闭环对比视频 · 支持多选筛选与循环播放</footer></main><script>{SCRIPT}</script></body></html>"""
    (output / "index.html").write_text(page)
    (output / "README.txt").write_text(
        "LIBERO five-Episode step0 / step1000 presentation bundle\n"
        "Open index.html in a modern browser. Download this complete directory.\n"
        "Contains HTML, MP4, JPEG/PNG plots and metrics JSON only.\n"
        "No checkpoints, tensor captures, logs, duplicate GIFs or NPZ arrays.\n"
    )
    inventory = {
        str(p.relative_to(output)): p.stat().st_size
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "bundle_manifest.json"
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(
            dict(
                generated_utc=datetime.now(timezone.utc).isoformat(),
                files=inventory,
                total_bytes_excluding_manifest=sum(inventory.values()),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{output / 'index.html'}\n{len(inventory)} files + manifest, {sum(inventory.values()) / 1024**2:.2f} MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/libero_bowl_plate"))
    parser.add_argument("--output", type=Path, default=Path("outputs/libero_bowl_plate/libero_bowl_plate_eval"))
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
