#!/usr/bin/env python3
"""Collect the one-attempt fixed-pack sweep into auditable CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from tools.seqlen_efficiency_data import SAMPLE_COUNTS

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "plans/v0.2.6_seqlen_efficiency"
WORLD_SIZE = 4


def read_benchmark(run: Path, scene: str) -> dict:
    paths = sorted(run.glob("**/seqlen_efficiency/rank_*.jsonl"))
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    if not records:
        raise ValueError("No benchmark JSONL")
    by_step = {}
    for record in records:
        by_step.setdefault(record["iteration"], []).append(record)
    if set(by_step) != set(range(1, 11)) or any(len(rank_rows) != WORLD_SIZE for rank_rows in by_step.values()):
        raise ValueError("Incomplete 10-step, four-rank benchmark")
    steady = {step: values for step, values in by_step.items() if all(v["window"] == "steady" for v in values)}
    if set(steady) != set(range(4, 9)):
        raise ValueError(f"Expected steady updates 4–8; found {sorted(steady)}")
    for rank in range(WORLD_SIZE):
        end_warmup = next(v["compile_unique_graphs"] for v in by_step[3] if v["rank"] == rank)
        end_steady = next(v["compile_unique_graphs"] for v in by_step[8] if v["rank"] == rank)
        if end_steady > end_warmup:
            raise ValueError(f"Rank {rank} compiled {end_steady - end_warmup} graph(s) during steady updates")
        steady_shapes = {
            tuple(mb["sample_lengths"])
            for step in steady
            for v in by_step[step]
            if v["rank"] == rank
            for mb in v["microbatches"]
        }
        profile_shapes = {
            tuple(mb["sample_lengths"])
            for step in (9, 10)
            for v in by_step[step]
            if v["rank"] == rank
            for mb in v["microbatches"]
        }
        if steady_shapes != profile_shapes:
            raise ValueError(f"Rank {rank} profile pack lengths differ from steady timing packs")
    elapsed = sum(max(v["seconds"] for v in values) for values in steady.values())
    samples = sum(sum(mb["samples"] for mb in v["microbatches"]) for values in steady.values() for v in values)
    real_tokens = sum(sum(mb["real_total"] for mb in v["microbatches"]) for values in steady.values() for v in values)
    lengths = [
        length
        for values in steady.values()
        for v in values
        for mb in v["microbatches"]
        for length in mb["sample_lengths"]
    ]
    result = {
        "steady_wall_seconds": elapsed,
        "seconds_per_step": elapsed / 5,
        "samples_per_second": samples / elapsed,
        "real_tokens_per_second": real_tokens / elapsed,
        "mean_real_sample_length": statistics.mean(lengths),
        "p50_real_sample_length": statistics.median(lengths),
        "p90_real_sample_length": sorted(lengths)[int(0.9 * (len(lengths) - 1))],
        "max_real_sample_length": max(lengths),
        "real_tokens_per_pack": real_tokens / sum(len(v["microbatches"]) for values in steady.values() for v in values),
        "compile_unique_graphs_at_steady_end": max(v["compile_unique_graphs"] for v in steady[8]),
    }
    if scene == "t2v":
        microbatches = [mb for values in steady.values() for v in values for mb in v["microbatches"]]
        gen_pairs = []
        pad_gen_pairs = []
        for mb in microbatches:
            splits = mb["split_lens"]
            pairs = list(zip(splits[::2], splits[1::2], strict=True))
            gen_pairs.append(sum(g * (u + g) for u, g in pairs))
            pad_und = mb["tokens"]["physical_und_token_length"] - sum(u for u, _ in pairs)
            pad_gen = mb["tokens"]["physical_gen_token_length"] - sum(g for _, g in pairs)
            pad_gen_pairs.append(pad_gen * (pad_und + pad_gen))
        result["real_gen_pairs_per_pack"] = statistics.mean(gen_pairs)
        result["pad_gen_pairs_per_pack"] = statistics.mean(pad_gen_pairs)
    return result


def get_nsys(run: Path, scene: str) -> dict:
    key = run.name
    report = run / "nsys" / f"{key}.nsys-rep"
    if not report.exists():
        raise FileNotFoundError(report)
    output = run / "nsys" / "kernels.json"
    if not output.exists() or output.stat().st_mtime_ns < report.stat().st_mtime_ns:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/seqlen_efficiency_nsys.py"),
                str(report),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    result = json.loads(output.read_text())
    if len(result["gpus"]) != WORLD_SIZE:
        raise ValueError(f"Expected {WORLD_SIZE} GPU traces, found {len(result['gpus'])}")
    gpus = result["gpus"]
    return {
        "attention_kernel_pct_mean": statistics.mean(g["categories"].get("attention", {}).get("pct", 0) for g in gpus),
        "attention_kernel_pct_min": min(g["categories"].get("attention", {}).get("pct", 0) for g in gpus),
        "attention_kernel_pct_max": max(g["categories"].get("attention", {}).get("pct", 0) for g in gpus),
        "gemm_kernel_pct_mean": statistics.mean(g["categories"].get("gemm", {}).get("pct", 0) for g in gpus),
        "gemm_kernel_pct_min": min(g["categories"].get("gemm", {}).get("pct", 0) for g in gpus),
        "gemm_kernel_pct_max": max(g["categories"].get("gemm", {}).get("pct", 0) for g in gpus),
        "attention_kernel_seconds_mean": statistics.mean(
            g["categories"].get("attention", {}).get("seconds", 0) for g in gpus
        ),
        "gemm_kernel_seconds_mean": statistics.mean(g["categories"].get("gemm", {}).get("seconds", 0) for g in gpus),
        "main_stream_seconds_mean": statistics.mean(g["primary_stream_seconds"] for g in gpus),
        "attention_kernel_seconds_per_microbatch": statistics.mean(
            g["categories"].get("attention", {}).get("seconds", 0) for g in gpus
        )
        / (4 if scene == "t2v" else 2),
        "gemm_kernel_seconds_per_microbatch": statistics.mean(
            g["categories"].get("gemm", {}).get("seconds", 0) for g in gpus
        )
        / (4 if scene == "t2v" else 2),
    }


def get_mfu(run: Path, scene: str) -> dict:
    output = run / "mfu.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/seqlen_efficiency_flops.py"),
            scene,
            str(run),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    mfu = json.loads(output.read_text())
    return {
        "estimated_useful_mfu_pct": mfu["mfu_pct"],
        "estimated_hfu_proxy_pct": mfu["hfu_proxy_pct"],
        "legacy_no_recompute_mfu_pct": mfu["legacy_no_recompute_mfu_pct"],
    }


def collect(root: Path, scenes: tuple[str, ...] = ("t2v",)) -> list[dict]:
    preflight = json.loads((root / "preflight.json").read_text())
    index = {(row["scene"], row["samples_per_pack"]): row for row in preflight}
    output = []
    for scene in scenes:
        for n in SAMPLE_COUNTS:
            source = index[(scene, n)]
            run = root / "runs" / f"{scene}_n{n:03d}"
            row = {
                "scene": scene,
                "samples_per_pack": n,
                "physical_total_seqlen": source["physical_total"],
                "physical_und": source["physical_und"],
                "physical_gen": source["physical_gen"],
                "physical_action": source["physical_action"],
                "preflight_real_total_seqlen": source["real_total"],
                "preflight_real_video": source["real_video"],
                "preflight_real_action": source["real_action"],
                "preflight_mean_real_sample_length": source["mean_real_sample_length"],
                "status": "pending",
            }
            status = run / "status.json"
            if status.exists():
                attempt = json.loads(status.read_text())
                row["status"] = "train_failed" if attempt["returncode"] else "completed"
                row["returncode"] = attempt["returncode"]
                if attempt["returncode"] == 0:
                    try:
                        for label, func in (
                            ("benchmark", lambda p: read_benchmark(p, scene)),
                            ("mfu", lambda p: get_mfu(p, scene)),
                            ("nsys", lambda p: get_nsys(p, scene)),
                        ):
                            row.update(func(run))
                    except Exception as error:
                        row["status"] = "incomplete_analysis"
                        row[f"{label}_error"] = str(error)
            output.append(row)
    return output


def plot_completed(rows: list[dict], delivery: Path) -> None:
    if sum(row["status"] == "completed" for row in rows) < 4:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metrics = (
        ("seconds_per_step", "Optimizer step wall (s)"),
        ("estimated_useful_mfu_pct", "Estimated MFU / HFU proxy (%)"),
        ("attention_kernel_pct_mean", "Attention: main-stream kernel time (%)"),
        ("gemm_kernel_pct_mean", "GEMM: main-stream kernel time (%)"),
    )
    for scene, color in (("t2v", "#1f77b4"), ("action", "#d95f02")):
        group = sorted(
            (row for row in rows if row["scene"] == scene and row["status"] == "completed"),
            key=lambda row: row["samples_per_pack"],
        )
        if len(group) < 2:
            continue
        for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
            axis.plot(
                range(len(group)),
                [row[metric] for row in group],
                marker="o",
                color=color,
                label=(
                    "Estimated MFU (real tokens, no AC)"
                    if metric == "estimated_useful_mfu_pct" and scene == "t2v"
                    else f"{scene} ({group[0]['physical_total_seqlen']:,} physical tokens)"
                ),
            )
            if metric == "estimated_useful_mfu_pct" and scene == "t2v":
                axis.plot(
                    range(len(group)),
                    [row["estimated_hfu_proxy_pct"] for row in group],
                    marker="s",
                    linestyle="--",
                    color="#ff7f0e",
                    label="Estimated HFU (AC + physical padding)",
                )
            axis.set_xticks(range(len(group)), [str(row["samples_per_pack"]) for row in group])
            axis.set_xlabel("Samples per pack (N); mean real length in report table")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle("Fixed physical pack, varied samples per pack; synthetic inputs, one run per group")
    delivery.mkdir(parents=True, exist_ok=True)
    figure.savefig(delivery / "seqlen_efficiency.png", dpi=180)
    svg = delivery / "seqlen_efficiency.svg"
    figure.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/v0.2.6_seqlen_efficiency")
    parser.add_argument("--scenes", nargs="+", choices=("t2v", "action"), default=["t2v"])
    args = parser.parse_args()
    root = args.root.resolve()
    scenes = tuple(dict.fromkeys(args.scenes))
    rows = collect(root, scenes)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# v0.2.6 固定 packed seqlen 的效率实测",
        "",
        "合成 offline latent 输入；每组一次 4×H100 训练。表中的 MFU/HFU 是逐 rank JSONL 离线计算，未读取 W&B Metrics；本实验的 W&B 模式为 disabled。MFU 估算真实 token 的 forward+backward，不含 AC 重计算和 padding。HFU 估算额外计入 full AC 层重计算，以及物理 UND/GEN padding 的逐 token 运算和独立 pad attention 段；两者都不是硬件计数器实测。分母均为四卡 BF16 峰值乘五步最慢 rank 总 wall。nsys 比例是每张 GPU 主计算 stream 的 kernel duration 比例，非 wall 比例。",
        "",
        "| 场景 | N | 物理总长 | 真实总长/pack | 平均真实 sample 长度 | step s | MFU % | HFU 估算 % | attention % | GEMM % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(row: dict, key: str, digits: int = 1) -> str:
        return f"{row[key]:.{digits}f}" if key in row else "—"

    for row in rows:
        valid = row["status"] == "completed"
        real_total = fmt(row, "real_tokens_per_pack", 0) if valid else fmt(row, "preflight_real_total_seqlen", 0)
        real_mean = fmt(row, "mean_real_sample_length") if valid else fmt(row, "preflight_mean_real_sample_length")
        lines.append(
            f"| {row['scene']} | {row['samples_per_pack']} | {row['physical_total_seqlen']} | "
            f"{real_total} | {real_mean} | "
            f"{fmt(row, 'seconds_per_step', 3) if valid else '—'} | "
            f"{fmt(row, 'estimated_useful_mfu_pct', 2) if valid else '—'} | "
            f"{fmt(row, 'estimated_hfu_proxy_pct', 2) if valid else '—'} | "
            f"{fmt(row, 'attention_kernel_pct_mean', 2) if valid else '—'} | "
            f"{fmt(row, 'gemm_kernel_pct_mean', 2) if valid else '—'} |"
        )
    lines += [
        "",
        "逐组数据见 [results.csv](../../outputs/v0.2.6_seqlen_efficiency/results.csv)；各 GPU 的 attention/GEMM 秒数、占比范围、top kernels 在输出目录 `runs/<组>/nsys/kernels.json`；逐 rank/step FLOPs 在同组 `mfu.json`，逐 microbatch 长度和计时在训练目录的 `seqlen_efficiency/rank_*.jsonl`。",
        "",
        "单轮合成数据结果仅用于当前配置的趋势判断。未完成组的真实总长和单样本长度来自 CPU packer preflight，其性能列保留空白；有效性状态见 CSV。",
        "",
    ]
    for scene in scenes:
        completed = [row for row in rows if row["scene"] == scene and row["status"] == "completed"]
        if len(completed) < 2:
            continue
        steps = [row["seconds_per_step"] for row in completed]
        mfus = [row["estimated_useful_mfu_pct"] for row in completed]
        hfus = [row["estimated_hfu_proxy_pct"] for row in completed]
        attn = [row["attention_kernel_pct_mean"] for row in completed]
        gemm = [row["gemm_kernel_pct_mean"] for row in completed]
        lines.append(
            f"{scene} 当前 {len(completed)} 组有效：完整 step wall 范围 {min(steps):.3f}–{max(steps):.3f} s "
            f"（最大/最小 {max(steps) / min(steps):.2f}×），有效 MFU {min(mfus):.2f}–{max(mfus):.2f}%，"
            f"HFU 估算 {min(hfus):.2f}–{max(hfus):.2f}%，attention {min(attn):.2f}–{max(attn):.2f}%，"
            f"GEMM {min(gemm):.2f}–{max(gemm):.2f}%。"
        )
    t2v = {row["samples_per_pack"]: row for row in rows if row["scene"] == "t2v" and row["status"] == "completed"}
    if scenes == ("t2v",) and len(t2v) == len(SAMPLE_COUNTS):
        one, middle, many = (t2v[n] for n in (1, 64, 256))
        lines += [
            "",
            "## 结论",
            "",
            "- **为什么 s/step 先降后增** — N=1→16，每个 GEN token 要看的上下文从 49,699 缩至 3,097，"
            "不同样本之间不做 attention；总真实 token 接近不变，真实 attention 工作却减少约 16 倍。"
            "GEMM 仍处理固定的 65,536 个物理 token，耗时基本不变。"
            "attention 每微批次省约 9.65 秒，GA=2 对应每 step 约 19.30 秒，与 33.17→13.80 秒的降幅接近。"
            "继续拆短后，这项收益变小，padding attention 和反向调度开销却增加（见第三条），于是 step 回升。",
            "- **为什么 MFU 与 HFU 走势不同** — 少做 attention 后 step 更快，但真实模型 FLOPs 减少得更多，"
            "所以 MFU 可以下降；这不代表 token 吞吐下降。按本报告口径，"
            "HFU = MFU + 真实 token 重计算项 + padding 前反向及重计算项。"
            "N=1→16，padding FLOPs/step 几乎不变，却随 attention 减少而被摊到更短的 step 中，"
            "其 HFU 贡献从 8.16 增至 19.90 个百分点，盖过 MFU 和真实 token 重计算项的下降，"
            "因此 HFU 先升。短样本端额外开销推高耗时，却没有相应增加解析 FLOPs，HFU 又下降。"
            "[分项对账](profile_recheck.md#mfu-与-hfu-差额的精确口径)保留具体数值。",
            "- **为什么 attention/GEMM kernel 占比不单调** — 拆短最初减少 attention 运算；"
            "但当前实现把 FA3 的单序列最大长度固定为整个 pack，反向为每个短样本也安排大量无对应 token 的工作块。"
            "N=64→256，GEN 主反向每次启动的工作块从 1,064,960 增至 4,210,688，kernel 调用次数不变；"
            "反向预处理及梯度转换由 0.215 升至 0.715 秒/微批次。再加上独立 padding 段变长，attention 时间反弹。"
            "GEMM 约 4.5 秒基本不变，因此 attention 占比先降后升，GEMM 占比反向变化；"
            "GEMM 占比下降不代表 GEMM 自身变慢。"
            "[代码与 nsys 证据](kernel_cause_recheck.md)列出调用链、grid 和分项耗时；尚未用对照实验分离两种开销的贡献。",
            "",
            "## 口径与限制",
            "",
            "四张 H100（GPU0–3），每组一次。step 为第 4–8 步无 profiler 的四卡最慢 rank 完整计时；"
            "kernel 秒/微批次为第 9–10 步 nsys 中，每卡同类 kernel 累计时间除以两步共 4 个微批次后取四卡平均，"
            "两种秒数不能直接相减。MFU/HFU 均为离线 FLOPs 估算，不是硬件计数器；"
            "此前 W&B `mfu/H100` 使用不同口径，详见 [口径复核](profile_recheck.md)。",
            "",
            "输入是合成 offline latent；九组各测一次，真实总长 48,992–53,056，"
            "N=1/2/4 的画幅比例更宽。结果只支持此配置的趋势。Action 场景按要求停止，未纳入报告。",
        ]
    if any(row["status"] == "completed" for row in rows):
        lines.append("")
    if sum(row["status"] == "completed" for row in rows) >= 4:
        lines.append(
            "[四联图](seqlen_efficiency.png)（另有 [SVG 版本](seqlen_efficiency.svg)）；"
            "[历史测试与当前 profile 复核](profile_recheck.md)。"
        )
        lines.append("")
    DELIVERY.mkdir(parents=True, exist_ok=True)
    (DELIVERY / "report.md").write_text("\n".join(lines))
    plot_completed(rows, DELIVERY)
    print(f"Wrote {root / 'results.csv'} and {DELIVERY / 'report.md'}")


if __name__ == "__main__":
    main()
