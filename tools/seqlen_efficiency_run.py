#!/usr/bin/env python3
"""Run the nine T2V fixed-pack training/profile groups on four GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.seqlen_efficiency_data import SAMPLE_COUNTS

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "t2v": "/share/project/eai_pwm/models/Cosmos3-Nano-DCP",
    "action": "/share/project/eai_pwm/models/Cosmos3-Nano-action-1b-v022/width_only_1b",
}
RECORD = ROOT / "plans/v0.2.6_seqlen_efficiency/record.md"
GPU_INDICES = (0, 1, 2, 3)


def gpu_status() -> list[dict]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in raw.splitlines():
        index, memory, utilization = (int(part.strip()) for part in line.split(","))
        rows.append({"index": index, "memory_mib": memory, "utilization": utilization})
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_gpus(poll_seconds: int, log_path: Path, gpu_indices: tuple[int, ...]) -> None:
    while True:
        rows = gpu_status()
        stamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a") as stream:
            stream.write(json.dumps({"utc": stamp, "gpus": rows}) + "\n")
        selected = [row for row in rows if row["index"] in gpu_indices]
        if len(selected) == len(gpu_indices) and all(
            row["memory_mib"] < 1000 and row["utilization"] < 10 for row in selected
        ):
            return
        print(f"{stamp}: GPUs occupied, next poll in {poll_seconds}s: {rows}", flush=True)
        time.sleep(poll_seconds)


def command_for(scene: str, n: int, path: Path, report: Path, master_port: int) -> list[str]:
    tail = [
        "--",
        "trainer.seed=42",
        "trainer.profiling.enable_nsys=true",
        "trainer.profiling.profile_freq=10",
        "trainer.profiling.profile_warmup=0",
        "trainer.profiling.profile_active=2",
        "trainer.profiling.target_ranks=[0,1,2,3]",
        "++trainer.callbacks.seqlen_efficiency._target_=cosmos_framework.callbacks.seqlen_efficiency_benchmark.SeqlenEfficiencyBenchmark",
        f"++trainer.callbacks.seqlen_efficiency.expected_samples_per_pack={n}",
    ]
    if scene == "t2v":
        tail += [
            "+model.config.vlm_config.model_instance.config.include_visual=true",
            "model.config.vlm_config.tokenizer.pretrained_model_name=/share/project/eai_pwm/models/Cosmos3-Nano",
        ]
    return [
        "nsys",
        "profile",
        f"--output={report}",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt,cudnn,cublas",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--stats=false",
        "--wait=all",
        str(ROOT / ".venv/bin/torchrun"),
        "--nproc-per-node=4",
        f"--master-port={master_port}",
        "-m",
        "cosmos_framework.scripts.train",
        f"--sft-toml={path}",
        *tail,
    ]


def record_attempt(root: Path, key: str, returncode: int, *, commit: bool = True) -> None:
    report_log = root / "runs" / key / "analysis.log"
    with report_log.open("w") as stream:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/seqlen_efficiency_report.py"), "--root", str(root)],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    row = None
    if result.returncode == 0:
        with (root / "results.csv").open() as stream:
            row = next(
                (
                    entry
                    for entry in csv.DictReader(stream)
                    if f"{entry['scene']}_n{int(entry['samples_per_pack']):03d}" == key
                ),
                None,
            )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = row["status"] if row else f"report_error_{result.returncode}"
    fields = []
    if row and status == "completed":
        fields = [
            f"step={float(row['seconds_per_step']):.3f}s",
            f"MFU={float(row['estimated_useful_mfu_pct']):.2f}%",
            f"attention={float(row['attention_kernel_pct_mean']):.2f}%",
            f"GEMM={float(row['gemm_kernel_pct_mean']):.2f}%",
        ]
    detail = ", ".join(fields)
    with RECORD.open("a") as stream:
        evidence = (root / "runs" / key).relative_to(ROOT)
        stream.write(
            f"\n- {stamp} {key} 一次性训练结束：exit={returncode}，分析={status}"
            f"{', ' + detail if detail else ''}；证据见 `{evidence}/`。\n"
        )
    if commit:
        subprocess.run(["git", "add", "-f", str(RECORD)], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"docs: record seqlen study attempt {key}"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/v0.2.6_seqlen_efficiency")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--scenes", nargs="+", choices=("t2v", "action"), default=["t2v"])
    parser.add_argument("--samples", nargs="+", type=int, default=list(SAMPLE_COUNTS))
    parser.add_argument("--gpu-indices", nargs=4, type=int, default=list(GPU_INDICES))
    parser.add_argument("--master-port", type=int, default=29674)
    parser.add_argument("--skip-record-commit", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or any(n not in SAMPLE_COUNTS for n in args.samples):
        parser.error("invalid poll interval or sample count")
    gpu_indices = tuple(args.gpu_indices)
    if len(set(gpu_indices)) != 4 or min(gpu_indices) < 0:
        parser.error("--gpu-indices must contain four distinct nonnegative indices")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        LD_LIBRARY_PATH="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONHASHSEED="42",
        WAN_VAE_PATH="/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth",
        HF_MODEL_PATH="/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        COSMOS_COMPILE_CACHE_PER_RANK="1",
        PYTHONPATH=str(ROOT),
        CUDA_VISIBLE_DEVICES=",".join(map(str, gpu_indices)),
    )
    for scene in args.scenes:
        for n in args.samples:
            key = f"{scene}_n{n:03d}"
            run = root / "runs" / key
            run.mkdir(parents=True, exist_ok=True)
            status = run / "status.json"
            if status.exists():
                print(f"Existing status, skipping completed attempt: {status}", flush=True)
                continue
            if (run / "launch.json").exists():
                raise RuntimeError(f"Unfinished attempt requires inspection before continuing: {run}")
            wait_for_gpus(args.poll_seconds, root / "gpu_polls.jsonl", gpu_indices)
            path = root / "configs" / f"{key}.toml"
            report = run / "nsys" / key
            report.parent.mkdir(parents=True, exist_ok=True)
            cmd = command_for(scene, n, path, report, args.master_port)
            env["BASE_CHECKPOINT_PATH"] = CHECKPOINTS[scene]
            env["IMAGINAIRE_OUTPUT_ROOT"] = str(run / "train")
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
            (run / "git_status.txt").write_text(git_status)
            (run / "git_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
            with (run / "gpu_info.txt").open("w") as stream:
                subprocess.run(["nvidia-smi", "-q"], stdout=stream, check=True)
            manifest = json.loads((root / "inputs" / scene / f"n{n:03d}" / "manifest.study.json").read_text())
            info = {
                "group": key,
                "gpu_indices": gpu_indices,
                "command": cmd,
                "git_revision": revision,
                "checkpoint": CHECKPOINTS[scene],
                "checkpoint_metadata_sha256": sha256(Path(CHECKPOINTS[scene]) / "model/.metadata"),
                "dataset_manifest": manifest,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            (run / "launch.json").write_text(json.dumps(info, indent=2) + "\n")
            print(f"Starting {key} at {info['started_utc']}", flush=True)
            with (run / "train.log").open("w") as stream:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            info.update(returncode=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
            status.write_text(json.dumps(info, indent=2) + "\n")
            print(f"Finished {key}: exit={result.returncode}", flush=True)
            record_attempt(root, key, result.returncode, commit=not args.skip_record_commit)
            if result.returncode:
                # Stop on the first failure so a shared setup problem cannot
                # consume all 18 one-attempt groups.
                return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
