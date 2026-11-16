"""Evaluate a saved small-action checkpoint on a prefix of the frozen episodes."""

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/action_tower_1b"))
    p.add_argument("--presets", nargs="+", default=["depth_1b", "width_heads_1b", "width_head_dim_1b", "width_only_1b"])
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--sim-python", default="/tmp/cosmos3-libero-env-20260918/bin/python")
    p.add_argument("--checkpoint-step", type=int, choices=range(200, 1001, 200), default=1000)
    p.add_argument("--episodes", type=int, choices=range(1, 6), default=2)
    p.add_argument("--workers-per-preset", type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    if len(args.gpus) != args.workers_per_preset * len(args.presets):
        p.error("Assign workers-per-preset independent GPUs to every preset")
    repo = Path.cwd()
    root = args.root.resolve()
    py = str(repo / ".venv/bin/python")
    fixture = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_bowl_plate_eval5")
    data = fixture.parent / "libero_goal_bowl_plate_v1"
    sim = fixture.parent / "libero_sim"
    stop = threading.Event()
    lock = threading.Lock()
    suffix = "" if args.checkpoint_step == 1000 else f"_step{args.checkpoint_step}"
    manifest = dict(
        jobs=[],
        completed=[],
        baseline_reused=True,
        global_batch_size=16,
        checkpoint_step=args.checkpoint_step,
        episodes=args.episodes,
    )
    path = root / ("eval_launch" + suffix + "_" + "_".join(args.presets) + ".json")

    def save():
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    def launch(cmd, env, log, kind, preset):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, cwd=repo, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT)
        with lock:
            manifest["jobs"].append(
                dict(command=cmd, environment=env, log=str(log), kind=kind, preset=preset, pid=proc.pid)
            )
            save()
        return proc

    def wait(proc):
        while proc.poll() is None:
            if stop.wait(1):
                proc.terminate()
                proc.wait(timeout=60)
                raise RuntimeError("Another evaluation worker failed")
        if proc.returncode:
            raise RuntimeError(f"Evaluation PID {proc.pid} exited {proc.returncode}")

    def worker(i):
        preset = args.presets[i // args.workers_per_preset]
        gpu = args.gpus[i]
        shard = i % args.workers_per_preset
        run = root / "train/cosmos3_action/action_tower_v022_libero_1b" / f"libero_bowl_plate_{preset}"
        checkpoint = run / f"checkpoints/iter_{args.checkpoint_step:09d}"
        cfg = run / "config.yaml"
        assert (checkpoint / "model/.metadata").exists(), checkpoint
        dest = root / ("eval" + suffix) / preset
        env = dict(
            LD_LIBRARY_PATH="",
            PYTHONPATH=str(repo),
            CUDA_VISIBLE_DEVICES=str(gpu),
            MASTER_PORT=str(29800 + gpu),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONHASHSEED="42",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        port = 8140 + gpu
        server = None
        try:
            if shard == 0:
                cmd = [
                    py,
                    "tools/validate_action_tower.py",
                    "capture",
                    "--checkpoint",
                    str(checkpoint / "model"),
                    "--config-file",
                    str(cfg),
                    "--initialization",
                    "checkpoint",
                    "--catalog",
                    str(data / "val.parquet"),
                    "--batch-file",
                    str(fixture / "eval_batches.pt"),
                    "--vae",
                    "/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth",
                    "--tokenizer",
                    "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct",
                    "--towers",
                    "three_tower",
                    "--execution",
                    "native",
                    "--attention-backend",
                    "sdpa",
                    "--samples",
                    str(args.episodes),
                    "--steps",
                    "35",
                    "--guidance",
                    "1.5",
                    "--shift",
                    "5",
                    "--seed",
                    "42",
                    "--decode-video",
                    "--output",
                    str(dest / "capture.pt"),
                ]
                wait(launch(cmd, env, dest / "capture.log", "offline", preset))
            cmd = [
                py,
                "-m",
                "cosmos_framework.scripts.action_policy_server_libero",
                "--checkpoint-path",
                str(checkpoint),
                "--config-file",
                str(cfg),
                "--experiment-overrides",
                "model.config.action_tower_init=checkpoint",
                "model.config.parallelism.data_parallel_shard_degree=1",
                "model.config.parallelism.data_parallel_replicate_degree=1",
                "--action-normalization",
                "quantile_rot",
                "--action-stats-path",
                "cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json",
                "--raw-action-dim",
                "10",
                "--action-chunk-size",
                "16",
                "--fps",
                "20",
                "--seed",
                "42",
                "--guidance",
                "1.5",
                "--num-steps",
                "35",
                "--plain-task-prompt",
                "--no-guardrails",
                "--no-use-torch-compile",
                "--no-use-ema-weights",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--output-dir",
                str(dest / f"server_{shard}"),
            ]
            server = launch(cmd, env, dest / f"server_{shard}.log", "server", preset)
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"{preset} server failed")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=3) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline or stop.wait(2):
                    raise RuntimeError("Server startup timed out or another worker failed")
            simenv = env | dict(
                LD_LIBRARY_PATH=str(sim / "systemlibs/usr/lib/x86_64-linux-gnu"),
                LIBERO_CONFIG_PATH=str(sim / "config"),
                MUJOCO_GL="osmesa",
                PYOPENGL_PLATFORM="osmesa",
                PYTHONPATH=f"{repo}:{sim / 'LIBERO'}",
                CUDA_VISIBLE_DEVICES="",
            )
            for trial in range(shard, args.episodes, args.workers_per_preset):
                cmd = [
                    args.sim_python,
                    "-m",
                    "cosmos_framework.simulation.libero.paired_trial_eval",
                    "--server-url",
                    f"http://127.0.0.1:{port}",
                    "--trials",
                    str(trial),
                    "--seed",
                    "42",
                    "--max-steps",
                    "1000",
                    "--warmup-steps",
                    "0",
                    "--action-only",
                    "--initial-states",
                    str(fixture / "episodes.json"),
                    "--output",
                    str(dest / "closed_loop" / f"shard{shard}"),
                ]
                wait(launch(cmd, simenv, dest / f"trial_{trial}.log", "simulation", preset))
                with lock:
                    manifest["completed"].append(dict(preset=preset, trial=trial))
                    save()
                print(preset, "trial", trial, "complete", flush=True)
        except Exception:
            stop.set()
            raise
        finally:
            if server and server.poll() is None:
                server.terminate()
                server.wait(timeout=60)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, range(len(args.gpus))))
    manifest["all_complete"] = True
    save()


if __name__ == "__main__":
    main()
