"""Run four aligned five-Episode evaluations using eight independent GPU workers."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/libero_bowl_plate"))
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--sim-python", default="/tmp/cosmos3-libero-env-20260918/bin/python")
    args = parser.parse_args()
    repo = Path.cwd()
    root = args.root.resolve()
    python = str(repo / ".venv/bin/python")
    simroot = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim")
    dataset = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_goal_bowl_plate_v1")
    fixture = args.fixture_root.resolve()
    variants = [(0, "two"), (0, "three"), (1000, "two"), (1000, "three")]
    # Initial policies often exhaust the full budget; measured native three-tower
    # requests are slower. Start these longer jobs first to avoid a late tail.
    jobs = [(step, tower, trial) for step in [0, 1000] for tower in ["three", "two"] for trial in range(5)]
    lock = threading.Lock()
    stop = threading.Event()
    manifest = dict(fixture=str(fixture), max_steps=1000, warmup_steps=0, jobs=[], completed=[])
    manifest_path = root / "eval5_launch_manifest.json"

    def save():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(manifest_path)

    def config(step, tower):
        run = root / f"train4_v2/cosmos3_action/action_tower/libero_bowl_plate_{tower}_tower"
        checkpoint = run / "checkpoints/iter_000001000" if step else Path("/share/project/eai_pwm/models/Cosmos3-Nano")
        initialization = "copy_gen" if step == 0 and tower == "three" else "checkpoint"
        return run / "config.yaml", checkpoint, initialization

    def launch(command, env, log, kind, gpu):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as output:
            proc = subprocess.Popen(
                command, env={**os.environ, **env}, stdout=output, stderr=subprocess.STDOUT, cwd=repo
            )
        with lock:
            manifest["jobs"].append(
                dict(kind=kind, gpu=gpu, pid=proc.pid, command=command, environment=env, log=str(log))
            )
            save()
        return proc

    def checked_wait(proc):
        while proc.poll() is None:
            if stop.wait(1):
                proc.terminate()
                raise RuntimeError("Another evaluation worker failed")
        if proc.returncode:
            raise RuntimeError(f"PID {proc.pid} failed with exit {proc.returncode}; see launch manifest logs")

    def worker(gpu):
        env = dict(
            LD_LIBRARY_PATH="",
            PYTHONPATH=str(repo),
            CUDA_VISIBLE_DEVICES=str(gpu),
            MASTER_PORT=str(29600 + gpu),
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONHASHSEED="42",
        )
        server = None
        current = None
        port = 8040 + gpu
        try:
            if gpu < 4:
                step, tower = variants[gpu]
                cfg, checkpoint, initialization = config(step, tower)
                capture = [
                    python,
                    "tools/validate_action_tower.py",
                    "capture",
                    "--checkpoint",
                    str(checkpoint / "model" if step else checkpoint),
                    "--config-file",
                    str(cfg),
                    "--initialization",
                    initialization,
                    "--catalog",
                    str(dataset / "val.parquet"),
                    "--batch-file",
                    str(fixture / "eval_batches.pt"),
                    "--vae",
                    "/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth",
                    "--tokenizer",
                    "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct",
                    "--towers",
                    tower + "_tower",
                    "--execution",
                    "native",
                    "--projection-mode",
                    "routed",
                    "--samples",
                    "5",
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
                    str(root / f"step{step}/{tower}_tower.pt"),
                ]
                checked_wait(launch(capture, env, root / f"step{step}/capture_{tower}.log", "capture", gpu))
            while not stop.is_set():
                with lock:
                    if not jobs:
                        break
                    index = next((i for i, job in enumerate(jobs) if job[:2] == current), 0)
                    step, tower, trial = jobs.pop(index)
                if current != (step, tower):
                    if server:
                        server.terminate()
                        server.wait(timeout=60)
                    cfg, checkpoint, initialization = config(step, tower)
                    command = [
                        python,
                        "-m",
                        "cosmos_framework.scripts.action_policy_server_libero",
                        "--checkpoint-path",
                        str(checkpoint),
                        "--config-file",
                        str(cfg),
                        "--experiment-overrides",
                        f"model.config.tower_structure={tower}_tower",
                        f"model.config.action_tower_init={initialization}",
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
                        str(root / f"step{step}/server_{tower}_gpu{gpu}"),
                    ]
                    server = launch(command, env, root / f"step{step}/server_{tower}_gpu{gpu}.log", "server", gpu)
                    current = (step, tower)
                    deadline = time.monotonic() + 900
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"Server on GPU {gpu} exited: {server.returncode}")
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=3) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        if time.monotonic() >= deadline or stop.wait(2):
                            raise RuntimeError(f"Server on GPU {gpu} did not become ready")
                simenv = dict(
                    LD_LIBRARY_PATH=str(simroot / "systemlibs/usr/lib/x86_64-linux-gnu"),
                    LIBERO_CONFIG_PATH=str(simroot / "config"),
                    MUJOCO_GL="osmesa",
                    PYOPENGL_PLATFORM="osmesa",
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="1",
                    LP_NUM_THREADS="4",
                    PYTHONPATH=f"{repo}:{simroot / 'LIBERO'}",
                    CUDA_VISIBLE_DEVICES="",
                )
                command = [
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
                    str(root / f"step{step}/closed_loop_{tower}_tower/shard{gpu}"),
                ]
                checked_wait(
                    launch(command, simenv, root / f"step{step}/sim_{tower}_trial{trial}.log", "simulator", gpu)
                )
                with lock:
                    manifest["completed"].append(dict(step=step, tower=tower, trial=trial, gpu=gpu))
                    save()
                print(f"Completed step{step} {tower} trial{trial} on GPU{gpu}", flush=True)
        except Exception:
            stop.set()
            raise
        finally:
            if server and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    manifest["all_complete"] = True
    with lock:
        save()
    print("All four captures and 20 closed-loop evaluations completed.", flush=True)


if __name__ == "__main__":
    main()
