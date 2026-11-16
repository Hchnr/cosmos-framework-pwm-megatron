# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sequential real Action128 training A/B; use an isolated output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import psutil


def _gpu_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    rows = [line.split(", ") for line in output.splitlines()]
    return {int(index): {"uuid": uuid, "used_mib": int(used)} for index, uuid, used in rows}


def _stop_experiment(process):
    # torchrun ranks create their own sessions. Killing only the launcher's
    # process group would leave those ranks and their data workers alive.
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--devices", help="Physical CUDA device IDs, e.g. 1,2,3,4,5,6,7")
    parser.add_argument("--suite", choices=["original", "k-pin"], default="original")
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--workers", type=int, default=23, help="Shared physical workers; legacy reference stays at 23")
    parser.add_argument("--episode-columns", choices=["all", "projected"], nargs="+", default=["projected"])
    parser.add_argument("--cases", nargs="+", help="k-pin case labels, e.g. legacy, background_k16, pool_w4_k16")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--production-compile", action="store_true", help="Keep recipe compile/VAE AOT settings")
    parser.add_argument("--gpu-preflight-max-used-mib", type=int, default=256)
    parser.add_argument("--gpu-preflight-timeout", type=int, default=600)
    parser.add_argument("--gpu-preflight-stable-seconds", type=int, default=15)
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")] if args.devices else list(range(args.gpus))
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("devices must be distinct nonnegative IDs")
    if args.steps < 1 or args.repeats < 1 or args.workers < 1 or any(depth < 1 for depth in args.depths):
        parser.error("steps, repeats and depths must be positive")
    args.gpus = len(devices)
    inventory = _gpu_inventory()
    device_uuids = [inventory[device]["uuid"] for device in devices]
    root = Path.cwd()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("all", "legacy_per_source", 23, 2, "background"),
        ("all", "shared_prefetch", 23, 8, "background"),
        ("projected", "legacy_per_source", 23, 2, "background"),
        ("projected", "shared_prefetch", 23, 8, "background"),
        ("projected", "shared_prefetch", 2, 1, "background"),
    ]
    if args.suite == "k-pin":
        configurations = {
            "legacy": ("legacy_per_source", 23, 2, "background"),
            "inline_k8": ("shared_prefetch", args.workers, 8, "inline"),
            **{f"background_k{depth}": ("shared_prefetch", args.workers, depth, "background") for depth in args.depths},
            "pool_w4_k16": ("shared_prefetch", 4, 16, "background"),
            "pool_w8_k16": ("shared_prefetch", 8, 16, "background"),
        }
        selected = args.cases or ["legacy", "inline_k8", *(f"background_k{depth}" for depth in args.depths)]
        if set(selected) - configurations.keys():
            parser.error(f"Unknown cases: {set(selected) - configurations.keys()}")
        cases = [(columns, *configurations[name]) for columns in args.episode_columns for name in selected]
    if args.pilot:
        cases = [("projected", "shared_prefetch", 23, 8, "background")]
    sources = [
        "cosmos_framework/data/generator/dataflow/shared_prefetch.py",
        "tools/action128_prefetch_training_benchmark.py",
        "tools/run_action128_gpu_prefetch_ablation.py",
        "examples/toml/sft_config/action128_wam_nano_160gpu.toml",
    ]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "devices": devices,
                "device_uuids": device_uuids,
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    manifest = []
    runs = [
        (repeat, case) for repeat in range(args.repeats) for case in (cases if repeat % 2 == 0 else reversed(cases))
    ]
    for index, (repeat, (columns, backend, workers, depth, pin_mode)) in enumerate(runs):
        if hashes != {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}:
            raise RuntimeError("Benchmark source changed during the matrix; use a fresh output directory")
        name = f"{columns}_{backend}_w{workers}_k{depth}"
        if args.suite == "k-pin" or args.repeats > 1:
            name += f"_{pin_mode}_r{repeat}"
        output = (args.output / name).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if (output / "training_benchmark").exists():
            raise RuntimeError(f"Refusing to append to existing training records: {output}")
        waiting = time.monotonic()
        free_since = None
        while True:
            preflight = _gpu_inventory()
            busy = {
                device: preflight[device]["used_mib"]
                for device in devices
                if preflight[device]["used_mib"] > args.gpu_preflight_max_used_mib
            }
            if not busy:
                if free_since is None:
                    free_since = time.monotonic()
                if time.monotonic() - free_since >= args.gpu_preflight_stable_seconds:
                    break
            else:
                free_since = None
            if time.monotonic() - waiting > args.gpu_preflight_timeout:
                raise RuntimeError(f"Selected GPUs remain occupied: {busy}; no training process was launched")
            if busy:
                print(f"WAIT occupied GPUs: {busy}", flush=True)
            time.sleep(5)
        overrides = {
            "NNODES": "1",
            "NPROC_PER_NODE": str(args.gpus),
            "DP_SHARD": str(args.gpus),
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29811 + index),
            "VIRTUAL_ENV": str(root / ".venv"),
            "WANDB_MODE": "disabled",
            "WANDB_API_KEY_FILE": "/nonexistent",
            "OUTPUT_ROOT": str(output),
            "IMAGINAIRE_OUTPUT_ROOT": str(output),
            "SFT_TOML_FILE": str(root / "examples/toml/sft_config/action128_wam_nano_160gpu.toml"),
            "SFT_TRAINING_MODULE": "tools.action128_prefetch_training_benchmark",
            "MAX_ITER": str(args.steps),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ACTION128_BENCH_EPISODE_COLUMNS": columns,
            "ACTION128_BENCH_PIN_MEMORY_MODE": pin_mode,
            "CUDA_VISIBLE_DEVICES": ",".join(device_uuids),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "LD_LIBRARY_PATH_OVERRIDE": os.environ.get("LD_LIBRARY_PATH", ""),
            "EXTRA_TAIL_OVERRIDES": " ".join(
                [
                    f"job.name=prefetch_gpu_{name}",
                    *(
                        [
                            "trainer.compile_cache.enabled=false",
                            "model.config.compile.enabled=false",
                            "trainer.callbacks.compile_tokenizer.enabled=false",
                        ]
                        if not args.production_compile
                        else []
                    ),
                    "trainer.save_zero_checkpoint=false",
                    f"dataloader_train.execution_backend={backend}",
                    f"dataloader_train.num_workers={workers}",
                    f"dataloader_train.prefetch_factor={depth}",
                    "dataloader_train.logical_workers=[" + ",".join(["1"] * 23) + "]",
                ]
            ),
        }
        if args.production_compile:
            # Both fresh processes must start with the same cache state. A
            # later candidate must not inherit the reference's compiled code.
            # Triton also needs local filesystem rename/unlink semantics.
            compiler_cache = Path(tempfile.mkdtemp(prefix="action128-prefetch-", dir="/tmp"))
            overrides["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache / "inductor")
            overrides["TRITON_CACHE_DIR"] = str(compiler_cache / "triton")
        environment = os.environ.copy()
        environment.update(overrides)
        environment["PATH"] = str(root / ".venv/bin") + os.pathsep + environment["PATH"]
        environment.pop("WANDB_API_KEY", None)
        command = ["bash", "examples/launch_sft_action128_wam_nano_160gpu.sh"]
        (output / "command.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "overrides": overrides,
                    "gpu_preflight": {device: preflight[device] for device in devices},
                    "model_paths": {
                        key: environment.get(key) for key in ("HF_MODEL_PATH", "WAN_VAE_PATH", "BASE_CHECKPOINT_PATH")
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"START {name}", flush=True)
        started = time.monotonic()
        memory_guard = False
        timed_out = False
        fatal_error = None
        last_sample = 0
        with (output / "launcher.log").open("w") as log, (args.output / "telemetry.jsonl").open("a") as telemetry:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                with (output / "launcher.log").open("rb") as errors:
                    errors.seek(0, os.SEEK_END)
                    errors.seek(max(0, errors.tell() - 65536))
                    tail = errors.read()
                    if b"torch.OutOfMemoryError:" in tail or b"torch._inductor.exc.InductorError:" in tail:
                        fatal_error = "CUDA OOM" if b"torch.OutOfMemoryError:" in tail else "Inductor error"
                        _stop_experiment(process)
                        break
                # The container includes reclaimable file cache. Protect the
                # experiment at high anonymous/shmem use, before kernel OOM.
                stat = Path("/sys/fs/cgroup/memory.stat")
                limit = Path("/sys/fs/cgroup/memory.max")
                if stat.exists() and limit.exists() and limit.read_text().strip() != "max":
                    values = dict(line.split() for line in stat.read_text().splitlines())
                    resident = int(values.get("anon", 0)) + int(values.get("shmem", 0))
                    if resident > int(limit.read_text()) * 0.90:
                        memory_guard = True
                        _stop_experiment(process)
                        break
                    if time.monotonic() - last_sample >= 2:
                        observation_time = time.time_ns()
                        telemetry_error = None
                        try:
                            query = subprocess.run(
                                [
                                    "nvidia-smi",
                                    f"--id={overrides['CUDA_VISIBLE_DEVICES']}",
                                    "--query-gpu=index,utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits",
                                ],
                                text=True,
                                capture_output=True,
                                timeout=10,
                            )
                            gpus = [line.split(", ") for line in query.stdout.splitlines()]
                            if query.returncode:
                                telemetry_error = query.stderr.strip()
                        except (subprocess.TimeoutExpired, OSError) as error:
                            gpus = []
                            telemetry_error = str(error)
                        telemetry.write(
                            json.dumps(
                                {
                                    "timestamp_ns": observation_time,
                                    "case": name,
                                    "cgroup_current": int(Path("/sys/fs/cgroup/memory.current").read_text()),
                                    **{key: int(values.get(key, 0)) for key in ("anon", "file", "shmem")},
                                    "gpus": gpus,
                                    "gpu_query_error": telemetry_error,
                                }
                            )
                            + "\n"
                        )
                        telemetry.flush()
                        last_sample = time.monotonic()
                if time.monotonic() - started > args.timeout:
                    timed_out = True
                    _stop_experiment(process)
                    break
                time.sleep(1)
            process.wait()
        manifest.append(
            {
                "name": name,
                "exit_code": process.returncode,
                "wall_seconds": time.monotonic() - started,
                "memory_guard": memory_guard,
                "timeout": timed_out,
                "fatal_error": fatal_error,
                "repeat": repeat,
                "columns": columns,
                "backend": backend,
                "workers": workers,
                "prefetch": depth,
                "pin_memory_mode": pin_mode,
                "devices": devices,
                "steps": args.steps,
            }
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"DONE {name}: {process.returncode}", flush=True)
        if process.returncode:
            raise RuntimeError(f"Training failed: inspect {output / 'launcher.log'}")


if __name__ == "__main__":
    main()
