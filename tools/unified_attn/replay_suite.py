# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Replay real activations in isolated backend processes on all eight ranks."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = [
    "split_default",
    "split_fa3_fork",
    "unified_fa3_grouped",
    "split_fa4",
    "unified_fa4_mask",
    "split_flex_triton",
    "unified_flex_triton",
]
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
summary = {}
for backend in BACKENDS:
    output = args.output / backend
    env = os.environ.copy()
    env.update(
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="2",
        PYTHONPATH=str(ROOT),
        TORCHINDUCTOR_CACHE_DIR=f"/tmp/cosmos-unified-attn/replay-{args.output.name}/{backend}",
    )
    if backend in ("split_fa3_fork", "unified_fa3_grouped"):
        env["COSMOS_FA3_PROVIDER"] = "fork"
        env["PYTHONPATH"] += ":" + str(ROOT / "plans/v0.2.7_unified_attn/artifacts/deps/hopper")
    else:
        env.pop("COSMOS_FA3_PROVIDER", None)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "tools/unified_attn/replay.py",
        "--run",
        str(args.run),
        "--backend",
        backend,
        "--output",
        str(output),
    ]
    if backend == "split_default":
        command += ["--write-reference"]
    print("START", backend, flush=True)
    with (args.output / f"{backend}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print("END", backend, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    ranks = [json.loads((output / f"rank{rank}.json").read_text()) for rank in range(8)]
    summary[backend] = {
        "max_relative_l2": max(
            metric["relative_l2"] for rank in ranks for layer in rank["errors"].values() for metric in layer.values()
        ),
        "ranks": ranks,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
