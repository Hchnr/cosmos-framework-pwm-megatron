# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run kernel comparisons in separate processes to avoid FA3 namespace collisions."""

import argparse
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
parser.add_argument("--geometry", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gen-tokens", type=int, default=98304)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
for frozen in (False, True):
    ref = args.output / f"reference_frozen{int(frozen)}"
    for backend in BACKENDS:
        name = f"{backend}_frozen{int(frozen)}"
        output = args.output / name
        env = os.environ.copy()
        env.update(
            LD_LIBRARY_PATH="",
            OMP_NUM_THREADS="2",
            PYTHONPATH=str(ROOT),
            TORCHINDUCTOR_CACHE_DIR=f"/tmp/cosmos-unified-attn/kernel-{args.output.name}/{name}",
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
            "tools/unified_attn/kernel_bench.py",
            "--backend",
            backend,
            "--geometry",
            str(args.geometry),
            "--output",
            str(output),
            "--gen-tokens",
            str(args.gen_tokens),
            "--reference",
            str(ref),
        ]
        if frozen:
            command += ["--frozen-und"]
        if backend == "split_default":
            command += ["--write-reference"]
        print("START", name, flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print("END", name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
