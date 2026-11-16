# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Sequential jobs: exclusive use of eight GPUs, immutable per-run directories."""

import argparse
import json
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
parser.add_argument("--stage", choices=["smoke", "bench", "profile"], required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--gen-tokens", type=int, default=98304)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--resume", action="store_true", help="Validate and skip completed runs; never overwrite")
args = parser.parse_args()
for round_id in range(args.rounds if args.stage == "bench" else 1):
    order = args.backends[round_id:] + args.backends[:round_id]
    for backend in order:
        name = f"{args.tag}_{args.stage}_r{round_id}_{backend}"
        command = [
            sys.executable,
            str(ROOT / "tools/unified_attn/run.py"),
            "--backend",
            backend,
            "--name",
            name,
            "--gen-tokens",
            str(args.gen_tokens),
            "--warmup",
            str(0 if args.stage == "smoke" else args.warmup),
        ]
        if args.stage == "profile":
            command += ["--nsys"]
        else:
            command += ["--steps", str(2 if args.stage == "smoke" else 30)]
            if round_id == 0 or backend == "split_default":
                command += ["--audit"]
        run = ROOT / "plans/v0.2.7_unified_attn/artifacts/runs" / name
        if run.exists():
            if not args.resume:
                raise RuntimeError(f"Refusing to reuse run {run}; choose a fresh --tag")
            saved = json.loads((run / "manifest.json").read_text())["args"]
            assert saved["backend"] == backend and saved["gen_tokens"] == args.gen_tokens
            assert saved["warmup"] == (0 if args.stage == "smoke" else args.warmup)
            assert saved["nsys"] == (args.stage == "profile")
            assert saved["steps"] == (2 if args.stage == "smoke" else 30)
            subprocess.run(
                [sys.executable, str(ROOT / "tools/unified_attn/summarize.py"), str(run)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            print("VERIFIED_EXISTING", name, flush=True)
            continue
        print("START", name, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        with (ROOT / "plans/v0.2.7_unified_attn/record.md").open("a") as record:
            record.write(f"\n- 自动实验 `{name}`：exit={result.returncode}；完整证据 `artifacts/runs/{name}/`。")
        if result.returncode:
            raise SystemExit(result.returncode)
        subprocess.run(
            [sys.executable, str(ROOT / "tools/unified_attn/summarize.py"), str(run)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print("DONE", name, json.loads((run / "summary.json").read_text())["step_median_s"], flush=True)
