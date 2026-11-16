# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Sequential, fail-fast runs; never co-schedule training jobs on the eight GPUs."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "benchmark", "profile"])
    parser.add_argument("--gen-tokens", type=int, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--filter-oversized", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/tmp/cosmos-ac-attn/runs"))
    args = parser.parse_args()
    if args.stage == "audit":
        jobs = [("full", "audit0"), ("full", "audit1"), ("attention_only", "audit0")]
        extra = ["--audit", "--warmup", "0", "--steps", "1"]
    elif args.stage == "benchmark":
        jobs = [
            (mode, f"r{repeat}")
            for repeat in range(3)
            for mode in (("full", "attention_only") if repeat % 2 == 0 else ("attention_only", "full"))
        ]
        extra = ["--warmup", "10", "--steps", "30"]
    else:
        jobs = [("full", "nsys"), ("attention_only", "nsys")]
        extra = ["--warmup", "10", "--nsys"]
    for mode, suffix in jobs:
        name = f"{args.prefix}_{mode}_{suffix}"
        admission = ["--filter-oversized"] if args.filter_oversized else []
        subprocess.run(
            [
                sys.executable,
                "tools/ac_attn/run.py",
                "--ac",
                mode,
                "--name",
                name,
                "--root",
                str(args.root),
                "--gen-tokens",
                str(args.gen_tokens),
                *extra,
                *admission,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "tools/unified_attn/summarize.py",
                str(args.root / name),
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )


if __name__ == "__main__":
    main()
