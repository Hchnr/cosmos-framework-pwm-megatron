# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run a TP benchmark with opt-in local projection fusion."""

import argparse
import os
import runpy
import sys
from pathlib import Path

from tools.tp_pp_fused_input_projection import FusionObservation, install


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--fusion", choices=("qkv", "gate_up", "both"), default="both")
    parser.add_argument("--report-dir", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    qkv, gate_up = args.fusion in ("qkv", "both"), args.fusion in ("gate_up", "both")
    observation = FusionObservation(args.report_dir, os.environ.get("RANK", ""), fuse_qkv=qkv, fuse_gate_up=gate_up)
    try:
        install(fuse_qkv=qkv, fuse_gate_up=gate_up, observation=observation)
        sys.path.insert(0, str(args.entry.resolve().parent))
        sys.argv = [str(args.entry), *remainder]
        runpy.run_path(str(args.entry), run_name="__main__")
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise
    observation.finish()


if __name__ == "__main__":
    main()
