# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run an opt-in eager FP32 TP K/V reconstruction diagnostic, never a benchmark."""

import argparse
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--full-ac", action="store_true")
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.tp_pp_fp32_reconstructed_kv import Observation, install

    observation = Observation(args.report_dir, os.environ.get("RANK", ""))
    try:
        install(observation)
        if args.full_ac:
            from tools.run_tp_pp_wide_linear_math import install_tp_checkpointing

            install_tp_checkpointing()
        sys.path.insert(0, str(args.entry.resolve().parent))
        sys.argv = [str(args.entry), *remainder]
        runpy.run_path(str(args.entry), run_name="__main__")
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise
    observation.finish()


if __name__ == "__main__":
    main()
