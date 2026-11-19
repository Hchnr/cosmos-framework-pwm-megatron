# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run an unchanged GPU test or benchmark with shared-input TP SUMs enabled."""

import argparse
import runpy
import sys
from pathlib import Path

from tools.tp_pp_shared_input_reduction import install


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    install()
    sys.path.insert(0, str(args.run_path.resolve().parent))
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")


if __name__ == "__main__":
    main()
