# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reuse the production-network four-update oracle at a selected TP/PP topology."""

import argparse
import functools
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from tools.tp_pp_shared_input_reduction import install

    # Replace only this test module's constructor binding; production topology
    # types and other modules retain their original definitions.
    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp
    )
    install()
    sys.argv = [sys.argv[0], *remainder]
    smoke.main()


if __name__ == "__main__":
    main()
