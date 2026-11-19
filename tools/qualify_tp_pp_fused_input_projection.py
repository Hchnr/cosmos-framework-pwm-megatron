# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run a small production-network oracle with opt-in local projection fusion."""

import argparse
import functools
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--fusion", choices=("qkv", "gate_up", "both"), default="both")
    parser.add_argument("--report-dir", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from tools.tp_pp_fused_input_projection import FusionObservation, install

    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp
    )
    qkv, gate_up = args.fusion in ("qkv", "both"), args.fusion in ("gate_up", "both")
    observation = FusionObservation(args.report_dir, os.environ.get("RANK", ""), fuse_qkv=qkv, fuse_gate_up=gate_up)
    try:
        install(fuse_qkv=qkv, fuse_gate_up=gate_up, observation=observation)
        sys.argv = [sys.argv[0], *remainder]
        smoke.main()
    except BaseException as error:
        observation.finish(error=f"{type(error).__name__}: {error}")
        raise
    observation.finish()


if __name__ == "__main__":
    main()
