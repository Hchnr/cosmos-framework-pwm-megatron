# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify chunked native and TP MLP math against the four-update oracle."""

import argparse
import functools
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    # The tiny oracle uses two tokens per modality. A chunk size of one is
    # necessary to exercise the split path instead of merely testing the shim.
    parser.add_argument("--chunk-tokens", type=int, default=1)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from tools.run_tp_pp_chunked_mlp_math import install

    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig, tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp
    )
    install(args.chunk_tokens)
    sys.argv = [sys.argv[0], *remainder]
    smoke.main()


if __name__ == "__main__":
    main()
