# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reduce FP32 full-payload oracle MLP peak memory by token-wise chunking.

The same opt-in execution is required for native and TP math runs. It is an
oracle capacity candidate, not the default training backend or a throughput
benchmark. It never changes checkpoint identities or optimizer state.
"""

import argparse
import runpy
import sys
from pathlib import Path

import torch


def install(chunk_tokens):
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP

    if chunk_tokens < 1:
        raise ValueError("Chunk size must be positive")
    original = Qwen3VLTextMLP.forward

    def forward(self, inputs):
        if inputs.ndim != 2:
            raise ValueError("The full-payload oracle requires packed 2-D token activations")
        if inputs.shape[0] <= chunk_tokens:
            return original(self, inputs)
        return torch.cat([original(self, part) for part in inputs.split(chunk_tokens, dim=0)], dim=0)

    Qwen3VLTextMLP.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    args, remainder = parser.parse_known_args()
    install(args.chunk_tokens)
    sys.path.insert(0, str(args.entry.resolve().parent))
    sys.argv = [str(args.entry), *remainder]
    runpy.run_path(str(args.entry), run_name="__main__")


if __name__ == "__main__":
    main()
