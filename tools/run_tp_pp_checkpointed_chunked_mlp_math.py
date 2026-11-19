# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimental per-chunk recomputation for the full-capacity FP32 oracle.

This diagnostic must be applied identically to the native and TP math runs.
It does not modify the training default or prove full-model numerical parity.
"""

import argparse
import runpy
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


def checkpointed_chunks(function, inputs, chunk_tokens):
    if inputs.ndim != 2 or chunk_tokens < 1:
        raise ValueError("Expected packed 2-D activations and a positive chunk size")
    if inputs.shape[0] <= chunk_tokens:
        return function(inputs)
    return torch.cat(
        [checkpoint(function, part, use_reentrant=False) for part in inputs.split(chunk_tokens, dim=0)],
        dim=0,
    )


def install(chunk_tokens):
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP

    if chunk_tokens < 1:
        raise ValueError("Chunk size must be positive")
    original = Qwen3VLTextMLP.forward

    def forward(self, inputs):
        return checkpointed_chunks(lambda part: original(self, part), inputs, chunk_tokens)

    Qwen3VLTextMLP.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    args, remainder = parser.parse_known_args()
    install(args.chunk_tokens)
    sys.path.insert(0, str(args.entry.resolve().parent))
    sys.argv = [str(args.entry), *remainder]
    runpy.run_path(str(args.entry), run_name="__main__")


if __name__ == "__main__":
    main()
