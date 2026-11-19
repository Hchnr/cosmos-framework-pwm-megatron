# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qualify flattened native and TP checkpoints with the production-I/O oracle."""

import argparse
import functools
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if "--compile" in remainder:
        raise ValueError(
            "This eager qualifier already installs full checkpoints; use no extra --compile wrapper"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    import tp_pp_network_smoke as smoke

    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from tools import run_tp_pp_flat_checkpoint_math as flat
    from tools.run_tp_pp_chunked_mlp_math import install as install_chunk

    counts = {"native": 0, "tp": 0}
    reference_models = []
    original_make_native = smoke.make_native

    def make_native():
        result = original_make_native()
        reference_models.append(result[0])
        return result

    smoke.make_native = make_native
    smoke.ModelParallelConfig = functools.partial(
        smoke.ModelParallelConfig,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
    )
    original_checkpoint = flat.flat_checkpoint

    def observed_tp(*positional, **keywords):
        counts["tp"] += 1
        return original_checkpoint(*positional, **keywords)

    flat.flat_checkpoint = observed_tp
    install_chunk(args.chunk_tokens)
    flat.install()
    original_stage = PipelineStage.__init__

    def initialize(stage, *positional, **keywords):
        original_stage(stage, *positional, **keywords)
        if len(reference_models) != 1:
            raise RuntimeError(
                "Expected one native reference before TP stage construction"
            )
        for layer in reference_models[0].language_model.model.layers:
            original_forward = layer.forward

            def observed_native(*values, _forward=original_forward, **options):
                counts["native"] += 1
                return original_checkpoint(_forward, *values, **options)

            layer.forward = observed_native

    PipelineStage.__init__ = initialize
    sys.argv = [sys.argv[0], "--output", str(args.output), *remainder]
    smoke.main()
    if counts["native"] < 4 or counts["tp"] < 4:
        raise RuntimeError(
            f"Flattened checkpoints were not exercised on both sides: {counts}"
        )
    report = {
        "passed": True,
        "native_checkpoint_calls": counts["native"],
        "tp_checkpoint_calls": counts["tp"],
        "chunk_tokens": args.chunk_tokens,
        "full_model_acceptance": False,
    }
    target = args.output.with_name(
        f"{args.output.stem}.flat_checkpoint.rank{os.environ['RANK']}.json"
    )
    with target.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
