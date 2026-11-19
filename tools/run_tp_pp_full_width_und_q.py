# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Add frozen UND Q reconstruction to the existing full-width K/V diagnostic.

Only useful for a targeted FP32 numerical experiment after the captured UND Q
replay identifies width-dependent arithmetic. This is not a production path.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path


def mark_und_q(stage, observation, linear_type):
    count = 0
    for layer_name, layer in stage.layers.items():
        projection = layer.self_attn.q_proj
        if (
            not isinstance(projection, linear_type)
            or not projection.column
            or projection.tp_size <= 1
            or projection.weight.requires_grad
            or projection.bias is not None
        ):
            raise ValueError("UND Q must be a frozen, bias-free column-parallel projection")
        if getattr(projection, "_fp32_reconstructed_kv", False):
            raise ValueError("UND Q was already marked for reconstruction")
        name = f"layers.{layer_name}.self_attn.q_proj"
        if name in observation.data["modules"]:
            raise ValueError(f"Duplicate reconstructed parameter: {name}")
        projection._fp32_reconstructed_kv = True
        projection._fp32_reconstructed_kv_name = name
        observation.data["modules"][name] = {
            "local_weight_shape": list(projection.weight.shape),
            "trainable": False,
            "forward_calls": 0,
            "kind": "frozen_und_q",
        }
        count += 1
    if not count:
        raise ValueError("No UND Q projections found")
    observation.save()
    return count


def install(kv, stage_type, linear_type):
    original = kv.install

    def augment(observation):
        original(observation)
        initialize = stage_type.__init__

        def initialize_with_und_q(stage, *args, **kwargs):
            initialize(stage, *args, **kwargs)
            mark_und_q(stage, observation, linear_type)

        stage_type.__init__ = initialize_with_und_q
        observation.data["und_q_reconstruction"] = True
        observation.data["full_model_acceptance"] = False

    kv.install = augment


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--vjp-entry", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    checkout = Path(os.environ["FULL_VJP_CHECKOUT"]).resolve(strict=True)
    if not args.vjp_entry.is_file():
        raise FileNotFoundError(args.vjp_entry)
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(args.vjp_entry.resolve().parent))
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from tools import tp_pp_fp32_reconstructed_kv as kv

    install(kv, PipelineStage, TensorParallelLinear)
    sys.argv = [str(args.vjp_entry), *remainder]
    runpy.run_path(str(args.vjp_entry), run_name="__main__")


if __name__ == "__main__":
    main()
