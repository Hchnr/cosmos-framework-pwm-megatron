# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in TP selective activation checkpointing using the native SAC policy.

This experimental entry does not change the default backend or baseline. It
retains parameter identities, compilation settings and RNG preservation, and
must pass numerical and memory qualification before performance acceptance.
"""

import argparse
import copy
import runpy
import sys
from pathlib import Path


def selective_config(original, patterns):
    if original.mode != "full":
        raise ValueError("The selective AC candidate requires a full-AC baseline")
    config = copy.copy(original)
    config.mode = "selective"
    config.save_ops_regex = list(patterns)
    return config


def selected_layers(names, count):
    names = tuple(names)
    if not 0 < count <= len(names):
        raise ValueError("Selective AC layer count must fit every pipeline stage")
    return frozenset(names[-count:])


def install(patterns, count):
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_selective_ac
    from cosmos_framework.training_backends.megatron_tp_pp import model, train

    def optimize(stage, model_config):
        ac, compile_config = model_config.activation_checkpointing, model_config.compile
        selected = selective_config(ac, patterns)
        if compile_config.enabled and (compile_config.compiled_region != "language" or compile_config.use_cuda_graphs):
            raise ValueError("The candidate requires the existing language-block compilation policy")
        # Consult the same provider as the default backend at call time. Other
        # opt-in experiments may wrap it to install and observe scheduling.
        options = model.inductor_compile_options(compile_config)
        selected_names = selected_layers(stage.layers, count)
        for name, block in list(stage.layers.items()):
            block = (
                _apply_selective_ac(block, selected)
                if name in selected_names
                else checkpoint_wrapper(
                    block, preserve_rng_state=ac.preserve_rng_state, determinism_check=ac.determinism_check
                )
            )
            if compile_config.enabled:
                block = torch.compile(
                    block, fullgraph=True, dynamic=compile_config.compile_dynamic, options=options or None
                )
            stage.layers[name] = block

    train.optimize_stage = optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    # --entry can be nested under another wrapper that also consumes --run-path.
    parser.add_argument("--run-path", "--entry", dest="run_path", type=Path, required=True)
    parser.add_argument("--save-op-regex", action="append", required=True)
    parser.add_argument("--selective-layers", type=int, default=4)
    parser.add_argument("--shared-input-reduction", action="store_true")
    args, remainder = parser.parse_known_args()
    if args.shared_input_reduction:
        from tools.tp_pp_shared_input_reduction import install as install_shared

        install_shared()
    install(args.save_op_regex, args.selective_layers)
    sys.path.insert(0, str(args.run_path.resolve().parent))
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")


if __name__ == "__main__":
    main()
