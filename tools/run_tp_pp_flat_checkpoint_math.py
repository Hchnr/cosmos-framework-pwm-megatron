# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Flatten nested checkpoint inputs for the opt-in full-capacity FP32 oracle.

Native and TP use the same non-reentrant block checkpoint. Nested tensor
inputs become top-level checkpoint inputs, allowing the caller's saved-tensor
offload hooks to move them to CPU. This is a diagnostic, not a training default.
"""

import argparse
import runpy
import sys
from pathlib import Path

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils import _pytree
from torch.utils.checkpoint import checkpoint


def flat_checkpoint(function, *args, preserve_rng_state=True, **kwargs):
    flat, spec = _pytree.tree_flatten((args, kwargs))

    def call(*values):
        positional, keyword = _pytree.tree_unflatten(values, spec)
        return function(*positional, **keyword)

    return checkpoint(
        call, *flat, use_reentrant=False, preserve_rng_state=preserve_rng_state
    )


def install():
    import tools.run_tp_pp_fp32_native_ac as native_ac
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

    def wrap(layer):
        return checkpoint_wrapper(
            layer, checkpoint_fn=flat_checkpoint, preserve_rng_state=True
        )

    def install_native_checkpointing():
        import tp_pp_gpu_leaf_reference as reference

        original_place = reference.place_gpu_trainables

        def place(model):
            handles = original_place(model)
            for index, layer in enumerate(model.language_model.model.layers):
                model.language_model.model.layers[index] = wrap(layer)
            return handles

        reference.place_gpu_trainables = place

    native_ac.install_native_checkpointing = install_native_checkpointing
    original_stage = PipelineStage.__init__

    def initialize(stage, *args, **kwargs):
        original_stage(stage, *args, **kwargs)
        for name, layer in list(stage.layers.items()):
            stage.layers[name] = wrap(layer)

    PipelineStage.__init__ = initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--checkpoint-entry", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    install()
    sys.path.insert(0, str(args.checkpoint_entry.resolve().parent))
    sys.argv = [str(args.checkpoint_entry), *remainder]
    runpy.run_path(str(args.checkpoint_entry), run_name="__main__")


if __name__ == "__main__":
    main()
