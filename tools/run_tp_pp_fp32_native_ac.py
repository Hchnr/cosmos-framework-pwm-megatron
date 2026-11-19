"""Run the native FP32 oracle with full per-block activation checkpointing.

Install after frozen-region streaming hooks so recomputation streams the same
native weights. The original helper, checkpoint, inputs, arithmetic and fixed
comparison thresholds remain intact. This entry is an oracle memory fix only.
"""

import argparse
import runpy
import sys
from pathlib import Path


def install_native_checkpointing():
    import tp_pp_gpu_leaf_reference as reference
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    original = reference.place_gpu_trainables

    def place(model):
        handles = original(model)
        for index, layer in enumerate(model.language_model.model.layers):
            model.language_model.model.layers[index] = checkpoint_wrapper(layer, preserve_rng_state=True)
        return handles

    reference.place_gpu_trainables = place


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    sys.path.insert(0, str(args.run_path.resolve().parent))
    install_native_checkpointing()
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")


if __name__ == "__main__":
    main()
