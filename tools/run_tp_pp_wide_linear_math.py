# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Isolate FP32 TP summation error using FP64 Linear accumulators on both sides.

Parameters, activations, gradients, Adam state and nonlinearities remain FP32.
Only native/MCore Linear GEMMs and their TP reductions use FP64 intermediates.
This is an explicit numerical diagnostic, not the BF16 performance runtime.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional


def install_wide_linears():
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear

    def native(self, inputs):
        if inputs.dtype != torch.float32 or self.weight.dtype != torch.float32:
            raise ValueError("Wide-accumulator math requires FP32 model/input storage")
        return functional.linear(
            inputs.double(), self.weight.double(), self.bias.double() if self.bias is not None else None
        ).float()

    def parallel(self, inputs):
        if inputs.dtype != torch.float32 or self.weight.dtype != torch.float32:
            raise ValueError("Wide-accumulator TP requires FP32 model/input storage")
        if torch.compiler.is_compiling():
            raise ValueError("This numerical diagnostic uses eager MCore Linear with full AC")
        bias = self.bias.double() if self.bias is not None else None
        output = self._eager_linear.apply(
            inputs.double(),
            self.weight.double(),
            bias if self.column else None,
            False,
            self.column and self.tp_size > 1,
            False,
            None,
            None,
            self._impl.tp_group,
            1,
            None,
        )
        if not self.column:
            if self.tp_size > 1:
                output = self._reduce.apply(output, self._impl.tp_group.group_name)
            if bias is not None:
                output = output + bias
        return output.float()

    nn.Linear.forward = native
    TensorParallelLinear.forward = parallel


def install_tp_checkpointing():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

    original = PipelineStage.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for name, layer in list(self.layers.items()):
            self.layers[name] = checkpoint_wrapper(layer, preserve_rng_state=True)

    PipelineStage.__init__ = initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--native-norm-order", type=Path)
    args, remainder = parser.parse_known_args()
    if "--native-block-control" in remainder:
        parser.error("Keep the native-block control separate from accumulator isolation")
    install_wide_linears()
    install_tp_checkpointing()
    if args.native_norm_order is not None:
        from tools.tp_pp_native_grad_norm import install_native_gradient_norm

        names = json.loads(args.native_norm_order.read_text())["names"]
        install_native_gradient_norm([name if name.startswith("net.") else "net." + name for name in names])
    from tools.run_tp_pp_traced_fp32_math import main as traced_main

    sys.argv = [sys.argv[0], "--run-path", str(args.helper), "--trace-dir", str(args.trace_dir), *remainder]
    traced_main()


if __name__ == "__main__":
    main()
