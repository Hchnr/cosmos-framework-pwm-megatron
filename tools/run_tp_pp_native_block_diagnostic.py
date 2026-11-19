"""Isolate TP GEMM/head reductions using replicated native block arithmetic.

Retains sharded parameters, PP, MCore DDP, loss and optimizer, but gathers each
block's weights and computes the native full block on each TP rank. This is a
slow diagnostic control, never a performance or production TP acceptance.
"""

import argparse
import runpy
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist


class GatherNativeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, axis, group):
        ctx.axis = axis
        ctx.rank = dist.get_rank(group)
        ctx.width = parameter.shape[axis]
        pieces = [torch.empty_like(parameter) for _ in range(dist.get_world_size(group))]
        dist.all_gather(pieces, parameter.contiguous(), group=group)
        return torch.cat(pieces, dim=axis)

    @staticmethod
    def backward(ctx, gradient):
        # Every TP rank computed the same full block and cotangent, so each
        # owner selects its shard. Summing would multiply the gradient by TP.
        return gradient.narrow(ctx.axis, ctx.rank * ctx.width, ctx.width).contiguous(), None, None


def clear_native_block_partial_flags(stage):
    # Materializing meta parameters can replace Parameter objects and discard
    # diagnostic attributes. Select the stage's actual layer parameters after
    # materialization; native full-head RMSNorm already computed the full grad.
    changed = 0
    for layer in stage.layers.values():
        for parameter in layer.parameters():
            changed += bool(getattr(parameter, "_cosmos_tp_partial_grad", False))
            parameter._cosmos_tp_partial_grad = False
    return changed


def install():
    from megatron.core import parallel_state as ps

    from cosmos_framework.model.generator.megatron_tp_pp.factory import nano_layer_factory
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage
    from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from cosmos_framework.training_backends.megatron_tp_pp.driver import ModelParallelDriver

    factory = nano_layer_factory(
        Qwen3VLTextConfig(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-6,
        )
    )
    original_stage = PipelineStage.__init__

    def initialize_stage(self, *args, **kwargs):
        original_stage(self, *args, **kwargs)
        group = ps.get_tensor_model_parallel_group()
        for index, layer in self.layers.items():
            native = factory(int(index))
            if set(dict(native.named_parameters())) != set(dict(layer.named_parameters())):
                raise ValueError("Native diagnostic parameter coverage differs")

            def forward(block, *inputs, _native=native, **keywords):
                parameters = {
                    name: GatherNativeWeight.apply(parameter, parameter.partition_dim, group)
                    if getattr(parameter, "tensor_model_parallel", False)
                    else parameter
                    for name, parameter in block.named_parameters()
                }
                return torch.func.functional_call(
                    _native, (parameters, dict(block.named_buffers())), inputs, keywords, strict=True
                )

            layer.forward = types.MethodType(forward, layer)
            for parameter in layer.parameters():
                parameter._cosmos_native_block_diagnostic = True

    original_driver = ModelParallelDriver.__init__

    def initialize_driver(self, *args, **kwargs):
        original_driver(self, *args, **kwargs)
        changed = clear_native_block_partial_flags(self.stage)
        if not changed:
            raise ValueError("Native block diagnostic did not find any head-local norm flags")
        print("DIAGNOSTIC: replicated native blocks; production TP acceptance disabled", flush=True)

    PipelineStage.__init__ = initialize_stage
    ModelParallelDriver.__init__ = initialize_driver


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-path", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if "--mode" not in remainder or remainder[remainder.index("--mode") + 1] != "tp":
        parser.error("Native-block isolation applies only to the TP diagnostic")
    sys.path.insert(0, str(args.run_path.resolve().parent))
    install()
    sys.argv = [str(args.run_path), *remainder]
    runpy.run_path(str(args.run_path), run_name="__main__")


if __name__ == "__main__":
    main()
