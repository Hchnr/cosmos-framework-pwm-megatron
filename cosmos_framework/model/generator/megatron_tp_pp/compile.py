"""Functional TP communication at the compiled MCore GEMM boundary.

MCore's eager async Work handles cannot be represented in FX. These two
autograd identities preserve the exact Column/Row communication contract using
PyTorch's traceable functional collectives. GEMMs and local wgrad/dgrad remain
the original MCore implementations. No sequence-parallel semantics are added.
"""

import torch
from megatron.core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication


class NonePreservingLinear(LinearWithGradAccumulationAndAsyncCommunication):
    """Keep MCore's GEMM/backward, while skipping disconnected autograd edges."""

    @staticmethod
    def forward(ctx, *args):
        output = LinearWithGradAccumulationAndAsyncCommunication.forward(ctx, *args)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * len(ctx.needs_input_grad)
        return LinearWithGradAccumulationAndAsyncCommunication.backward(ctx, gradient)


def _sum(tensor, group_name):
    output = torch.ops._c10d_functional.all_reduce(tensor.contiguous(), "sum", group_name)
    return torch.ops._c10d_functional.wait_tensor(output)


class CopyToTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group_name):
        ctx.group_name = group_name
        ctx.set_materialize_grads(False)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient):
        return (_sum(gradient, ctx.group_name) if gradient is not None else None), None


class ReduceFromTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group_name):
        ctx.set_materialize_grads(False)
        return _sum(tensor, group_name)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None
