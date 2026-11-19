"""Adapt native MCore Linear to the existing three-tower block math.

The factory accepts a meta decoder layer: only local TP storage is allocated.
Canonical parameter names, bias, freezing and norm implementations are retained.
"""

from dataclasses import replace

import torch
from torch import nn


class TensorParallelLinear(nn.Module):
    """Tensor-only forward with unchanged ``weight``/``bias`` checkpoint keys."""

    def __init__(self, source, config, *, column, tp_rank, tp_size):
        super().__init__()
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

        self.in_features, self.out_features = source.in_features, source.out_features
        self.axis = 0 if column else 1
        if source.weight.shape[self.axis] % tp_size:
            raise ValueError("Projection dimension is not divisible by TP")
        kind = ColumnParallelLinear if column else RowParallelLinear
        kwargs = {"gather_output": False} if column else {"input_is_parallel": True}
        impl = kind(
            self.in_features,
            self.out_features,
            config=config,
            init_method=lambda t: t,
            bias=source.bias is not None,
            skip_bias_add=False,
            **kwargs,
        )
        # The MCore module is kept outside the registered tree; register exactly
        # its parameters here, preserving source FQNs without an extra `.impl`.
        self.__dict__["_impl"] = impl
        from .compile import CopyToTensorParallel, NonePreservingLinear, ReduceFromTensorParallel

        tp_group = impl.tp_group

        def traceable_linear(x, weight, bias):
            # Keep ProcessGroup in the closure: it cannot be an FX graph input.
            # AOTAutograd traces the original MCore local GEMM autograd function.
            # Communication uses equivalent functional collectives, since eager
            # c10d Work handles have no FakeTensor implementation.
            if column and tp_size > 1:
                x = CopyToTensorParallel.apply(x, tp_group.group_name)
            output = NonePreservingLinear.apply(
                x,
                weight,
                bias if column else None,
                False,
                False,
                False,
                None,
                None,
                tp_group,
                1,
                None,
            )
            if not column:
                if tp_size > 1:
                    output = ReduceFromTensorParallel.apply(output, tp_group.group_name)
                if bias is not None:
                    output = output + bias
            return output

        self.__dict__["_traceable_linear"] = torch.compiler.allow_in_graph(traceable_linear)
        self.__dict__["_eager_linear"] = NonePreservingLinear
        self.__dict__["_reduce"] = ReduceFromTensorParallel
        self.column, self.tp_size = column, tp_size
        self.weight = impl.weight
        self.bias = impl.bias
        for key in ("weight", "bias"):
            old, new = getattr(source, key), getattr(self, key)
            if old is None:
                continue
            new.requires_grad_(old.requires_grad)
            if not old.is_meta:
                value = old.detach()
                if key == "weight" or column:
                    axis = self.axis if key == "weight" else 0
                    n = value.shape[axis] // tp_size
                    value = value.narrow(axis, tp_rank * n, n)
                with torch.no_grad():
                    new.copy_(value)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Device/dtype transforms may replace Parameter objects. Rebind outside
        # the compiled/AC region: mutating module state inside an AC higher-order
        # operation prevents Dynamo from capturing full blocks.
        self._impl.weight, self._impl.bias = self.weight, self.bias
        return self

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self._traceable_linear(x, self.weight, self.bias)
        output = self._eager_linear.apply(
            x,
            self.weight,
            self.bias if self.column else None,
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
            if self.bias is not None:
                output = output + self.bias
        return output


def parallelize_layer(layer, config, tp_rank, tp_size):
    """Replace only projections, then update head counts to rank-local values."""
    if getattr(layer, "_cosmos_tensor_parallel", False):
        raise ValueError("Layer is already tensor parallel")
    attn = layer.self_attn
    block = layer.action_block_moe_gen_action
    if block.adapters:
        raise ValueError("TP supports matching Action/VIDEO head layouts (width_only preset)")
    for count in (attn.num_attention_heads, attn.num_key_value_heads, block.shape.q_heads, block.shape.kv_heads):
        if count % tp_size:
            raise ValueError("Q and KV heads must be divisible by TP")
    if (attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim) != (
        block.shape.q_heads,
        block.shape.kv_heads,
        block.shape.head_dim,
    ):
        raise ValueError("Action and main tower attention layouts must match")
    columns = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
    rows = {"o_proj", "down_proj"}
    # Snapshot traversal before mutation. No fused concatenation of TP shards.
    for name, module in list(layer.named_modules()):
        leaf = name.rsplit(".", 1)[-1].removesuffix("_moe_gen")
        if isinstance(module, nn.Linear) and leaf in columns | rows:
            parent_path, _, key = name.rpartition(".")
            parent = layer.get_submodule(parent_path) if parent_path else layer
            setattr(
                parent,
                key,
                TensorParallelLinear(
                    module,
                    config,
                    column=leaf in columns,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                ),
            )
    attn.num_attention_heads //= tp_size
    attn.num_key_value_heads //= tp_size
    block.shape = replace(block.shape, q_heads=block.shape.q_heads // tp_size, kv_heads=block.shape.kv_heads // tp_size)
    # Q/K RMSNorm weights are shared across heads, so their local head partials
    # need SUM. Residual norms already see full dgrad from ColumnParallelLinear.
    for name, param in layer.named_parameters():
        param._cosmos_tp_partial_grad = ".q_norm" in name or ".k_norm" in name
    layer._cosmos_tensor_parallel = True
    layer.small_action_enabled = True
    return layer
