# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in experiment: launch distinct three-stream row SUMs before consuming them.

This changes only an in-memory diagnostic model; ranks must use identical
three-stream topology. GPU/NCCL and Inductor overlap are not yet qualified.
"""

import argparse
import inspect
import runpy
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F


class LaunchRowSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group_name):
        ctx.set_materialize_grads(False)
        return torch.ops._c10d_functional.all_reduce(tensor.contiguous(), "sum", group_name)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class WaitRowSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.set_materialize_grads(False)
        return torch.ops._c10d_functional.wait_tensor(tensor)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def launch(projection, tensor):
    if projection.column:
        raise ValueError("Expected row-parallel projection")
    local = projection._async_row_local(tensor, projection.weight)
    pending = LaunchRowSum.apply(local, projection._impl.tp_group.group_name) if projection.tp_size > 1 else local
    return pending, projection.bias, projection.tp_size


def finish(pending):
    tensor, bias, tp_size = pending
    value = WaitRowSum.apply(tensor) if tp_size > 1 else tensor
    return value if bias is None else value + bias


def gate_up(module, value, *, action):
    if getattr(module, "_training_fuse_gate_up", False):
        raise ValueError("Training gate/up fusion is outside this candidate")
    shared = getattr(module, "_shared_input_gate_up", False) if action else type(module).__name__ == "SharedInputMLP"
    if shared:
        from tools.tp_pp_shared_input_reduction import shared_input

        value = shared_input((module.gate_proj, module.up_proj), value)
        gate = module.gate_proj._shared_local_linear(value, module.gate_proj.weight, module.gate_proj.bias)
        up = module.up_proj._shared_local_linear(value, module.up_proj.weight, module.up_proj.bias)
    else:
        gate, up = module.gate_proj(value), module.up_proj(value)
    return (F.silu(gate) if action else module.act_fn(gate)) * up


def async_update(layer, attn, block, u, v, a, yu, yv, ya):
    if block is None or ya is None:
        raise ValueError("An action tower is required")
    # Each launch can progress while subsequent independent local GEMMs run.
    # Keep launch order fixed on every rank, even if downstream losses omit a
    # modality; do not materialize unused gradients in backward.
    pending_a = launch(block.o_proj, ya)
    pending_u = launch(attn.o_proj, yu)
    pending_v = launch(attn.o_proj_moe_gen, yv)
    a = a + finish(pending_a)
    ga = gate_up(block, block.post_attention_layernorm(a), action=True)
    pending_ma = launch(block.down_proj, ga)
    u = u + finish(pending_u)
    gu = gate_up(layer.mlp, layer.post_attention_layernorm(u), action=False)
    pending_mu = launch(layer.mlp.down_proj, gu)
    v = v + finish(pending_v)
    gv = gate_up(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(v), action=False)
    pending_mv = launch(layer.mlp_moe_gen.down_proj, gv)
    return u + finish(pending_mu), v + finish(pending_mv), a + finish(pending_ma)


def install():
    from cosmos_framework.model.generator.megatron_tp_pp.compile import (
        NonePreservingLinear,
    )
    from cosmos_framework.model.generator.megatron_tp_pp.layers import (
        TensorParallelLinear,
    )
    from cosmos_framework.model.generator.mot import small_action_tower as module

    if getattr(module, "_async_row_installed", False):
        raise RuntimeError("Candidate already installed")
    original_forward = module.small_layer_forward
    original_init = TensorParallelLinear.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not self.column:
            group = self._impl.tp_group

            def local(value, weight):
                return NonePreservingLinear.apply(value, weight, None, False, False, False, None, None, group, 1, None)

            self.__dict__["_async_row_local"] = torch.compiler.allow_in_graph(local)

    source = textwrap.dedent(inspect.getsource(original_forward))
    first = "        a = a + block.o_proj(ya)\n"
    last = "    out = dict(pack)\n"
    if source.count(first) != 1 or source.count(last) != 1:
        raise RuntimeError("Unexpected three-stream layer; reject transformation")
    start, end = source.index(first), source.index(last)
    source = (
        source[:start]
        + "    else:\n        ya = None\n"
        + "    u, v, a = _async_row_update(layer, attn, block, u, v, a, yu, yv, ya)\n"
        + source[end:]
    )
    module._async_row_update = async_update
    namespace = dict(module.__dict__)
    exec(compile(source, original_forward.__code__.co_filename, "exec"), namespace)  # noqa: S102
    transformed = namespace["small_layer_forward"]

    def guarded(layer, pack, *, memory_value=None, gen_only=False):
        if memory_value is not None or gen_only or getattr(layer, "action_block_moe_gen_action", None) is None:
            raise ValueError("Candidate supports ordinary training with an action tower only")
        return transformed(layer, pack, memory_value=memory_value, gen_only=gen_only)

    TensorParallelLinear.__init__ = initialize
    module.small_layer_forward = guarded
    module._async_row_installed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--shared-input-reduction", action="store_true")
    args, remainder = parser.parse_known_args()
    if args.shared_input_reduction:
        from tools.tp_pp_shared_input_reduction import install as install_shared

        install_shared()
    install()
    sys.path.insert(0, str(args.entry.resolve().parent))
    sys.argv = [str(args.entry), *remainder]
    runpy.run_path(str(args.entry), run_name="__main__")


if __name__ == "__main__":
    # run_path below temporarily replaces sys.modules['__main__']. Dynamo
    # guards must resolve candidate helpers from a stable importable module,
    # not from this temporary script namespace.
    from tools.run_tp_pp_async_row_reduction import main as imported_main

    imported_main()
