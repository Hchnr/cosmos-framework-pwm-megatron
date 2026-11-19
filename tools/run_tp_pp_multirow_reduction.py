# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""In-memory production TP layer candidate; never patches source files.

Assumes identical three-stream physical shapes and branch topology on all TP
ranks. This deliberately rejects inference caches and absent action towers.
Only opt-in diagnostics may install it.
"""

import argparse
import inspect
import runpy
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.tp_pp_multirow_reduction import GroupedRowSum


def _row_group(projections, inputs):
    if len(projections) != len(inputs) or not projections:
        raise ValueError("Each local row projection requires an input")
    first = projections[0]
    if any(p.column or p.tp_size != first.tp_size or p._impl.tp_group is not first._impl.tp_group for p in projections):
        raise ValueError("Grouped rows must share a TP group and degree")
    partials = tuple(p._multirow_local(value, p.weight) for p, value in zip(projections, inputs, strict=True))
    outputs = GroupedRowSum.apply(first._impl.tp_group.group_name, *partials) if first.tp_size > 1 else partials
    return tuple(value if p.bias is None else value + p.bias for p, value in zip(projections, outputs, strict=True))


def _gate_up(module, value, *, action):
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


def _fused_update(layer, attn, block, u, v, a, yu, yv, ya):
    if block is None or ya is None:
        raise ValueError("An action tower is required for the opt-in diagnostic")
    # Attention outputs already exist independently before any of these row
    # projections; none consumes the residual of another modality.
    da, du, dv = _row_group((block.o_proj, attn.o_proj, attn.o_proj_moe_gen), (ya, yu, yv))
    a, u, v = a + da, u + du, v + dv
    ga = _gate_up(block, block.post_attention_layernorm(a), action=True)
    gu = _gate_up(layer.mlp, layer.post_attention_layernorm(u), action=False)
    gv = _gate_up(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(v), action=False)
    ma, mu, mv = _row_group(
        (block.down_proj, layer.mlp.down_proj, layer.mlp_moe_gen.down_proj),
        (ga, gu, gv),
    )
    return u + mu, v + mv, a + ma


def install():
    from cosmos_framework.model.generator.megatron_tp_pp.compile import (
        NonePreservingLinear,
    )
    from cosmos_framework.model.generator.megatron_tp_pp.layers import (
        TensorParallelLinear,
    )
    from cosmos_framework.model.generator.mot import small_action_tower as module

    if getattr(module, "_multirow_installed", False):
        raise RuntimeError("Multirow candidate is already installed")
    original_forward = module.small_layer_forward
    original_init = TensorParallelLinear.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not self.column:
            group = self._impl.tp_group

            def local(value, weight):
                return NonePreservingLinear.apply(value, weight, None, False, False, False, None, None, group, 1, None)

            self.__dict__["_multirow_local"] = torch.compiler.allow_in_graph(local)

    source = textwrap.dedent(inspect.getsource(original_forward))
    first = "        a = a + block.o_proj(ya)\n"
    last = "    out = dict(pack)\n"
    if source.count(first) != 1 or source.count(last) != 1:
        raise RuntimeError("Unexpected three-stream source; refuse ambiguous transformation")
    start, end = source.index(first), source.index(last)
    source = (
        source[:start]
        + "    else:\n        ya = None\n"
        + "    u, v, a = _multirow_fused_update(layer, attn, block, u, v, a, yu, yv, ya)\n"
        + source[end:]
    )
    # Dynamo creates module-based guards for globals of the transformed
    # function, so the injected helper must also exist on the source module.
    module._multirow_fused_update = _fused_update
    namespace = dict(module.__dict__)
    # Generated solely from the already-loaded local function, with exact
    # sentinel checks above; untrusted strings are never used.
    exec(compile(source, original_forward.__code__.co_filename, "exec"), namespace)  # noqa: S102
    transformed = namespace["small_layer_forward"]

    def guarded(layer, pack, *, memory_value=None, gen_only=False):
        if memory_value is not None or gen_only or getattr(layer, "action_block_moe_gen_action", None) is None:
            raise ValueError("Multirow candidate only supports ordinary training with an action tower")
        return transformed(layer, pack, memory_value=memory_value, gen_only=gen_only)

    TensorParallelLinear.__init__ = initialize
    module.small_layer_forward = guarded
    module._multirow_installed = True


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
    main()
