"""Training projection fusion without changing checkpoint parameter names.

Weights remain separate Parameters for optimizer/DCP compatibility. Concatenation
inside the compiled region lets a single GEMM consume each shared input tensor.
Only plain dense Linear modules are eligible; LoRA/custom linears retain their
original forward implementations.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FusedInputMLP(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn

    def forward(self, x):
        gate, up = F.linear(x, torch.cat((self.gate_proj.weight, self.up_proj.weight))).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


def _plain_biasless(module, names):
    return all(type(getattr(module, name, None)) is nn.Linear and getattr(module, name).bias is None for name in names)


def fuse_decoder_projections(layers):
    for layer in layers:
        attn = layer.self_attn
        suffixes = [suffix for suffix in ("", "_moe_gen") if hasattr(attn, "q_proj" + suffix)]
        if all(
            _plain_biasless(attn, [name + suffix for name in ("q_proj", "k_proj", "v_proj")]) for suffix in suffixes
        ):
            attn._training_fuse_qkv = True
        for name in ("mlp", "mlp_moe_gen"):
            mlp = getattr(layer, name, None)
            if (
                mlp is not None
                and hasattr(mlp, "act_fn")
                and _plain_biasless(mlp, ("gate_proj", "up_proj", "down_proj"))
            ):
                setattr(layer, name, FusedInputMLP(mlp))
        action = getattr(layer, "action_block_moe_gen_action", None)
        if action is not None:
            if _plain_biasless(action, ("q_proj", "k_proj", "v_proj")):
                action._training_fuse_qkv = True
            if _plain_biasless(action, ("gate_proj", "up_proj", "down_proj")):
                action._training_fuse_gate_up = True
