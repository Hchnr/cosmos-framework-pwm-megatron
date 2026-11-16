# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU regression: production SAC wrapper must save FA3, but recompute GEMMs.

Run directly to avoid pytest's global CUDA environment overrides. Measures
dispatcher events, not CUDA kernel name guesses, after compilation warmup.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from flash_attn_3_nv.flash_attn_interface import flash_attn_varlen_func

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_ac_to_transformer_block

PATTERN = r"^_flash_attn_forward\.default$"


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(4096, 6144, bias=False, device="cuda", dtype=torch.bfloat16)
        self.out = torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
        self.mlp = torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16)

    def forward(self, x, cu):
        q, k, v = self.qkv(x).split((4096, 1024, 1024), dim=-1)
        y, _ = flash_attn_varlen_func(
            q.reshape(-1, 32, 128),
            k.reshape(-1, 8, 128),
            v.reshape(-1, 8, 128),
            cu,
            cu,
            256,
            256,
            causal=True,
            deterministic=True,
            return_attn_probs=True,
        )
        y = self.out(y.flatten(1)) + x
        return self.mlp(torch.nn.functional.silu(y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    original = Block()
    x = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)
    results = []
    for compiled in (False, True):
        reference = None
        counts = {}
        for name, mode, patterns in (
            ("none", "none", []),
            ("full", "full", []),
            ("legacy_regex", "selective", ["fmha"]),
            ("attention_only", "selective", [PATTERN]),
        ):
            torch.compiler.reset()
            block = copy.deepcopy(original)
            fn = (
                block
                if mode == "none"
                else _apply_ac_to_transformer_block(
                    block, ActivationCheckpointingConfig(mode=mode, save_ops_regex=patterns)
                )
            )
            if compiled:
                fn = torch.compile(fn, fullgraph=True, dynamic=False)

            def step():
                block.zero_grad(set_to_none=True)
                inp = x.clone().requires_grad_()
                out = fn(inp, cu)
                out.float().square().mean().backward()
                return out.detach(), inp.grad.detach(), [p.grad.detach().clone() for p in block.parameters()]

            for _ in range(2):
                step()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                got = step()
                torch.cuda.synchronize()
            events = {e.key: e.count for e in prof.key_averages()}
            # TorchDispatchMode can emit nested entries for a single dispatch.
            # Count only leaf GEMMs and the extension's actual fwd invocation.
            events["aten::mm"] = sum(
                e.name == "aten::mm" and not any(c.name == "aten::mm" for c in e.cpu_children) for e in prof.events()
            )
            counts[name] = {k: v for k, v in events.items() if "flash_attn" in k or k in ("aten::mm", "aten::addmm")}
            flat = [got[0], got[1], *got[2]]
            if reference is None:
                reference = flat
            errors = []
            for a, b in zip(reference, flat, strict=True):
                torch.testing.assert_close(a, b, rtol=0.01, atol=0.001)
                errors.append(float((a.float() - b.float()).abs().max()))
            results.append({"compiled": compiled, "mode": name, "counts": counts[name], "max_abs_errors": errors})
        op = "flash_attn_3::fwd"
        assert counts["full"][op] == counts["legacy_regex"][op] == 2, counts
        assert counts["attention_only"][op] == counts["none"][op] == 1, counts
        # Same selective machinery with/without FA caching: dispatch accounting
        # is comparable here; full AC lacks TorchDispatchMode nesting.
        assert counts["attention_only"]["aten::mm"] == counts["legacy_regex"]["aten::mm"], counts
        assert counts["attention_only"]["aten::mm"] > counts["none"]["aten::mm"], counts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
