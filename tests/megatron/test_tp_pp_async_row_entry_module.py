# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regress the script/runpy/Dynamo interaction without distributed or CUDA."""

import runpy
import sys
from pathlib import Path

import torch

from tools import run_tp_pp_async_row_reduction as candidate

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "tools/run_tp_pp_async_row_reduction.py"


def test_script_helpers_survive_nested_main_replacement_and_dynamo_guards(tmp_path, monkeypatch):
    # Direct script execution used to install functions whose module was
    # __main__, then replace that module via the benchmark run_path. A tiny
    # real CPU Dynamo graph exercises the affected guard in the new namespace.
    benchmark = tmp_path / "benchmark.py"
    training = tmp_path / "training.py"
    training.write_text(
        "import sys, torch\n"
        "from types import SimpleNamespace as NS\n"
        "from tools import run_tp_pp_async_row_reduction as candidate\n"
        "assert not hasattr(sys.modules['__main__'], 'finish')\n"
        "assert candidate.finish.__module__ == 'tools.run_tp_pp_async_row_reduction'\n"
        "assert candidate.async_update.__module__ == 'tools.run_tp_pp_async_row_reduction'\n"
        "class Row:\n"
        "    column = False\n"
        "    tp_size = 1\n"
        "    weight = None\n"
        "    bias = None\n"
        "    def _async_row_local(self, x, weight): return x\n"
        "def identity(x): return x\n"
        "def mlp():\n"
        "    return NS(gate_proj=identity, up_proj=identity, down_proj=Row(), act_fn=identity)\n"
        "layer = NS(mlp=mlp(), mlp_moe_gen=mlp(), post_attention_layernorm=identity, "
        "post_attention_layernorm_moe_gen=identity)\n"
        "block = NS(o_proj=Row(), down_proj=Row(), gate_proj=identity, up_proj=identity, "
        "post_attention_layernorm=identity)\n"
        "attn = NS(o_proj=Row(), o_proj_moe_gen=Row())\n"
        "def graph(x):\n"
        "    return candidate.async_update(layer, attn, block, x, x, x, x, x, x)\n"
        "compiled = torch.compile(graph, backend='eager', fullgraph=True)\n"
        "for x in (torch.tensor([2.]), torch.tensor([4.])):\n"
        "    for actual, expected in zip(compiled(x), graph(x), strict=True):\n"
        "        torch.testing.assert_close(actual, expected)\n"
    )
    benchmark.write_text(f"import runpy\nrunpy.run_path({str(training)!r}, run_name='__main__')\n")
    installs = []
    monkeypatch.setattr(candidate, "install", lambda: installs.append(candidate.finish))
    monkeypatch.setattr(sys, "argv", [str(ENTRY), "--entry", str(benchmark)])
    torch._dynamo.reset()
    try:
        runpy.run_path(str(ENTRY), run_name="__main__")
    finally:
        torch._dynamo.reset()
    assert installs == [candidate.finish]
