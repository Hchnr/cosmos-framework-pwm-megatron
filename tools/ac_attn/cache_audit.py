# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Inspect generated Inductor attention graphs without importing/executing them.

Supplementary evidence only: CUDA replay counts must still come from nsys.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def inspect(path):
    source = path.read_text()
    if "torch.ops.flash_attn_3._flash_attn_" not in source:
        return None
    tree = ast.parse(source)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forward = calls.count("torch.ops.flash_attn_3._flash_attn_forward.default")
    backward = calls.count("torch.ops.flash_attn_3._flash_attn_backward.default")
    # Locate actual emitted graph, not benchmark input generation.
    graph = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call")
    returns = [ast.unparse(n.value) for n in ast.walk(graph) if isinstance(n, ast.Return)]
    saved_buffers = []
    if not backward:
        for node in ast.walk(graph):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                # Current two-tower graph returns UND and GEN hidden states first.
                saved_buffers.extend(ast.unparse(n) for n in node.value.elts[2:] if ast.unparse(n).startswith("buf"))
    attention_results = {}
    attention_calls = set()
    for node in ast.walk(graph):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = ast.unparse(node.targets[0])
        if isinstance(node.value, ast.Call) and "_flash_attn_forward.default" in ast.unparse(node.value.func):
            attention_calls.add(target)
        if isinstance(node.value, ast.Subscript) and ast.unparse(node.value.value) in attention_calls:
            attention_results[target] = ast.unparse(node.value)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "kind": "backward" if backward else "forward",
        "fa_forward_calls": forward,
        "fa_backward_calls": backward,
        "gemm_calls": sum("extern_kernels.mm" == c for c in calls),
        "returns": returns,
        "attention_result_variables": attention_results,
        "saved_buffers_after_two_tower_outputs": saved_buffers,
        "non_attention_saved_buffers": [b for b in saved_buffers if b not in attention_results],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.run / "compile_cache" / "rank0").rglob("*.py")):
        result = inspect(path)
        if result:
            rows.append(result)
    assert rows, "No attention graphs found"
    (args.run / "cache_audit.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(
            row["kind"],
            "FA fwd",
            row["fa_forward_calls"],
            "FA bwd",
            row["fa_backward_calls"],
            "GEMMs",
            row["gemm_calls"],
            row["path"],
        )


if __name__ == "__main__":
    main()
