# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Create an isolated oracle helper with correct PP layout and topology reporting."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def transform(source):
    old_import = (
        "    from tp_pp_real_rf_crop import move_batch\n\n"
        "    from cosmos_framework.model.generator.megatron_tp_pp.factory import nano_layer_factory"
    )
    new_import = old_import.replace(
        "move_batch\n\n", "move_batch\n    from tools.tp_pp_original_capacity_layout import initial_shapes\n\n"
    )
    old_layout = (
        "        layout=PayloadLayout(((network.fixed_und_tokens, 4096), (12, 4096), (6, 1280)), context_width=5376),"
    )
    new_layout = "        layout=PayloadLayout(initial_shapes(network, original_capacity=args.original_capacity), context_width=5376),"
    old_pp_report = '"pipeline_parallel_size": 2,'
    new_pp_report = '"pipeline_parallel_size": topology.pipeline_model_parallel_size,'
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Unexpected oracle source; refusing an ambiguous helper transformation") from error
    pp_values = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "pipeline_parallel_size"
    ]
    pp_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "pipeline_parallel_size"
    ]
    valid_pp_values = (
        ast.dump(ast.parse("2", mode="eval").body),
        ast.dump(ast.parse("topology.pipeline_model_parallel_size", mode="eval").body),
    )
    if (
        any(source.count(item) != 1 for item in (old_import, old_layout))
        or source.count(old_pp_report) + source.count(new_pp_report) != 1
        or len(pp_values) != 1
        or ast.dump(pp_values[0]) not in valid_pp_values
        or pp_writes
    ):
        raise ValueError("Unexpected oracle source; refusing an ambiguous helper transformation")
    result = (
        source.replace(old_import, new_import).replace(old_layout, new_layout).replace(old_pp_report, new_pp_report)
    )
    compile(result, "<isolated-oracle-helper>", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    result = transform(source)
    with args.output.open("x") as handle:
        handle.write(result)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(result.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
