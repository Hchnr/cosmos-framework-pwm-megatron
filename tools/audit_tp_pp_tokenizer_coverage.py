# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Audit structural AOT coverage of the recipe's actual Action128 catalogs.

This reads catalog metadata, never media or sampling state. It does not prove
that a complete model can consume the packages or that an update fits memory.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import tomllib

from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution


def canvas_sizes(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "view_layout")
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "sizes" for target in node.targets)
    )
    sizes = ast.literal_eval(assignment.value)
    if len(sizes) != 3 or any(len(family) != 3 for family in sizes):
        raise ValueError("Action128 view layout changed; review the structural audit")
    return {(height, width) for family in sizes for width, height in family}


def variant_keys(canvases, tokenizer):
    keys = set()
    chunks = tokenizer["encode_chunk_frames"]
    for height, width in canvases:
        chunk = chunks[get_vision_data_resolution((height, width))] if isinstance(chunks, dict) else chunks
        if height % 2 or width % 2 or chunk < 1:
            raise ValueError("Invalid tokenizer chunk or spatial patch dimensions")
        for cache in (0, 1, 2):
            keys.add((1 if cache == 0 else chunk, height // 2, width // 2, cache))
        for duration in tokenizer.get("encode_exact_durations") or []:
            remaining = duration - 1
            if remaining > 0 and remaining % chunk:
                keys.add((remaining % chunk, height // 2, width // 2, 1 if remaining // chunk == 0 else 2))
    return keys


def audit(recipe, artifact):
    config = tomllib.loads(recipe.read_text())
    identity = json.loads((artifact / "build_identity.json").read_text())
    manifest = json.loads((artifact / "manifest.json").read_text())
    layout_path = Path("cosmos_framework/data/generator/action/unified_action128.py")
    canvases = canvas_sizes(layout_path.read_text())
    configured = {tuple(size) for size in config["trainer"]["callbacks"]["compile_tokenizer"]["spatial_sizes"]}
    rows, unknown = [], []
    for entry in config["dataloader_train"]["datasets"]:
        path = Path(entry["path"])
        table = pq.read_table(path, columns=["type", "data"])
        supported = True
        for row in table.to_pylist():
            data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            if row["type"] != "action" or data.get("adapter") != "unified_action128":
                supported = False
        rows.append({"path": str(path), "rows": table.num_rows, "all_unified_action128": supported})
        if not supported or table.num_rows == 0:
            unknown.append(str(path))
    expected = variant_keys(canvases, identity["tokenizer"])
    actual = {tuple(map(int, key.split(","))) for key in manifest["variants"]}
    source = Path("cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py")
    checks = {
        "all_catalogs_use_audited_layout": bool(rows) and not unknown,
        "recipe_covers_all_structural_canvases": canvases <= configured,
        "cache_covers_all_structural_variants": expected <= actual,
        "cache_compiler_canvases_match_recipe": configured
        == {tuple(size) for size in identity["compiler"]["spatial_sizes"]},
        "cache_identity_matches": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        == manifest["fingerprint"],
        "cache_tokenizer_source_matches": hashlib.sha256(source.read_bytes()).hexdigest() == identity["source_sha256"],
    }
    return {
        "passed": all(checks.values()),
        "scope": "structural canvas/temporal coverage under the cache's declared tokenizer configuration",
        "checks": checks,
        "layout_source_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
        "catalogs": rows,
        "unknown_catalogs": unknown,
        "structural_canvases": sorted(canvases),
        "missing_canvases": sorted(canvases - configured),
        "expected_variant_count": len(expected),
        "missing_variants": sorted(expected - actual),
        "media_decoded": False,
        "sampling_state_consumed": False,
        "actual_training_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.toml, args.artifact)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "catalogs"}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
