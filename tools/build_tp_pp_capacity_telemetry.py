# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Add reporting-only comparison telemetry to an isolated diagnostics snapshot."""

import hashlib


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Unexpected diagnostic helper; refusing ambiguous telemetry transformation")
    result = source.replace(old, new, 1)
    compile(result, "<capacity-telemetry>", "exec")
    return result


def comparator(source):
    anchor = '        "relative_l2": relative_l2,\n'
    return _replace_once(source, anchor, anchor + '        "reference_squared_l2": aa,\n')


def helper(source):
    anchor = '                    "maximum_absolute_error": absolute.item(),\n'
    source = _replace_once(
        source,
        anchor,
        anchor + '                    "comparison_relative_l2": difference.get("relative_l2"),\n'
        '                    "comparison_maximum_absolute_error": difference.get("maximum_absolute_error"),\n'
        '                    "reference_squared_l2": difference.get("reference_squared_l2"),\n'
        '                    "nonfinite": difference.get("nonfinite", False),\n',
    )
    source = _replace_once(
        source,
        "for actual, target in zip(output[key], expected[key], strict=True):",
        "for sample_index, (actual, target) in enumerate(zip(output[key], expected[key], strict=True)):",
    )
    source = _replace_once(
        source,
        '"slot": batch["slot_id"], "field": key, **difference',
        '"slot": batch["slot_id"], "field": key, "sample_index": sample_index, **difference',
    )
    source = _replace_once(
        source,
        "    gate_failures, compute_parameter_checks = [], []\n",
        "    gate_failures, compute_parameter_checks = [], []\n    norm_checks, step_checks = [], []\n",
    )
    source = _replace_once(
        source,
        "        if not norm_passed:\n",
        '        norm_checks.append({"update": update, "actual": float(torch.tensor(norm).item()), "reference": float(torch.tensor(records[update]["norm"]).item()), "passed": norm_passed})\n'
        "        if not norm_passed:\n",
    )
    source = _replace_once(
        source,
        "            if not step_passed:\n",
        '            step_checks.append({"update": update, "name": spec.canonical_name, "actual": int(entry.step), "reference": int(saved["step"].item()), "passed": step_passed})\n'
        "            if not step_passed:\n",
    )
    source = _replace_once(
        source,
        '                "compute_parameter_checks": compute_parameter_checks,\n',
        '                "compute_parameter_checks": compute_parameter_checks,\n'
        '                "norm_checks": norm_checks,\n                "step_checks": step_checks,\n',
    )
    return _replace_once(
        source,
        '                "scope": reference["scope"],',
        '                "rank": dist.get_rank(),\n                "scope": reference["scope"],',
    )


def apply(snapshot):
    paths = {
        snapshot / "tools/compare_tp_pp_runtime_checkpoints.py": comparator,
        snapshot / "tests/megatron/tp_pp_full_source_math_cuda_compare_streamed_reference_v2.py": helper,
    }
    transformed = {}
    provenance = {}
    for path, transform in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular isolated snapshot file: {path}")
        source = path.read_bytes()
        transformed[path] = transform(source.decode()).encode()
        provenance[str(path.relative_to(snapshot))] = {
            "before_sha256": hashlib.sha256(source).hexdigest(),
            "after_sha256": hashlib.sha256(transformed[path]).hexdigest(),
        }
    for path, content in transformed.items():
        path.write_bytes(content)
    return provenance
