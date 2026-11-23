# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.build_tp_pp_original_capacity_helper import transform
from tools.tp_pp_original_capacity_layout import initial_shapes


def test_original_capacity_uses_native_video_and_action_limits():
    network = SimpleNamespace(fixed_und_tokens=11776, fixed_gen_tokens=62592, fixed_action_tokens=2944)
    assert initial_shapes(network, original_capacity=True) == ((11776, 4096), (59648, 4096), (2944, 1280))
    assert initial_shapes(network, original_capacity=False) == ((11776, 4096), (12, 4096), (6, 1280))
    network.fixed_action_tokens = network.fixed_gen_tokens
    with pytest.raises(ValueError, match="positive"):
        initial_shapes(network, original_capacity=True)


@pytest.mark.parametrize("legacy_pp", [False, True], ids=["dynamic-pp", "legacy-pp2"])
def test_oracle_helper_transform_is_explicit_and_rejects_double_application(legacy_pp):
    source = Path("tests/megatron/tp_pp_full_source_math.py").read_text()
    dynamic_pp = '"pipeline_parallel_size": topology.pipeline_model_parallel_size,'
    assert source.count(dynamic_pp) == 1
    if legacy_pp:
        source = source.replace(dynamic_pp, '"pipeline_parallel_size": 2,', 1)
    result = transform(source)
    assert "initial_shapes(network, original_capacity=args.original_capacity)" in result
    assert '"pipeline_parallel_size": 2,' not in result
    assert '"pipeline_parallel_size": topology.pipeline_model_parallel_size,' in result
    report = [
        value
        for node in ast.walk(ast.parse(result))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "pipeline_parallel_size"
    ]
    assert len(report) == 1
    expression = compile(ast.Expression(body=report[0]), "<pp-report>", "eval")
    for actual_pp in (1, 2):
        assert eval(expression, {"topology": SimpleNamespace(pipeline_model_parallel_size=actual_pp)}) == actual_pp
    with pytest.raises(ValueError, match="Unexpected oracle source"):
        transform(result)


@pytest.mark.parametrize(
    "replacement",
    [
        '"other_parallel_size": topology.pipeline_model_parallel_size,',
        '"pipeline_parallel_size": topology.pipeline_model_parallel_size,\n                "pipeline_parallel_size": topology.pipeline_model_parallel_size,',
        '"pipeline_parallel_size": topology.pipeline_model_parallel_size,\n                "pipeline_parallel_size": 2,',
        '"pipeline_parallel_size": 2,\n                "pipeline_parallel_size": 2,',
        '"pipeline_parallel_size": topology.pipeline_model_parallel_size,\n                "pipeline_parallel_size": 3,',
        '"pipeline_parallel_size": 3,',
        '"pipeline_parallel_size": topology.tensor_model_parallel_size,',
    ],
    ids=[
        "missing",
        "duplicate-dynamic",
        "mixed",
        "duplicate-legacy",
        "conflicting",
        "wrong-constant",
        "wrong-topology",
    ],
)
def test_oracle_helper_rejects_missing_duplicate_or_conflicting_pp_report(replacement):
    source = Path("tests/megatron/tp_pp_full_source_math.py").read_text()
    dynamic_pp = '"pipeline_parallel_size": topology.pipeline_model_parallel_size,'
    assert source.count(dynamic_pp) == 1
    source = source.replace(dynamic_pp, replacement, 1)
    with pytest.raises(ValueError, match="Unexpected oracle source"):
        transform(source)


def test_oracle_helper_rejects_separate_pp_overwrite():
    source = Path("tests/megatron/tp_pp_full_source_math.py").read_text()
    source += '\nreport = {}\nreport["pipeline_parallel_size"] = 3\n'
    with pytest.raises(ValueError, match="Unexpected oracle source"):
        transform(source)
