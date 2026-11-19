# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical numerical exports cover each TP shard and every DP fragment."""

import copy

import pytest
import torch

from cosmos_framework.training_backends.megatron_tp_pp.numerical_state import validate_fragment_coverage
from tools.compare_tp_pp_native_numerical_state import plain


def fragments():
    name = "net.language_model.model.layers.0.self_attn.o_proj.weight"
    parameters = {name: {"optimizer_fields": ["master_param"], "global_shape": [3, 8]}}
    rows = [
        {
            "layout": {"canonical_name": name, "shape": [3, 8], "axis": 1, "local_shape": [3, 4], "tensor_rank": tp},
            "start": start,
            "end": start + 6,
        }
        for tp in (0, 1)
        for start in (0, 6)
    ]
    return rows, parameters


def test_optimizer_fragment_coverage_includes_both_parallel_dimensions():
    rows, parameters = fragments()
    validate_fragment_coverage(rows, parameters, 2)
    with pytest.raises(ValueError, match="Missing tensor-parallel"):
        validate_fragment_coverage(rows[:2], parameters, 2)
    with pytest.raises(ValueError, match="Overlapping or missing"):
        validate_fragment_coverage(rows + rows[:1], parameters, 2)
    with pytest.raises(ValueError, match="Incomplete local"):
        validate_fragment_coverage(rows[:1] + rows[2:], parameters, 2)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["layout"]["shape"] = [6, 4]
    with pytest.raises(ValueError, match="Canonical optimizer shape"):
        validate_fragment_coverage(corrupted, parameters, 2)


def test_torch_payload_tuple_metadata_matches_json_without_changing_values():
    assert plain({"shape": (3, 8), "step": torch.tensor([1205]), "dtype": "float32"}) == {
        "shape": [3, 8],
        "step": [1205],
        "dtype": "float32",
    }
