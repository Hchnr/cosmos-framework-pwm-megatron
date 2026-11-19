# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

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


def test_oracle_helper_transform_is_explicit_and_rejects_double_application():
    source = Path("tests/megatron/tp_pp_full_source_math.py").read_text()
    result = transform(source)
    assert "initial_shapes(network, original_capacity=args.original_capacity)" in result
    with pytest.raises(ValueError, match="Unexpected oracle source"):
        transform(result)
