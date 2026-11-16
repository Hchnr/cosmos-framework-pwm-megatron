# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.configs.base.defaults.compile import CompileConfig, inductor_compile_options


def test_inductor_compile_options_includes_reduction_split_override() -> None:
    config = CompileConfig(
        max_autotune_pointwise=True,
        coordinate_descent_tuning=True,
        mix_order_reduction_split_size=32,
    )

    assert inductor_compile_options(config) == {
        "max_autotune_pointwise": True,
        "coordinate_descent_tuning": True,
        "triton.mix_order_reduction_split_size": 32,
    }


def test_inductor_compile_options_omits_inactive_options() -> None:
    assert inductor_compile_options(CompileConfig()) == {}


@pytest.mark.parametrize("split_size", [0, 7, 31, 256])
def test_reduction_split_override_rejects_unsupported_values(split_size: int) -> None:
    with pytest.raises(ValueError, match="mix_order_reduction_split_size"):
        CompileConfig(mix_order_reduction_split_size=split_size)
