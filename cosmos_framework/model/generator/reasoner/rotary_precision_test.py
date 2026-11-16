# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep fractional Action128 positions distinct under mixed precision."""

import pytest
import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRotaryEmbedding
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding


@pytest.mark.parametrize(
    "config_class,rotary_class",
    [
        pytest.param(Qwen3VLTextConfig, Qwen3VLTextRotaryEmbedding, id="dense"),
        pytest.param(Qwen3VLMoeTextConfig, Qwen3VLMoeTextRotaryEmbedding, id="moe"),
    ],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpus(1))])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("fractional", [False, True], ids=["text", "action"])
def test_rotary_autocast_preserves_positions(config_class, rotary_class, device, output_dtype, fractional):
    config = config_class(
        hidden_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
    )
    rotary = rotary_class(config).to(device=device)
    count = 61
    if fractional:
        # A two-second 30 Hz action sequence after the 15000-position modality margin.
        positions = torch.zeros(3, 1, count, device=device, dtype=torch.float32)
        positions[0, 0] = 15300 + torch.arange(count, device=device) * 0.2
    else:
        positions = (15300 + torch.arange(count, device=device)).unsqueeze(0)
    original_positions = positions.clone()
    x = torch.zeros(1, count, 128, device=device, dtype=output_dtype)

    with torch.autocast(device_type=device, enabled=False):
        reference = rotary(x, positions)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        actual = rotary(x, positions)
        assert torch.is_autocast_enabled(device)

    for observed, expected in zip(actual, reference, strict=True):
        assert observed.dtype == output_dtype
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(positions, original_positions, rtol=0, atol=0)

    # Independent scalar-angle construction in float64, without the production
    # phase matmul or apply_interleaved_mrope helper.
    coordinates = positions.double() if fractional else positions.double().unsqueeze(0).expand(3, -1, -1)
    axes = [0] * rotary.inv_freq.numel()
    for axis, offset in ((1, 1), (2, 2)):
        for index in range(offset, config.rope_scaling["mrope_section"][axis] * 3, 3):
            axes[index] = axis
    angles = torch.stack(
        [coordinates[axis] * frequency.double() for axis, frequency in zip(axes, rotary.inv_freq, strict=True)],
        dim=-1,
    )
    expected_cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(output_dtype)
    expected_sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(output_dtype)
    for observed, expected in zip(actual, (expected_cos, expected_sin), strict=True):
        tolerance = 0.005 if output_dtype == torch.bfloat16 else 0.001
        torch.testing.assert_close(observed, expected, rtol=0, atol=tolerance)

    encoded = torch.cat(actual, dim=-1)[0]
    assert torch.unique(encoded, dim=0).shape[0] == count
