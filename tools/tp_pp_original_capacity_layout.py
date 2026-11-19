# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Provide the native capacity as the initial PP layout for FP32 diagnostics."""


def initial_shapes(network, *, original_capacity):
    if not original_capacity:
        return ((network.fixed_und_tokens, 4096), (12, 4096), (6, 1280))
    vision = network.fixed_gen_tokens - network.fixed_action_tokens
    if min(network.fixed_und_tokens, vision, network.fixed_action_tokens) <= 0:
        raise ValueError("Original-capacity token limits must be positive")
    return ((network.fixed_und_tokens, 4096), (vision, 4096), (network.fixed_action_tokens, 1280))
