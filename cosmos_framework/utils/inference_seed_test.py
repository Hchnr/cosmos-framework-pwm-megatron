# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.utils.inference_seed import modality_noise_seed


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
def test_modality_seeds_preserve_video_and_separate_streams_at_uint32_boundaries(seed):
    derived = [modality_noise_seed(seed, kind) for kind in ("vision", "action", "sound")]
    assert derived[0] == seed
    assert len(set(derived)) == 3
    assert all(0 <= value < 2**32 for value in derived)
    assert derived == [modality_noise_seed(seed, kind) for kind in ("vision", "action", "sound")]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_modality_seeds_reject_out_of_range_seed(seed):
    with pytest.raises(ValueError, match="uint32"):
        modality_noise_seed(seed, "action")
