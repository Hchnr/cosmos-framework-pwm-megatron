# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from tools.qualify_tp_pp_nested_checkpoint_memory import run_comparison


def test_outer_recomputation_compares_all_gradients_without_claiming_cuda_result():
    result = run_comparison(tokens=13, width=7, intermediate=17, chunk_tokens=4, device=torch.device("cpu"))
    assert result["outputs_and_all_gradients_close"]
    assert result["plain"]["outer_block_calls"] >= 2
    assert result["nested"]["outer_block_calls"] >= 2
    assert result["cuda_peak_reduced"] is None
    assert not result["passed"]
    assert not result["full_model_acceptance"]


@pytest.mark.parametrize("tokens,width,intermediate,chunk", [(0, 2, 3, 1), (4, 0, 3, 1), (4, 2, 3, 4)])
def test_rejects_invalid_or_unsplit_shapes(tokens, width, intermediate, chunk):
    with pytest.raises(ValueError):
        run_comparison(
            tokens=tokens, width=width, intermediate=intermediate, chunk_tokens=chunk, device=torch.device("cpu")
        )
