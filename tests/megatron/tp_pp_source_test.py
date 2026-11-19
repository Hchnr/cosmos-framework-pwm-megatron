# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read sharded checkpoint slices and non-tensor optimizer fields exactly."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader


@pytest.mark.parametrize("axis", [0, 1])
def test_reads_only_local_tp_slice_and_scalar_bytes(tmp_path, axis):
    tensor = torch.arange(96, dtype=torch.float32).view(8, 12)
    dcp.save({"weight": tensor, "betas": (0.9, 0.95), "step": torch.tensor(1200)}, checkpoint_id=tmp_path)
    reader = CanonicalDCPReader(tmp_path)
    for rank in range(2):
        actual = reader.read("weight", axis=axis, rank=rank, size=2)
        torch.testing.assert_close(actual, tensor.chunk(2, dim=axis)[rank], atol=0, rtol=0)
    assert reader.read("betas") == (0.9, 0.95)
    assert reader.read("step").item() == 1200
    together = reader.read_many(["weight", "betas", "step"])
    torch.testing.assert_close(together["weight"], tensor, atol=0, rtol=0)
    assert together["betas"] == (0.9, 0.95)
    assert together["step"].item() == 1200
