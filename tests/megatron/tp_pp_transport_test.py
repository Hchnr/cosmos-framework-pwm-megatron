# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit tensor schemas preserve values and reject detached trainable graphs."""

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.megatron_tp_pp.payload import PayloadLayout
from cosmos_framework.training_backends.megatron_tp_pp.transport import decode, encode


def test_tensor_schema_preserves_dataclasses_shapes_dtypes_and_aliases():
    shared = torch.arange(12, dtype=torch.float32).view(3, 4)
    source = {
        "slot_id": (1200, 2, 191),
        "layout": PayloadLayout(((3, 64), (4, 64), (2, 32)), context_width=96),
        "packed_sequence": PackedSequence(
            text_ids=torch.tensor([1, 2]), action=ModalityData(tokens=[shared], timesteps=torch.tensor([0.5]))
        ),
        "extra": {False: [torch.tensor(True), torch.tensor(7), shared, torch.empty(0, dtype=torch.bfloat16)]},
    }
    header, buffers = encode(source, device="cpu")
    actual = decode(header, buffers)
    assert actual["slot_id"] == source["slot_id"] and actual["layout"] == source["layout"]
    assert isinstance(actual["packed_sequence"], PackedSequence)
    assert isinstance(actual["packed_sequence"].action, ModalityData)
    torch.testing.assert_close(actual["packed_sequence"].action.tokens[0], shared, rtol=0, atol=0)
    assert actual["packed_sequence"].action.tokens[0] is actual["extra"][False][2]
    assert actual["extra"][False][0].dtype == torch.bool
    assert actual["extra"][False][1].shape == ()
    assert actual["extra"][False][3].dtype == torch.bfloat16


def test_trainable_tensors_and_kernel_objects_cannot_enter_metadata_transport():
    with pytest.raises(ValueError, match="differentiable PP"):
        encode({"activation": torch.randn(3, requires_grad=True)}, device="cpu")
    with pytest.raises(TypeError):
        encode({"kernel_mask": object()}, device="cpu")
