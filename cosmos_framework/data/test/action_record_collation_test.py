# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord, ActionProcessor
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator


def test_catalog_collator_preserves_action_externalization_metadata():
    record = ActionProcessingRecord(raw_action_dim=3, action_normalizer=None)
    batch = VFMListCollator().collate(
        [
            {"action": torch.ones(5, 8), "action_processing_record": record},
            {"action": torch.ones(3, 8), "action_processing_record": record},
        ]
    )
    assert batch["action_processing_record"] == [record, record]
    for index, length in enumerate((5, 3)):
        external = ActionProcessor.postprocess_action(
            batch["action"][index][0], batch["action_processing_record"][index]
        )
        assert external.shape == (length, 3)
