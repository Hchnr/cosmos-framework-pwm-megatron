# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import numpy as np
import torch

from tools.validate_action128_prefetch import payload_digest


def test_payload_digest_handles_scalar_empty_and_strided_tensors():
    payload = {
        "scalar": torch.tensor(1),
        "empty": torch.empty(0, 4),
        "numpy_empty": np.empty((0, 3)),
        "strided": torch.arange(12).reshape(3, 4).transpose(0, 1),
        "action_loss_mask": torch.tensor([True, False]),
    }
    assert payload_digest(payload) == payload_digest(copy.deepcopy(payload))
    altered = copy.deepcopy(payload)
    altered["action_loss_mask"][1] = True
    assert payload_digest(payload) != payload_digest(altered)


def test_payload_digest_excludes_execution_metadata_but_preserves_pack_boundaries():
    payload = {"action": [[torch.ones(3, 128)]], "_worker_batch_time": 0.5, "_cosmos_loader": {"global_id": 7}}
    changed_timing = copy.deepcopy(payload)
    changed_timing["_worker_batch_time"] = 1000.0
    changed_timing["_cosmos_loader"] = {"global_id": 99}
    assert payload_digest(payload) == payload_digest(changed_timing)
    changed_timing["action"] = [[torch.ones(1, 128)], [torch.ones(2, 128)]]
    assert payload_digest(payload) != payload_digest(changed_timing)
