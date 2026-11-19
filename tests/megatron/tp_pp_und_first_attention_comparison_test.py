# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from tools.compare_tp_pp_und_first_attention import difference, verify_runtime_parameters
from tools.run_tp_pp_und_first_attention_capture import UND_MODULES, tensor_digest


def populate(directory):
    for name in UND_MODULES:
        tag = name.replace(".", "_")
        full = torch.arange(32, dtype=torch.float32).view(8, 4) if name == "self_attn.q_proj" else torch.ones(4)
        for kind, rank, size in [("native", 0, 1), *(("tp", rank, 4) for rank in range(4))]:
            weight = full.chunk(4, 0)[rank].clone() if kind == "tp" and name == "self_attn.q_proj" else full.clone()
            value = {
                "weight": weight,
                "bias": None,
                "weight_device": "cuda:0",
                "weight_stride": weight.stride(),
                "weight_sha256": tensor_digest(weight),
                "bias_sha256": None,
                "epsilon": 1e-6,
                "rank": rank,
                "tp_rank": rank,
                "tp_size": size,
            }
            target = directory / "weights" / kind / f"{tag}.rank{rank}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, target)


def test_runtime_weight_shards_match_exactly(tmp_path):
    populate(tmp_path)
    report, _ = verify_runtime_parameters(tmp_path)
    assert all(all(item["weight_equal_by_rank"]) and all(item["eps_equal_by_rank"]) for item in report.values())


def test_bad_weight_hash_cannot_report_success(tmp_path):
    populate(tmp_path)
    path = tmp_path / "weights/tp/self_attn_q_proj.rank2.pt"
    data = torch.load(path, weights_only=True)
    data["weight"][0, 0] += 1
    torch.save(data, path)
    with pytest.raises(ValueError, match="Invalid captured runtime parameter"):
        verify_runtime_parameters(tmp_path)


def test_hash_valid_but_different_weights_are_explicit(tmp_path):
    populate(tmp_path)
    path = tmp_path / "weights/tp/self_attn_q_proj.rank2.pt"
    data = torch.load(path, weights_only=True)
    data["weight"][0, 0] += 1
    data["weight_sha256"] = tensor_digest(data["weight"])
    torch.save(data, path)
    report, _ = verify_runtime_parameters(tmp_path)
    assert report["self_attn.q_proj"]["weight_equal_by_rank"] == [True, True, False, True]


def test_difference_checks_dtype_and_shape():
    with pytest.raises(ValueError):
        difference(torch.ones(3), torch.ones(2))
    assert difference(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))["exact"]
