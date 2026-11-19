# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.qualify_tp_pp_sharded_vae_gpu import compare_wire, validate_manifest


def test_actual_wire_encoding_covers_packed_tokens_and_noise():
    from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised
    from cosmos_framework.training_backends.megatron_tp_pp.transport import encode

    packed = PackedSequence(
        sample_lens=[3],
        split_lens=[1, 2],
        attn_modes=["causal", "full"],
        sequence_length=3,
        text_ids=torch.tensor([1]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.arange(3).expand(3, -1),
        vision=ModalityData(
            sequence_indexes=torch.tensor([1, 2]),
            mse_loss_indexes=torch.tensor([2]),
            tokens=[torch.tensor([[1.0], [2.0]])],
            condition_mask=[torch.tensor([1.0, 0.0])],
        ),
    )
    noise = GenerationDataNoised(
        batch_size=1,
        epsilon_vision=[torch.tensor([0.2, 0.3])],
        xt_tokens_vision=[torch.tensor([1.0, 2.0])],
        vt_target_vision=[torch.tensor([0.8, 1.7])],
    )
    baseline = {"packed_sequence": packed, "loss_inputs": {"gen_data_noised": noise}, "sample_count": 1}
    candidate = copy.deepcopy(baseline)
    h1, t1 = encode(baseline, device="cpu")
    h2, t2 = encode(candidate, device="cpu")
    assert compare_wire(h1, t1, h2, t2, atol=0, rtol=0)["torch.float32"]["exact"]
    candidate["loss_inputs"]["gen_data_noised"].epsilon_vision[0][0] += 0.1
    changed_h, changed_t = encode(candidate, device="cpu")
    with pytest.raises(AssertionError):
        compare_wire(h1, t1, changed_h, changed_t, atol=0, rtol=0)


def test_complete_wire_checks_metadata_floating_integer_and_nan():
    header = b"same structure"
    reference = {"torch.float32": torch.tensor([1.0, 2.0]), "torch.int64": torch.tensor([4, 5])}
    identical = {name: value.clone() for name, value in reference.items()}
    stats = compare_wire(header, reference, header, identical, atol=0.0, rtol=0.0)
    assert stats["torch.float32"]["exact"] and stats["torch.int64"]["bytes"] == 16
    with pytest.raises(ValueError, match="structure"):
        compare_wire(header, reference, b"changed structure", identical, atol=0, rtol=0)
    with pytest.raises(ValueError, match="shape/dtype"):
        compare_wire(header, reference, header, {**identical, "torch.float32": torch.ones(1)}, atol=0, rtol=0)
    with pytest.raises(ValueError, match="integer/bool"):
        compare_wire(header, reference, header, {**identical, "torch.int64": torch.tensor([4, 6])}, atol=0, rtol=0)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_wire(
            header, reference, header, {**identical, "torch.float32": torch.tensor([1.0, float("nan")])}, atol=1, rtol=1
        )
    shifted = {**identical, "torch.float32": torch.tensor([1.0, 2.001])}
    with pytest.raises(AssertionError):
        compare_wire(header, reference, header, shifted, atol=0, rtol=0)
    assert not compare_wire(header, reference, header, shifted, atol=0.01, rtol=0)["torch.float32"]["exact"]


def test_capture_manifest_requires_order_and_positive_samples():
    good = {
        "passed": True,
        "records": [
            {"slot": [1200, 0, 0], "samples": 9, "sha256": "a"},
            {"slot": [1200, 1, 0], "samples": 11, "sha256": "b"},
        ],
    }
    assert validate_manifest(good) == 2
    for edit in ("failed", "order", "samples", "sha256"):
        bad = copy.deepcopy(good)
        if edit == "failed":
            bad["passed"] = False
        elif edit == "order":
            bad["records"][1]["slot"][1] = 2
        elif edit == "samples":
            bad["records"][1]["samples"] = 0
        else:
            bad["records"][1]["sha256"] = ""
        with pytest.raises(ValueError):
            validate_manifest(bad)


def test_gpu_qualification_help_does_not_require_gpu_or_pythonpath(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/qualify_tp_pp_sharded_vae_gpu.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--raw-fixture" in result.stdout
    assert "--deterministic-algorithms" in result.stdout
