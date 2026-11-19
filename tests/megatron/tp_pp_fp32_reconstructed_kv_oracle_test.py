# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qualify_tp_pp_fp32_reconstructed_kv import validate
from tools.tp_pp_fp32_reconstructed_kv import Observation


def valid_inputs():
    smoke = {
        "passed": True,
        "compiled": False,
        "groups": {"tp": [0, 1], "pp": [0], "dp": [0]},
        "relative_gradient_l2": {f"{i}/layer.action_block_moe_gen_action.weight": 1e-6 for i in range(4)},
    }
    observation = SimpleNamespace(
        data={
            "stage_parameter_dtypes": ["torch.float32"],
            "modules": {
                "layers.0.self_attn.k_proj": {"forward_calls": 1},
                "layers.0.self_attn.v_proj": {"forward_calls": 1},
            },
        }
    )
    exercise = {"missing_action": 1, "missing_noisy_targets": 1, "reference_parameter_dtypes": ["torch.float32"]}
    return smoke, observation, exercise


def test_qualification_requires_actual_kv_calls_dtype_and_missing_modalities():
    smoke, observation, exercise = valid_inputs()
    assert validate(smoke, observation, exercise)["gradient_entries"] == 4
    observation.data["modules"]["layers.0.self_attn.k_proj"]["forward_calls"] = 0
    with pytest.raises(ValueError, match="No reconstructed K"):
        validate(smoke, observation, exercise)
    smoke, observation, exercise = valid_inputs()
    observation.data["stage_parameter_dtypes"] = ["torch.bfloat16"]
    with pytest.raises(ValueError, match="must all be FP32"):
        validate(smoke, observation, exercise)
    smoke, observation, exercise = valid_inputs()
    exercise["missing_action"] = 0
    with pytest.raises(ValueError, match="missing-modality"):
        validate(smoke, observation, exercise)


def test_sidecar_fails_closed_without_forward(tmp_path):
    observation = Observation(tmp_path, 0)
    with pytest.raises(RuntimeError, match="No reconstructed K/V forward"):
        observation.finish()
    with pytest.raises(FileExistsError):
        Observation(tmp_path, 0)


def test_qualification_entry_help_needs_no_gpu_or_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "tools/qualify_tp_pp_fp32_reconstructed_kv.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "rank-divergent" in result.stdout
