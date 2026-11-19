# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
import types

import pytest

from tools.run_tp_pp_untraced_capacity_math import validate_contract

COMMON = [
    "--original-capacity",
    "--retain-full-text",
    "--deterministic",
    "--objective",
    "native_rf",
    "--attention-backend",
    "flex",
]
NATIVE = [
    "--mode",
    "native",
    "--gpu-leaf-reference",
    "--preserve-saved-layout",
    "--native-device",
    "cuda",
    "--native-optimizer",
    "native_cuda",
]
TP = ["--mode", "tp", "--tp", "4", "--native-adam-arithmetic"]


def test_original_capacity_contract_accepts_valid_native_and_tp():
    assert validate_contract(COMMON + NATIVE, None, 4096) == "native"
    assert validate_contract(COMMON + TP, object(), 4096) == "tp"


@pytest.mark.parametrize("mutation", [[], ["--replay-native-state"], ["--original-capacity"]])
def test_rejects_weakened_or_ambiguous_capacity_contract(mutation):
    arguments = COMMON + TP + mutation if mutation else COMMON[1:] + TP
    with pytest.raises(ValueError):
        validate_contract(arguments, object(), 4096)


def test_native_uses_flat_patch_on_the_imported_module(monkeypatch):
    from tools import run_tp_pp_checkpointed_chunked_mlp_math as chunks
    from tools import run_tp_pp_flat_checkpoint_math as flat
    from tools import run_tp_pp_fp32_native_ac as native
    from tools import run_tp_pp_untraced_capacity_math as runner

    calls = []
    monkeypatch.setattr(flat, "install", lambda: calls.append("flat"))
    monkeypatch.setattr(chunks, "install", lambda size: calls.append(("chunks", size)))
    monkeypatch.setattr(native, "install_native_checkpointing", lambda: calls.append("native"))
    runner.install("native", 4096, None)
    assert calls == ["flat", ("chunks", 4096), "native"]


def test_flat_install_replaces_native_ac_function_and_wraps_real_layers(monkeypatch):
    from tools import run_tp_pp_flat_checkpoint_math as flat
    from tools import run_tp_pp_fp32_native_ac as native

    stage_module = types.ModuleType("cosmos_framework.model.generator.megatron_tp_pp.stage")
    stage_module.PipelineStage = type("PipelineStage", (), {})
    reference = types.ModuleType("tp_pp_gpu_leaf_reference")
    reference.place_gpu_trainables = lambda model: "gpu-leaf-handles"
    monkeypatch.setitem(sys.modules, stage_module.__name__, stage_module)
    monkeypatch.setitem(sys.modules, reference.__name__, reference)
    original_install = native.install_native_checkpointing
    monkeypatch.setattr(native, "install_native_checkpointing", original_install)
    wrapped = []

    def checkpoint_wrapper(layer, *, checkpoint_fn, preserve_rng_state):
        assert checkpoint_fn is flat.flat_checkpoint
        assert preserve_rng_state is True
        wrapped.append(layer)
        return ("flat-checkpoint", layer)

    monkeypatch.setattr(flat, "checkpoint_wrapper", checkpoint_wrapper)
    flat.install()
    native.install_native_checkpointing()
    model = types.SimpleNamespace(language_model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[1, 2])))
    assert reference.place_gpu_trainables(model) == "gpu-leaf-handles"
    assert wrapped == [1, 2]
    assert model.language_model.model.layers == [("flat-checkpoint", 1), ("flat-checkpoint", 2)]
