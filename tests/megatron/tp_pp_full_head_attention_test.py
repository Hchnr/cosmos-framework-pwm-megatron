import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("world", [2, 4])
def test_full_head_attention_matches_native_gqa_for_all_grad_paths(tmp_path, world):
    env = os.environ.copy()
    env.update(
        ATTENTION_TEST_OUTPUT=str(tmp_path),
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LD_LIBRARY_PATH="",
    )
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH", ""))))
    worker = Path(__file__).with_name("tp_pp_full_head_gloo_worker.py")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        str(worker),
    ]
    subprocess.run(command, check=True, env=env, timeout=120, capture_output=True, text=True)
    for rank in range(world):
        report = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert report["passed"] and report["gradient_patterns"] == 8 and report["world"] == world
        assert report["unused_grad_none"] and report["empty_query"] and report["mask_mismatch_rejected"]


def test_install_only_intercepts_tp_layers_and_restores_context(monkeypatch):
    from types import SimpleNamespace

    import torch
    from megatron.core import parallel_state

    from cosmos_framework.model.generator.megatron_tp_pp.layers import (
        TensorParallelLinear,
    )
    from cosmos_framework.model.generator.mot import small_action_tower as tower
    from tools import tp_pp_full_head_attention_experiment as candidate

    calls = []

    def native(*args, **kwargs):
        calls.append("native")
        return "native"

    def layer_forward(layer, *args, **kwargs):
        return tower.attend(None, None, None, None)

    monkeypatch.setattr(tower, "attend", native)
    monkeypatch.setattr(tower, "small_layer_forward", layer_forward)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: "group")
    monkeypatch.setattr(candidate, "full_head_attention", lambda *args, **kwargs: "tp")
    audit = {"forward_calls": 0, "backward_calls": 0}
    candidate.install(audit)
    plain = SimpleNamespace(self_attn=SimpleNamespace(q_proj=torch.nn.Linear(2, 2)))
    parallel = SimpleNamespace(self_attn=SimpleNamespace(q_proj=TensorParallelLinear.__new__(TensorParallelLinear)))
    assert tower.small_layer_forward(plain) == "native"
    assert tower.small_layer_forward(parallel) == "tp"
    assert tower.attend(None, None, None, None) == "native"
    assert audit["forward_calls"] == 1 and calls == ["native", "native"]
