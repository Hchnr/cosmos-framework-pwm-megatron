# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU guard regression: imported CLI, -m CLI, and an old-entry control.

These tests intercept installation equally in all three subprocesses; they do
not test the transformed model, NCCL or Inductor. PYTHONPATH must include the
diagnostic worktree, as it did in the failed benchmark invocation.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "tools/run_tp_pp_async_row_reduction.py"
MODULE = "tools.run_tp_pp_async_row_reduction"

SITE = """\
import builtins
import importlib.abc
import importlib.machinery
import sys

TARGET = 'tools.run_tp_pp_async_row_reduction'

def install_stub(target):
    builtins._async_guard_target = target
    print('INSTALL_MODULE=' + target.__module__, flush=True)

builtins._async_guard_install_stub = install_stub

class Loader:
    def __init__(self, original): self.original = original
    def __getattr__(self, name): return getattr(self.original, name)
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        module.install = lambda: install_stub(module.async_update)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET: return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None: spec.loader = Loader(spec.loader)
        return spec

sys.meta_path.insert(0, Finder())
"""

TRAINING = """\
import builtins
import os
import sys
from types import SimpleNamespace as NS
import torch
from torch._dynamo.guards import GuardBuilder

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_available()
target = builtins._async_guard_target
# The legacy helper and this runpy training fixture have distinct __main__
# namespaces. Satisfy unrelated global guards so the negative control reaches
# the specific missing finish closure; never install finish itself here.
for name in ('F', 'launch', 'gate_up'):
    globals()[name] = target.__globals__[name]
assert not hasattr(sys.modules['__main__'], 'finish')
expected_module = '__main__' if os.environ['ASYNC_ENTRY_CASE'] == 'legacy' else 'tools.run_tp_pp_async_row_reduction'
assert target.__module__ == expected_module
assert target.__globals__['finish'].__module__ == expected_module

seen = []
original = GuardBuilder.CLOSURE_MATCH
def observe_guard(builder, guard):
    if guard.name.endswith('.finish'):
        seen.append(guard.name)
        print('CLOSURE_MATCH=' + guard.name, flush=True)
    return original(builder, guard)
GuardBuilder.CLOSURE_MATCH = observe_guard

class Row:
    column = False
    tp_size = 1
    weight = None
    bias = None
    def _async_row_local(self, value, weight): return value
def identity(value): return value
def mlp(): return NS(gate_proj=identity, up_proj=identity, down_proj=Row(), act_fn=identity)
layer = NS(mlp=mlp(), mlp_moe_gen=mlp(), post_attention_layernorm=identity,
           post_attention_layernorm_moe_gen=identity)
block = NS(o_proj=Row(), down_proj=Row(), gate_proj=identity, up_proj=identity,
           post_attention_layernorm=identity)
attn = NS(o_proj=Row(), o_proj_moe_gen=Row())
def graph(value): return target(layer, attn, block, value, value, value, value, value, value)

compilations = []
def eager_backend(graph_module, example_inputs):
    compilations.append(1)
    return torch._dynamo.lookup_backend('eager')(graph_module, example_inputs)
compiled = torch.compile(graph, backend=eager_backend, fullgraph=True)
try:
    for value in (torch.tensor([2.]), torch.tensor([4.])):
        for actual, expected in zip(compiled(value), graph(value), strict=True):
            torch.testing.assert_close(actual, expected)
except Exception as error:
    if os.environ['ASYNC_ENTRY_CASE'] != 'legacy': raise
    assert "module '__main__' has no attribute 'finish'" in str(error), repr(error)
    assert any('__import___main__' in name and name.endswith('.finish') for name in seen), seen
    print('LEGACY_EXPECTED_FINISH_GUARD_FAILURE', flush=True)
else:
    assert os.environ['ASYNC_ENTRY_CASE'] != 'legacy', 'old entry unexpectedly passed'
    assert any('__import_tools_dot_run_tp_pp_async_row_reduction' in name and name.endswith('.finish')
               for name in seen), seen
    assert len(compilations) == 1, compilations
    print('STABLE_FINISH_GUARD_REEVALUATED', flush=True)
"""


@pytest.mark.parametrize("mode", ("script", "module", "legacy"))
def test_entry_module_guard_with_old_entry_control(tmp_path, mode):
    (tmp_path / "sitecustomize.py").write_text(SITE)
    training = tmp_path / "training.py"
    benchmark = tmp_path / "benchmark.py"
    training.write_text(TRAINING)
    benchmark.write_text(f"import runpy\nrunpy.run_path({str(training)!r}, run_name='__main__')\n")
    if mode == "legacy":
        original = ENTRY.read_text()
        anchor = "    install()\n    sys.path.insert(0, str(args.entry.resolve().parent))"
        assert original.count(anchor) == 1
        original = original.replace(
            anchor,
            "    import builtins\n    builtins._async_guard_install_stub(async_update)\n"
            "    sys.path.insert(0, str(args.entry.resolve().parent))",
        )
        marker = 'if __name__ == "__main__":\n'
        assert original.count(marker) == 1
        legacy = tmp_path / "legacy_async_entry.py"
        legacy.write_text(original.split(marker)[0] + marker + "    main()\n")
        launch = [str(legacy)]
    elif mode == "module":
        launch = ["-m", MODULE]
    else:
        launch = [str(ENTRY)]
    pycache = tmp_path / "pycache"
    pycache.mkdir()
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(tmp_path), str(ROOT))),
        ASYNC_ENTRY_CASE=mode,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(pycache),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        LD_LIBRARY_PATH="",
    )
    run = subprocess.run(
        [sys.executable, *launch, "--entry", str(benchmark)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    print(f"[{mode}] stdout:\n{run.stdout}\n[{mode}] stderr:\n{run.stderr}")
    assert run.returncode == 0, f"{mode} exit {run.returncode}\n{run.stdout}\n{run.stderr}"
    if mode == "legacy":
        assert "INSTALL_MODULE=__main__" in run.stdout
        assert "LEGACY_EXPECTED_FINISH_GUARD_FAILURE" in run.stdout
    else:
        assert "INSTALL_MODULE=" + MODULE in run.stdout
        assert "STABLE_FINISH_GUARD_REEVALUATED" in run.stdout
