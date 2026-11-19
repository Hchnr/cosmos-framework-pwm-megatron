"""CPU integration contract for the isolated default-runtime patch."""

import importlib
import importlib.util
import json
import os
import sys
import types
from contextlib import nullcontext

import pytest
import torch
from torch import nn


@pytest.fixture
def runtime(monkeypatch):
    candidate = os.environ.get("COSMOS_TP_PP_DRIVER_CANDIDATE")
    if candidate:
        name = "cosmos_framework.training_backends.megatron_tp_pp._candidate_driver"
        spec = importlib.util.spec_from_file_location(name, candidate)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("cosmos_framework.training_backends.megatron_tp_pp.driver")
    schedules = types.ModuleType("megatron.core.pipeline_parallel.schedules")
    failure = {"after": None}

    def backward(loss):
        loss.backward()

    def execute(*, forward_step_func, data_iterator, model, num_microbatches, **kwargs):
        result = []
        for index in range(num_microbatches):
            output, loss_fn = forward_step_func(data_iterator, model)
            loss, metrics = loss_fn(output)
            schedules.backward_step(loss / num_microbatches)
            result.append(metrics)
            if index + 1 == failure["after"]:
                raise RuntimeError("intentional schedule failure")
        return result

    schedules.backward_step = backward
    schedules.get_forward_backward_func = lambda: execute
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    pipeline = types.ModuleType("megatron.core.pipeline_parallel")
    megatron.core, core.pipeline_parallel, pipeline.schedules = core, pipeline, schedules
    for key, value in {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.pipeline_parallel": pipeline,
        "megatron.core.pipeline_parallel.schedules": schedules,
    }.items():
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "nvtx_range", lambda *args, **kwargs: nullcontext())
    monkeypatch.delenv("COSMOS_TP_PP_PROFILE_SLOTS", raising=False)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 2)

        def forward(self, batch):
            return self.layer(batch["x"])

        def zero_grad_buffer(self):
            pass

    def make(interval, root):
        torch.manual_seed(42)
        network = Network()
        driver = module.ModelParallelDriver.__new__(module.ModelParallelDriver)
        driver.model = network
        driver.optimizer = types.SimpleNamespace(zero_grad=lambda: network.zero_grad(set_to_none=True))
        driver.topology = types.SimpleNamespace(pipeline_model_parallel_size=1)
        driver.stage = types.SimpleNamespace(config=types.SimpleNamespace(variable_seq_lengths=False))
        driver.microbatches, driver.iteration = 4, 1200
        driver.microbatch_gc_interval = interval
        driver._microbatch_gc_trace_root = root
        driver._profile_finished, driver._used = False, set()
        return driver

    return make, schedules, backward, failure


def batches():
    return [{"x": torch.arange(15).reshape(5, 3).float() / (index + 10)} for index in range(4)]


def loss(output, batch):
    value = output.square().mean()
    return value, {"loss": value.detach()}


def test_gc_driver_scope_preserves_gradients_and_emits_complete_trace(runtime, tmp_path):
    make, schedules, original, _ = runtime
    reference = make(0, None)
    reference.forward_backward(batches(), loss)
    candidate = make(2, tmp_path)
    result = candidate.forward_backward(batches(), loss)
    assert len(result) == 4
    assert all(
        torch.equal(a.grad, b.grad)
        for a, b in zip(reference.model.parameters(), candidate.model.parameters(), strict=True)
    )
    assert schedules.backward_step is original
    rows = [json.loads(line) for line in (tmp_path / "microbatch_gc.rank0.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["iteration"] == 1201 and row["interval"] == 2
    assert row["schedule_completed"] and row["completed_backwards"] == 4
    assert [r["completed_backwards"] for r in row["collections"]] == [2, 4]


def test_failed_schedule_restores_backward_and_records_incomplete_scope(runtime, tmp_path):
    make, schedules, original, failure = runtime
    failure["after"] = 2
    candidate = make(2, tmp_path)
    with pytest.raises(RuntimeError, match="intentional schedule failure"):
        candidate.forward_backward(batches(), loss)
    assert schedules.backward_step is original
    row = json.loads((tmp_path / "microbatch_gc.rank0.jsonl").read_text())
    assert not row["schedule_completed"] and row["completed_backwards"] == 2
    assert [r["completed_backwards"] for r in row["collections"]] == [2]
