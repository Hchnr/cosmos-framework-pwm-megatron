"""Performance timing must survive the standard logging callback's reset."""

import json
from types import SimpleNamespace

import torch

from cosmos_framework.callbacks.action128_performance import Action128Performance


def test_phase_snapshot_survives_reset_without_cuda_synchronization(tmp_path, monkeypatch):
    times = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr("cosmos_framework.callbacks.action128_performance.time.perf_counter", lambda: next(times))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("performance callback added a CUDA synchronization")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    audit = Action128Performance()
    audit.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    timer = SimpleNamespace(results={"forward": [1.0], "backward": [2.0], "optimizer_step": [0.5]})
    audit.trainer = SimpleNamespace(training_timer=timer)
    audit.on_train_start(None)
    audit.on_before_dataloading()
    audit.on_after_dataloading()
    audit.on_training_step_batch_end(None, None, None, None)
    timer.results = {key: [] for key in timer.results}
    data = {
        "packed_sequence": SimpleNamespace(vision=SimpleNamespace(token_shapes=[(2, 3, 4)])),
        "dataset_name": "test",
        "__key__": ["test/1/0"],
    }
    output = dict(und_token_length=10, vision_token_length=24, action_token_length=8)
    audit.on_training_step_end(None, data, output, None, iteration=1)
    row = json.loads(audit.path.read_text())
    assert row["step_wall_s"] == 5.0 and row["dataload_wall_s"] == 2.0
    assert row["host_phase_s"] == {"forward": 1.0, "backward": 2.0, "optimizer_step": 0.5}
