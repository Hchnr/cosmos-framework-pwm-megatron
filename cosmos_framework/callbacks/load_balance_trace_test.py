# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.callbacks.load_balance_trace as trace_module
from cosmos_framework.callbacks.load_balance_trace import LoadBalanceTrace
from cosmos_framework.data.generator.sequence_packing import SequencePlan

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _model(*, cp_rank: int = 0, cp_slot: int = 0, cp_size: int = 1) -> SimpleNamespace:
    parallel_dims = SimpleNamespace(
        cp_enabled=cp_size > 1,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_mesh=None,
        dp_shard=4,
        dp_replicate=2,
    )
    return SimpleNamespace(parallel_dims=parallel_dims, _cp_window_slot=cp_slot)


def _batch() -> dict[str, object]:
    return {
        "__key__": ["video-a", "video-b"],
        "dataset_name": ["video", "video"],
        "source_dataset_name": ["bridge", "bridge"],
        "text_token_ids": [[torch.zeros(7, dtype=torch.int64)], [torch.zeros(3, dtype=torch.int64)]],
        "video": [
            [torch.zeros(3, 5, 8, 8, dtype=torch.uint8)],
            [torch.zeros(3, 9, 8, 8, dtype=torch.uint8)],
        ],
        "sequence_plan": [
            SequencePlan(has_text=True, has_vision=True),
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0, 1]),
        ],
        "_dropped_count": 2,
    }


def _output() -> dict[str, object]:
    return {
        "task_type": "generation",
        "fixed_profile": "generation",
        "und_token_length": 10,
        "gen_token_length": 20,
        "physical_und_token_length": 16,
        "physical_gen_token_length": 24,
        "vision_token_length": 20,
        "batch_size": 2,
        "split_lens": [4, 8, 6, 12],
        "attn_modes": ["causal", "full", "causal", "full"],
        "vae_pixel_shapes": [(5, 8, 8), (9, 8, 8)],
    }


def _run_batch(
    callback: LoadBalanceTrace,
    model: SimpleNamespace,
    *,
    iteration: int,
    batch: dict[str, object] | None = None,
) -> None:
    data = batch or _batch()
    callback.on_training_step_batch_start(model, data, iteration=iteration)
    callback.on_before_forward(iteration=iteration)
    callback.on_after_forward(iteration=iteration)
    callback.on_before_backward(model, torch.tensor(1.0), iteration=iteration)
    callback.on_after_backward(model, iteration=iteration)
    callback.on_training_step_batch_end(model, data, _output(), torch.tensor(1.0), iteration=iteration)


def _read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_disabled_callback_has_no_side_effects(tmp_path: Path) -> None:
    output_dir = tmp_path / "trace"
    callback = LoadBalanceTrace(enabled=False, output_dir=str(output_dir))
    model = _model()

    callback.on_train_start(model, iteration=0)
    _run_batch(callback, model, iteration=0)
    callback.on_train_end(model, iteration=1)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"enabled": True}, "output_dir is required"),
        ({"flush_every_n_batches": 0}, "flush_every_n_batches"),
        ({"max_samples_per_record": 0}, "max_samples_per_record"),
    ],
)
def test_constructor_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LoadBalanceTrace(**kwargs)


def test_writes_sample_pack_and_output_metadata_without_tensor_values(tmp_path: Path) -> None:
    output_dir = tmp_path / "trace"
    callback = LoadBalanceTrace(
        enabled=True,
        output_dir=str(output_dir),
        flush_every_n_batches=10,
        record_cuda_timing=False,
    )
    callback.config = SimpleNamespace(job=SimpleNamespace(name="vision_sft_edge"))
    model = _model()

    callback.on_train_start(model, iteration=0)
    _run_batch(callback, model, iteration=0)
    callback.on_train_end(model, iteration=1)

    records = _read_records(output_dir / "rank_00000.jsonl")
    assert [record["record_type"] for record in records] == ["trace_start", "microbatch", "trace_end"]
    microbatch = records[1]
    assert microbatch["active_sample_ids"] == ["video-a", "video-b"]
    assert microbatch["is_active_payload_owner"] is True
    assert microbatch["data"]["text_token_counts"] == [7, 3]
    assert microbatch["data"]["video_shapes"] == [[[3, 5, 8, 8]], [[3, 9, 8, 8]]]
    assert [plan["vision_conditioning"] for plan in microbatch["data"]["sequence_plans"]] == ["t2v", "v2v"]
    assert microbatch["output"]["split_lens"] == [4, 8, 6, 12]
    assert microbatch["output"]["task_type"] == "generation"
    assert microbatch["output"]["fixed_profile"] == "generation"
    assert microbatch["output"]["physical_und_token_length"] == 16
    assert microbatch["output"]["physical_gen_token_length"] == 24
    assert microbatch["output"]["vae_pixel_shapes"] == [[5, 8, 8], [9, 8, 8]]
    assert microbatch["forward_host_ms"] >= 0
    assert microbatch["backward_host_ms"] >= 0
    assert "batch_cuda_ms" not in microbatch


def test_cp_window_marks_only_payload_owner_active(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(trace_module, "_rank_and_world_size", lambda: (1, 8))
    output_dir = tmp_path / "trace"
    callback = LoadBalanceTrace(
        enabled=True,
        output_dir=str(output_dir),
        flush_every_n_batches=1,
        record_cuda_timing=False,
    )
    model = _model(cp_rank=1, cp_slot=0, cp_size=2)

    callback.on_train_start(model, iteration=4)
    _run_batch(callback, model, iteration=4)
    model._cp_window_slot = 1
    _run_batch(callback, model, iteration=4)
    callback.on_train_end(model, iteration=5)

    records = _read_records(output_dir / "rank_00001.jsonl")
    microbatches = [record for record in records if record["record_type"] == "microbatch"]
    assert [record["cp_window_slot"] for record in microbatches] == [0, 1]
    assert [record["cp_window_index"] for record in microbatches] == [0, 0]
    assert [record["microbatch_in_iteration"] for record in microbatches] == [0, 1]
    assert [record["encodes_local_batch"] for record in microbatches] == [True, False]
    assert microbatches[0]["is_active_payload_owner"] is False
    assert microbatches[0]["active_sample_ids"] == []
    assert microbatches[0]["data"]["local_sample_ids"] == ["video-a", "video-b"]
    assert microbatches[1]["is_active_payload_owner"] is True
    assert microbatches[1]["active_sample_ids"] == ["video-a", "video-b"]


def test_trace_links_loader_and_media_timing_to_sample_ids(tmp_path):
    callback = LoadBalanceTrace(
        enabled=True, output_dir=str(tmp_path), flush_every_n_batches=1, record_cuda_timing=False
    )
    batch = _batch()
    batch["_cosmos_loader"] = {"timing": {"queue_seconds": 7.0, "prepare_seconds": 2.0}}
    batch["decode_metadata"] = [
        {
            "timing_seconds": {"numeric_read": 0.1, "video": 1.8, "sample": 2.0},
            "views": [{"timing_seconds": {"decode_with_io": 1.5, "resize": 0.2}, "unused": "omit"}],
        },
        {"views": [{"timing_seconds": {"decode_with_io": 0.3}}]},
    ]
    model = _model()
    callback.on_train_start(model, iteration=0)
    _run_batch(callback, model, iteration=0, batch=batch)
    callback.on_train_end(model, iteration=1)
    microbatch = next(r for r in _read_records(tmp_path / "rank_00000.jsonl") if r["record_type"] == "microbatch")
    assert microbatch["active_sample_ids"] == ["video-a", "video-b"]
    data = microbatch["data"]
    assert data["loader_timing"]["queue_seconds"] == 7.0
    assert data["media_timings"][0]["seconds"]["video"] == 1.8
    assert data["media_timings"][0]["views"][0]["seconds"]["decode_with_io"] == 1.5
    assert data["media_timings"][1]["sample_index"] == 1
    assert "unused" not in json.dumps(data["media_timings"])


def test_cuda_events_are_only_synchronized_at_train_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[SimpleNamespace] = []

    def make_event(*, enable_timing: bool) -> SimpleNamespace:
        assert enable_timing is True
        event = SimpleNamespace(
            recorded=False,
            synchronized=False,
            record=lambda: setattr(event, "recorded", True),
            query=lambda: False,
            synchronize=lambda: setattr(event, "synchronized", True),
            elapsed_time=lambda other: 1.25,
        )
        events.append(event)
        return event

    monkeypatch.setattr(trace_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trace_module.torch.cuda, "Event", make_event)
    output_dir = tmp_path / "trace"
    callback = LoadBalanceTrace(
        enabled=True,
        output_dir=str(output_dir),
        flush_every_n_batches=1,
        record_cuda_timing=True,
    )
    model = _model()

    callback.on_train_start(model, iteration=0)
    _run_batch(callback, model, iteration=0)

    assert len(events) == 6
    assert all(event.recorded for event in events)
    assert not any(event.synchronized for event in events)
    assert [record["record_type"] for record in _read_records(output_dir / "rank_00000.jsonl")] == ["trace_start"]

    callback.on_train_end(model, iteration=1)

    assert events[-1].synchronized is True
    records = _read_records(output_dir / "rank_00000.jsonl")
    microbatch = records[1]
    assert microbatch["batch_cuda_ms"] == 1.25
    assert microbatch["forward_cuda_ms"] == 1.25
    assert microbatch["backward_cuda_ms"] == 1.25


def test_torchrun_none_run_id_falls_back_to_job_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")
    callback = LoadBalanceTrace(
        enabled=True,
        output_dir=str(tmp_path / "trace"),
        record_cuda_timing=False,
    )
    callback.config = SimpleNamespace(job=SimpleNamespace(name="trace-smoke"))

    callback.on_train_start(_model(), iteration=0)
    callback.on_train_end(_model(), iteration=0)

    records = _read_records(tmp_path / "trace" / "rank_00000.jsonl")
    assert {record["run_id"] for record in records} == {"trace-smoke"}
