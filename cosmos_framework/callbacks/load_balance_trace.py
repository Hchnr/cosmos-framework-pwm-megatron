# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-overhead per-rank trace for load-balance analysis.

The callback is disabled by default. When enabled, every rank appends one JSON
record per consumed microbatch to its own file. No distributed collective is
added to the training hot path. CUDA timings use delayed event reads so normal
microbatches do not call ``torch.cuda.synchronize()``.

With context parallelism, the raw ``data_batch`` remains rank-local while the
model rotates one owner's processed payload across the CP group. Records
therefore distinguish ``local_sample_ids`` from ``active_sample_ids``; the
latter are populated only on the payload-owner rank and can be joined offline.
"""

from __future__ import annotations

import json
import os
import socket
import time
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback

_SCHEMA_VERSION = 1
_OUTPUT_FEATURE_KEYS = (
    "task_type",
    "fixed_profile",
    "und_token_length",
    "gen_token_length",
    "physical_und_token_length",
    "physical_gen_token_length",
    "vision_token_length",
    "action_token_length",
    "sound_token_length",
    "is_image_batch",
    "batch_size",
    "split_lens",
    "attn_modes",
    "vae_pixel_shapes",
    "vae_latent_shapes",
    "vision_input_mode",
)


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _optional_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _string_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value[:limit]]


def _collect_tensor_shapes(value: Any, shapes: list[list[int]]) -> None:
    if isinstance(value, torch.Tensor):
        shapes.append([int(dim) for dim in value.shape])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensor_shapes(item, shapes)


def _per_sample_tensor_shapes(value: Any, limit: int) -> list[list[list[int]]]:
    """Return tensor shapes while preserving the outer sample dimension."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [[[]]]
        return [[[int(dim) for dim in value.shape[1:]]] for _ in range(min(int(value.shape[0]), limit))]
    if not isinstance(value, (list, tuple)):
        return []

    result: list[list[list[int]]] = []
    for sample in value[:limit]:
        sample_shapes: list[list[int]] = []
        _collect_tensor_shapes(sample, sample_shapes)
        result.append(sample_shapes)
    return result


def _tensor_numel(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_numel(item) for item in value)
    return 0


def _per_sample_token_counts(value: Any, limit: int) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [1]
        batch_size = min(int(value.shape[0]), limit)
        per_sample = int(value[0].numel()) if batch_size else 0
        return [per_sample] * batch_size
    if not isinstance(value, (list, tuple)):
        return []
    return [_tensor_numel(sample) for sample in value[:limit]]


def _sequence_plan_summaries(value: Any, limit: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    plans = value if isinstance(value, (list, tuple)) else [value]
    summaries: list[dict[str, Any]] = []
    for plan in plans[:limit]:
        vision_indexes = getattr(plan, "condition_frame_indexes_vision", []) or []
        action_indexes = getattr(plan, "condition_frame_indexes_action", []) or []
        sound_indexes = getattr(plan, "condition_frame_indexes_sound", []) or []
        vision_condition_frames = len(vision_indexes)
        if not getattr(plan, "has_vision", False):
            vision_conditioning = "none"
        elif vision_condition_frames == 0:
            vision_conditioning = "t2v"
        elif vision_condition_frames == 1:
            vision_conditioning = "i2v"
        else:
            vision_conditioning = "v2v"
        summaries.append(
            {
                "has_text": bool(getattr(plan, "has_text", False)),
                "has_vision": bool(getattr(plan, "has_vision", False)),
                "has_action": bool(getattr(plan, "has_action", False)),
                "has_sound": bool(getattr(plan, "has_sound", False)),
                "vision_conditioning": vision_conditioning,
                "vision_condition_frames": vision_condition_frames,
                "action_condition_steps": len(action_indexes),
                "sound_condition_steps": len(sound_indexes),
            }
        )
    return summaries


def _json_metadata(value: Any, *, limit: int = 256) -> Any:
    """Serialize metadata without reading tensor values or synchronizing CUDA."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": [int(dim) for dim in value.shape],
            "tensor_dtype": str(value.dtype),
            "tensor_device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item, limit=limit) for item in value[:limit]]
    if isinstance(value, dict):
        return {str(key): _json_metadata(item, limit=limit) for key, item in list(value.items())[:limit]}
    return str(value)


class LoadBalanceTrace(Callback):
    """Write per-rank JSONL records for offline load-balance attribution.

    Args:
        enabled: Master switch. Disabled callbacks do no filesystem or CUDA work.
        output_dir: Local directory. Each rank writes ``rank_XXXXX.jsonl``.
        flush_every_n_batches: Number of completed records buffered before append.
        record_cuda_timing: Record delayed CUDA-event batch/forward/backward times.
        max_samples_per_record: Cap list-valued sample metadata in one record.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "",
        flush_every_n_batches: int = 100,
        record_cuda_timing: bool = True,
        max_samples_per_record: int = 64,
    ) -> None:
        super().__init__()
        if flush_every_n_batches < 1:
            raise ValueError(f"flush_every_n_batches must be >= 1, got {flush_every_n_batches}.")
        if max_samples_per_record < 1:
            raise ValueError(f"max_samples_per_record must be >= 1, got {max_samples_per_record}.")
        if enabled and not output_dir:
            raise ValueError("output_dir is required when LoadBalanceTrace is enabled.")

        self.enabled = enabled
        self.output_dir = output_dir
        self.flush_every_n_batches = flush_every_n_batches
        self.record_cuda_timing = record_cuda_timing
        self.max_samples_per_record = max_samples_per_record

        self._started = False
        self._rank = 0
        self._world_size = 1
        self._output_path: Path | None = None
        self._batch_index = 0
        self._last_iteration: int | None = None
        self._microbatch_in_iteration = -1
        self._cp_window_index = -1
        self._active: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self._pending_cuda: list[tuple[dict[str, Any], dict[str, torch.cuda.Event]]] = []
        self._dataloader_start_ns: int | None = None
        self._last_dataloader_host_ms: float | None = None

    def _run_id(self) -> str:
        job = getattr(getattr(self, "config", None), "job", None)
        job_name = str(getattr(job, "name", "local"))
        for env_name in ("TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "WANDB_RUN_ID"):
            value = os.environ.get(env_name)
            if value and value.lower() not in {"none", "null"}:
                return value
        return job_name

    @staticmethod
    def _topology(model: ImaginaireModel, rank: int, world_size: int) -> dict[str, Any]:
        parallel_dims = getattr(model, "parallel_dims", None)
        cp_size = int(getattr(parallel_dims, "cp_size", 1)) if parallel_dims is not None else 1
        cp_rank = int(getattr(parallel_dims, "cp_rank", 0)) if parallel_dims is not None else 0
        cp_group_ranks = [rank]
        if parallel_dims is not None and bool(getattr(parallel_dims, "cp_enabled", False)):
            cp_mesh = getattr(parallel_dims, "cp_mesh", None)
            if cp_mesh is not None and dist.is_available() and dist.is_initialized():
                cp_group = cp_mesh.get_group()
                cp_group_ranks = [int(group_rank) for group_rank in dist.get_process_group_ranks(cp_group)]

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": _optional_env_int("LOCAL_RANK"),
            "node_rank": _optional_env_int("GROUP_RANK"),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "dp_shard": int(getattr(parallel_dims, "dp_shard", 1)) if parallel_dims is not None else 1,
            "dp_replicate": int(getattr(parallel_dims, "dp_replicate", 1)) if parallel_dims is not None else 1,
            "cp_size": cp_size,
            "cp_rank": cp_rank,
            "cp_group_ranks": cp_group_ranks,
            "cp_group_id": min(cp_group_ranks),
        }

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        if self._output_path is None:
            raise RuntimeError("LoadBalanceTrace output path is not initialized.")
        with self._output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")

    def _ensure_started(self, model: ImaginaireModel, iteration: int) -> None:
        if self._started or not self.enabled:
            return
        self._rank, self._world_size = _rank_and_world_size()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = output_dir / f"rank_{self._rank:05d}.jsonl"
        topology = self._topology(model, self._rank, self._world_size)
        self._write_records(
            [
                {
                    "record_type": "trace_start",
                    "schema_version": _SCHEMA_VERSION,
                    "run_id": self._run_id(),
                    "iteration": int(iteration),
                    "started_at_unix_ns": time.time_ns(),
                    **topology,
                }
            ]
        )
        self._started = True

    def _new_cuda_event(self) -> torch.cuda.Event | None:
        if not (self.record_cuda_timing and torch.cuda.is_available()):
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _append_record(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        if len(self._records) >= self.flush_every_n_batches:
            self._write_records(self._records)
            self._records.clear()

    def _drain_cuda_records(self, *, wait: bool) -> None:
        while self._pending_cuda:
            record, events = self._pending_cuda[0]
            end_event = events["batch_end"]
            if wait:
                end_event.synchronize()
            elif not end_event.query():
                break

            for name, start_key, end_key in (
                ("batch_cuda_ms", "batch_start", "batch_end"),
                ("forward_cuda_ms", "forward_start", "forward_end"),
                ("backward_cuda_ms", "backward_start", "backward_end"),
            ):
                start_event = events.get(start_key)
                phase_end_event = events.get(end_key)
                if start_event is not None and phase_end_event is not None:
                    record[name] = float(start_event.elapsed_time(phase_end_event))
            self._pending_cuda.pop(0)
            self._append_record(record)

    def _data_features(self, data_batch: dict[str, Any]) -> dict[str, Any]:
        limit = self.max_samples_per_record
        sample_ids_raw = data_batch.get("action_sample_fingerprint")
        if sample_ids_raw is None:
            sample_ids_raw = data_batch.get("__key__")
        sample_ids = _string_list(sample_ids_raw, limit)
        features = {
            "local_sample_ids": sample_ids,
            "sample_count_from_ids": len(sample_ids),
            "dataset_names": _string_list(data_batch.get("dataset_name"), limit),
            "source_dataset_names": _string_list(data_batch.get("source_dataset_name"), limit),
            "sequence_plans": _sequence_plan_summaries(data_batch.get("sequence_plan"), limit),
            "text_token_counts": _per_sample_token_counts(data_batch.get("text_token_ids"), limit),
            "video_shapes": _per_sample_tensor_shapes(data_batch.get("video"), limit),
            "video_latent_shapes": _per_sample_tensor_shapes(data_batch.get("video_latent"), limit),
            "image_shapes": _per_sample_tensor_shapes(data_batch.get("images"), limit),
            "action_shapes": _per_sample_tensor_shapes(data_batch.get("action"), limit),
            "sound_shapes": _per_sample_tensor_shapes(data_batch.get("sound"), limit),
            "vision_input_mode": _string_list(data_batch.get("vision_input_mode"), limit),
            "vae_latent_shapes": _json_metadata(data_batch.get("vae_latent_shapes")),
            "vae_latent_io_ms": _json_metadata(data_batch.get("vae_latent_io_ms")),
            "vae_latent_nbytes": _json_metadata(data_batch.get("vae_latent_nbytes")),
            "vae_latent_cache_hits": _json_metadata(data_batch.get("vae_latent_cache_hits")),
            "vae_latent_cache_misses": _json_metadata(data_batch.get("vae_latent_cache_misses")),
            "dropped_count": _json_metadata(data_batch.get("_dropped_count")),
        }
        loader_metadata = data_batch.get("_cosmos_loader", {})
        if isinstance(loader_metadata, dict) and "timing" in loader_metadata:
            features["loader_timing"] = _json_metadata(loader_metadata["timing"])
        metadata = data_batch.get("decode_metadata", [])
        if isinstance(metadata, dict):
            metadata = [metadata]
        media_timings = []
        if isinstance(metadata, (list, tuple)):
            for index, sample in enumerate(metadata[:limit]):
                if not isinstance(sample, dict):
                    continue
                view_timings = [
                    {"view_index": view_index, "seconds": view["timing_seconds"]}
                    for view_index, view in enumerate(sample.get("views", [])[:3])
                    if isinstance(view, dict) and "timing_seconds" in view
                ]
                if "timing_seconds" in sample or view_timings:
                    media_timings.append(
                        {
                            "sample_index": index,
                            "seconds": sample.get("timing_seconds", {}),
                            "views": view_timings,
                        }
                    )
        if media_timings:
            features["media_timings"] = _json_metadata(media_timings)
        return features

    @staticmethod
    def _output_features(output_batch: dict[str, Any]) -> dict[str, Any]:
        return {key: _json_metadata(output_batch[key]) for key in _OUTPUT_FEATURE_KEYS if key in output_batch}

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self._ensure_started(model, iteration)

    def on_before_dataloading(self, iteration: int = 0) -> None:
        del iteration
        if self.enabled:
            self._dataloader_start_ns = time.perf_counter_ns()

    def on_after_dataloading(self, iteration: int = 0) -> None:
        del iteration
        if self.enabled and self._dataloader_start_ns is not None:
            self._last_dataloader_host_ms = (time.perf_counter_ns() - self._dataloader_start_ns) / 1e6
            self._dataloader_start_ns = None

    def on_training_step_batch_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if not self.enabled:
            return
        self._ensure_started(model, iteration)
        self._drain_cuda_records(wait=False)
        if self._active is not None:
            raise RuntimeError("LoadBalanceTrace received a new batch before the prior batch ended.")

        if self._last_iteration == iteration:
            self._microbatch_in_iteration += 1
        else:
            self._last_iteration = iteration
            self._microbatch_in_iteration = 0

        topology = self._topology(model, self._rank, self._world_size)
        cp_slot = int(getattr(model, "_cp_window_slot", 0)) if topology["cp_size"] > 1 else 0
        if self._cp_window_index < 0 or cp_slot == 0:
            self._cp_window_index += 1
        is_payload_owner = topology["cp_rank"] == cp_slot
        data_features = self._data_features(data)

        active = {
            "record_type": "microbatch",
            "schema_version": _SCHEMA_VERSION,
            "run_id": self._run_id(),
            "timestamp_unix_ns": time.time_ns(),
            "iteration": int(iteration),
            "batch_index": self._batch_index,
            "microbatch_in_iteration": self._microbatch_in_iteration,
            "cp_window_index": self._cp_window_index,
            "cp_window_slot": cp_slot,
            "active_payload_owner_cp_rank": cp_slot,
            "is_active_payload_owner": is_payload_owner,
            "encodes_local_batch": cp_slot == 0,
            "active_sample_ids": data_features["local_sample_ids"] if is_payload_owner else [],
            "dataloader_host_ms": self._last_dataloader_host_ms,
            "batch_host_start_ns": time.perf_counter_ns(),
            "forward_host_ms": 0.0,
            "backward_host_ms": 0.0,
            "data": data_features,
            **topology,
        }
        batch_start_event = self._new_cuda_event()
        active["cuda_events"] = {"batch_start": batch_start_event} if batch_start_event is not None else {}
        active["phase_start_ns"] = {}
        self._active = active

    def on_before_forward(self, iteration: int = 0) -> None:
        del iteration
        if self._active is None:
            return
        self._active["phase_start_ns"]["forward"] = time.perf_counter_ns()
        event = self._new_cuda_event()
        if event is not None:
            self._active["cuda_events"]["forward_start"] = event

    def on_after_forward(self, iteration: int = 0) -> None:
        del iteration
        if self._active is None:
            return
        started = self._active["phase_start_ns"].pop("forward", None)
        if started is not None:
            self._active["forward_host_ms"] += (time.perf_counter_ns() - started) / 1e6
        event = self._new_cuda_event()
        if event is not None:
            self._active["cuda_events"]["forward_end"] = event

    def on_before_backward(self, model: ImaginaireModel, loss: torch.Tensor, iteration: int = 0) -> None:
        del model, loss, iteration
        if self._active is None:
            return
        self._active["phase_start_ns"]["backward"] = time.perf_counter_ns()
        event = self._new_cuda_event()
        if event is not None:
            self._active["cuda_events"]["backward_start"] = event

    def on_after_backward(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        if self._active is None:
            return
        started = self._active["phase_start_ns"].pop("backward", None)
        if started is not None:
            self._active["backward_host_ms"] += (time.perf_counter_ns() - started) / 1e6
        event = self._new_cuda_event()
        if event is not None:
            self._active["cuda_events"]["backward_end"] = event

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, loss, iteration
        if not self.enabled:
            return
        if self._active is None:
            raise RuntimeError("LoadBalanceTrace batch ended without a matching batch start.")

        active = self._active
        active["batch_host_ms"] = (time.perf_counter_ns() - active.pop("batch_host_start_ns")) / 1e6
        active["output"] = self._output_features(output_batch)
        active.pop("phase_start_ns", None)
        events: dict[str, torch.cuda.Event] = active.pop("cuda_events")
        batch_end_event = self._new_cuda_event()
        if batch_end_event is not None:
            events["batch_end"] = batch_end_event
            self._pending_cuda.append((active, events))
        else:
            self._append_record(active)

        self._active = None
        self._batch_index += 1
        self._drain_cuda_records(wait=False)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        if not self.enabled:
            return
        self._drain_cuda_records(wait=True)
        self._append_record(
            {
                "record_type": "trace_end",
                "schema_version": _SCHEMA_VERSION,
                "run_id": self._run_id(),
                "iteration": int(iteration),
                "ended_at_unix_ns": time.time_ns(),
                "rank": self._rank,
                "world_size": self._world_size,
                "batch_count": self._batch_index,
            }
        )
        self._write_records(self._records)
        self._records.clear()
