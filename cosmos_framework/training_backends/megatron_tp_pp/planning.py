"""Bounded streaming inputs, optional loss-denominator preflight and replay identity.

Rank-level loss streams raw media once. Sample-level averaging or explicit
replay verification first retains counts/hashes, then replays the same native
cursors/RNG. VAE/noising always runs once, during actual training.
"""

import hashlib
import os
import tempfile
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils import log
from cosmos_framework.utils.nvtx import nvtx_range

from .batch import reference_loss_weight
from .logical_loader import LogicalDataLoader

_TIMING = {
    "_sample_time",
    "_aug_time",
    "_pre_aug_time",
    "_aug_step_times",
    "_worker_batch_time",
    "_worker_aug_time",
    "_worker_io_time",
    "_worker_aug_step_times",
    "_worker_id",
    "timing",
}


def batch_fingerprint(batch, *, allow_cuda=False):
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" and not allow_cuda:
                raise ValueError("Preflight hashes raw CPU data, before VAE or noising")
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().data)
        elif is_dataclass(value):
            digest.update(type(value).__name__.encode())
            for field in fields(value):
                digest.update(field.name.encode())
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                if key in _TIMING:
                    continue
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str((type(value).__name__, len(value))).encode())
            for child in value:
                visit(child)
        elif isinstance(value, (str, int, float, bool, bytes)) or value is None:
            digest.update(repr(value).encode())
            digest.update(b"\0")
        else:
            raise TypeError(f"No deterministic raw batch fingerprint registered for {type(value).__name__}")

    visit(batch)
    return digest.hexdigest()


def _raw_record(loader, raw, slot, preparation, *, hash_raw=True):
    if raw["logical_data_rank"] != slot.logical_rank:
        raise ValueError("Logical loader order disagrees with the update slot schedule")
    packed = raw["packed_sequence"]
    count = len(packed.sample_lens)
    if count < 1:
        raise ValueError("A logical packed batch must contain at least one sample")
    return {
        "slot": (slot.update, slot.micro_step, slot.logical_rank),
        "source": raw["dataset_name"],
        "samples": count,
        "image": preparation.is_image_batch(raw),
        "sha256": batch_fingerprint(loader.fingerprint_batch(raw)) if hash_raw else None,
        "tokens": {
            "text": packed.text_indexes.numel(),
            "vision": packed.vision.sequence_indexes.numel() if packed.vision is not None else 0,
            "action": packed.action.sequence_indexes.numel() if packed.action is not None else 0,
        },
    }


def completed_sample_totals(records, topology, *, owner, model_group, source_rank, dp_group):
    """Collect reporting counts after a rank-level loss update, without rereading media."""
    counts = [[0, 0] for _ in range(topology.reference_micro_steps_per_update)]
    if owner:
        for record in records:
            counts[record["slot"][1]][0 if record["image"] else 1] += record["samples"]
    totals = torch.tensor(counts, dtype=torch.float64, device="cuda")
    if owner:
        dist.all_reduce(totals, group=dp_group)
    dist.broadcast(totals, src=source_rank, group=model_group)
    return totals.cpu().tolist()


def preflight_update(loader, slots, topology, preparation, *, owner, model_group, source_rank, dp_group):
    """All model-parallel ranks participate in bounded per-batch progress P2P."""
    snapshot = loader.logical_state_dict() if owner else None
    records = []
    totals = torch.zeros(topology.reference_micro_steps_per_update, 2, dtype=torch.float64, device="cuda")
    try:
        for slot in slots:
            progress = torch.empty(2, dtype=torch.int64, device="cuda")
            if owner:
                load_started = time.monotonic()
                with nvtx_range("tp_pp.preflight_load"):
                    batch = next(loader)
                load_seconds = time.monotonic() - load_started
                hash_started = time.monotonic()
                with nvtx_range("tp_pp.preflight_hash"):
                    record = _raw_record(loader, batch, slot, preparation)
                hash_seconds = time.monotonic() - hash_started
                record["host_seconds"] = {"preflight_load": load_seconds, "preflight_hash": hash_seconds}
                records.append(record)
                count, image = record["samples"], record["image"]
                totals[slot.micro_step, 0 if image else 1] += count
                progress.copy_(torch.tensor([count, image], dtype=torch.int64, device="cuda"))
                loader.commit_batch(batch)
                del batch
            # Keep every stage/TP peer making progress while the owner decodes
            # long raw windows; no peer waits for an entire 576-slot preflight.
            dist.broadcast(progress, src=source_rank, group=model_group)
            if owner and (len(records) == 1 or len(records) % 32 == 0 or len(records) == len(slots)):
                log.info(f"TP/PP denominator preflight: {len(records)}/{len(slots)} slots", rank0_only=False)
        if owner:
            dist.all_reduce(totals, group=dp_group)
        dist.broadcast(totals, src=source_rank, group=model_group)
    finally:
        if owner:
            loader.close()
    if owner:
        replacement = LogicalDataLoader(
            loader.template,
            snapshot,
            num_workers=loader.num_workers,
            prefetch_factor=loader.prefetch_factor,
            max_live_streams=loader._runner_factory.keywords["max_live_streams"],
        )
    else:
        replacement = None
    return replacement, records, totals.cpu().tolist()


def prepare_slot(
    loader, slot, record, totals, topology, preparation, rng, *, trace_prepared=False, capture_dir=None, hash_raw=True
):
    if totals is None and preparation.config.rectified_flow_training_config.sample_level_loss_averaging:
        raise ValueError("Sample-level averaging requires preflight denominators")
    started = time.monotonic()
    with nvtx_range("tp_pp.replay_load"):
        raw = next(loader)
    loaded = time.monotonic()
    with nvtx_range("tp_pp.replay_hash"):
        if record:
            matches = (
                raw["logical_data_rank"] == slot.logical_rank
                and batch_fingerprint(loader.fingerprint_batch(raw)) == record["sha256"]
            )
        else:
            # Rank-level averaging needs no future sample denominator. Record
            # the actual input as it is consumed; independent native replay
            # can verify this same manifest without a second decode here.
            record.update(
                _raw_record(
                    loader, raw, slot, preparation, hash_raw=hash_raw or trace_prepared or capture_dir is not None
                )
            )
            matches = True
    if not matches:
        raise ValueError("Native replay changed samples, packing or augmentation after denominator preflight")
    hashed = time.monotonic()
    with nvtx_range("tp_pp.native_prepare"), rng.activate(slot.logical_rank):
        batch = preparation.prepare(raw, slot.update)
    prepared = time.monotonic()
    record.setdefault("host_seconds", {}).update(
        replay_load=loaded - started,
        replay_hash=hashed - loaded,
        prepare_enqueue=prepared - hashed,
    )
    if trace_prepared:
        # Acceptance-only D2H hashing: covers actual epsilon, noised tokens,
        # targets and sampled timesteps, without retaining prepared batches.
        record["prepared_noise_sha256"] = batch_fingerprint(batch["loss_inputs"], allow_cuda=True)
        record["prepared_packed_sha256"] = batch_fingerprint(batch["packed_sequence"], allow_cuda=True)
    record["host_seconds"]["acceptance_trace"] = time.monotonic() - prepared
    if batch["sample_count"] != record["samples"] or batch["loss_inputs"]["is_image_batch"] != record["image"]:
        raise ValueError("Prepared sample/modality counts differ from the native static pack")
    if preparation.config.rectified_flow_training_config.sample_level_loss_averaging:
        batch["weight"] = reference_loss_weight(
            topology,
            local_samples=record["samples"],
            reference_modality_samples=totals[slot.micro_step][0 if record["image"] else 1],
        )
    batch["slot_id"] = (slot.update, slot.micro_step, slot.logical_rank)
    if capture_dir is not None:
        # Acceptance-only immutable spool for an independent native/FSDP replay.
        # One batch at a time, outside the compiled model and the logical RNG.
        capture_started = time.monotonic()
        directory = Path(capture_dir) / f"iter_{slot.update:09d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"micro{slot.micro_step:03d}.rank{slot.logical_rank:03d}.pt"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".", suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(batch, handle)
            os.link(temporary, target)  # Atomic publication; refuses an existing replay input.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        record["host_seconds"]["acceptance_capture"] = time.monotonic() - capture_started
    loader.commit_batch(raw)
    return batch
