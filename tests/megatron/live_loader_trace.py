"""Content/cursor/RNG traces for real-catalog checkpoint continuation checks."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


def _hash(value, digest):
    if isinstance(value, torch.Tensor):
        local = value.detach().cpu().contiguous()
        digest.update(str((str(local.dtype), tuple(local.shape))).encode())
        digest.update(memoryview(local.reshape(-1).view(torch.uint8).numpy()))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        _hash({field.name: getattr(value, field.name) for field in dataclasses.fields(value)}, digest)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _hash(str(key), digest)
            _hash(value[key], digest)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _hash(child, digest)
    elif isinstance(value, bytes):
        digest.update(value)
    else:
        digest.update(repr(value).encode())
    digest.update(b"\0")


def batch_identity(batch):
    digest = hashlib.sha256()
    for key in (
        "video",
        "action",
        "text_token_ids",
        "sequence_plan",
        "vision_loss_mask",
        "action_loss_mask",
        "action_dim_mask",
        "domain_id",
        "normalization_domain_id",
    ):
        _hash(batch.get(key), digest)
    metadata = batch.get("_cosmos_loader", {})
    return {
        "dataset": batch.get("dataset_name"),
        "worker": batch["sample_worker_id"].detach().cpu().tolist(),
        "epoch": batch["sample_epoch"].detach().cpu().tolist(),
        "index": batch["sample_index"].detach().cpu().tolist(),
        "global_id": metadata.get("global_id"),
        "source": metadata.get("source"),
        "resume_rng_sha256": hashlib.sha256((metadata.get("resume_rng") or b"")).hexdigest(),
        "content_sha256": digest.hexdigest(),
    }


class LiveTrace(Callback):
    def __init__(self, output):
        super().__init__()
        self.output = Path(output)
        self.records = []

    def on_training_step_batch_start(self, model, data_batch, iteration=0):
        self.records.append({"iteration": iteration, **batch_identity(data_batch)})

    def on_train_end(self, model, iteration=0):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        path = self.output.with_name(f"{self.output.stem}_rank_{os.environ['RANK']}.json")
        path.write_text(json.dumps(self.records, indent=2) + "\n")
