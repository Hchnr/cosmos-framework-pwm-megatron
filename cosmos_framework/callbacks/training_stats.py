# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.data.generator.action.utils.domain_utils import EMBODIMENT_TO_DOMAIN_ID
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback

# Build inverse mapping: domain_id -> embodiment_type. First occurrence wins when multiple embodiment names share the
# same domain id.
DOMAIN_ID_TO_EMBODIMENT: dict[int, str] = {}
for _k, _v in EMBODIMENT_TO_DOMAIN_ID.items():
    DOMAIN_ID_TO_EMBODIMENT.setdefault(_v, _k)


class _SampleLossRecord:
    """Accumulate detached sample sums/counts without a per-step device sync."""

    def __init__(self):
        self.total = 0.0
        self.count = 0.0

    def add(self, mean, sample_count):
        value = mean.detach().float().reshape(())
        finite = torch.isfinite(value)
        self.total = self.total + torch.where(finite, value, 0.0) * sample_count
        self.count = self.count + finite.float() * sample_count

    def pair(self):
        return torch.stack((self.total, self.count)).cpu().tolist()


class TrainingStatsCallback(Callback):
    """Callback for tracking and logging training mode and embodiment statistics to wandb."""

    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self._mode_counts: dict[str, int] = {}
        self._mode_total_count: int = 0
        self._embodiment_counts: dict[str, int] = {}
        self._embodiment_total_count: int = 0
        self._per_embodiment_loss: dict[str, _SampleLossRecord] = {}
        self._per_embodiment_sub_loss: dict[str, dict[str, _SampleLossRecord]] = {}
        self._action128_registry = None
        self._skipped_mixed_samples = 0

    def on_train_start(self, model, iteration=0) -> None:
        spec = getattr(model.config, "action128_spec", None)
        self._action128_registry = spec["registry"] if spec else None

    def _action128_embodiments(self, data_batch) -> list[str] | None:
        registry = self._action128_registry
        if registry is None or "normalization_domain_id" not in data_batch:
            return None

        def ids(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().flatten().tolist()
            if isinstance(value, (list, tuple)):
                return [item for child in value for item in ids(child)]
            return [int(value)]

        # Projection head 0 is shared by every EEF embodiment. Only the source
        # normalization registry can identify its originating robot correctly.
        return [
            registry["embodiments"][registry["domain_embodiments"][index]]
            for index in ids(data_batch["normalization_domain_id"])
        ]

    def _accumulate_mode_counts(self, data_batch: dict[str, torch.Tensor]) -> None:
        modes = data_batch.get("mode", None)
        if modes is None:
            return

        if isinstance(modes, str):
            modes_list = [modes]
        elif isinstance(modes, (list, tuple)):
            modes_list = [str(m) for m in modes]
        elif isinstance(modes, torch.Tensor):
            # Defensive: support cases where mode might be encoded numerically.
            modes_list = [str(m) for m in modes.detach().cpu().tolist()]
        else:
            modes_list = [str(modes)]

        for mode in modes_list:
            self._mode_total_count += 1
            self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

    def _accumulate_embodiment_counts(self, data_batch: dict[str, torch.Tensor]) -> None:
        embodiments = self._action128_embodiments(data_batch)
        if embodiments is not None:
            for embodiment in embodiments:
                self._embodiment_total_count += 1
                self._embodiment_counts[embodiment] = self._embodiment_counts.get(embodiment, 0) + 1
            return
        domain_ids = data_batch.get("domain_id", None)
        if domain_ids is None:
            return

        if isinstance(domain_ids, int):
            domain_id_list = [domain_ids]
        elif isinstance(domain_ids, (list, tuple)):
            domain_id_list = [int(d) for d in domain_ids if d is not None]
        elif isinstance(domain_ids, torch.Tensor):
            # Flatten to handle any shape (scalar, 1D, or 2D with trailing dim)
            domain_id_list = [int(d) for d in domain_ids.detach().cpu().flatten().tolist()]
        else:
            domain_id_list = [int(domain_ids)]

        for domain_id in domain_id_list:
            embodiment = DOMAIN_ID_TO_EMBODIMENT.get(domain_id, f"unknown_{domain_id}")
            self._embodiment_total_count += 1
            self._embodiment_counts[embodiment] = self._embodiment_counts.get(embodiment, 0) + 1

    def _gather_global_mode_counts(self) -> tuple[int, dict[str, int]]:
        """
        Returns (global_total, global_mode_counts) aggregated across all ranks.
        """
        local: dict[str, int] = dict(self._mode_counts)
        local["__total__"] = int(self._mode_total_count)

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        global_total = 0
        global_counts: dict[str, int] = {}
        for item in gathered:
            if not item:
                continue
            global_total += int(item.get("__total__", 0))
            for k, v in item.items():
                if k == "__total__":
                    continue
                global_counts[k] = global_counts.get(k, 0) + int(v)
        return global_total, global_counts

    def _gather_global_embodiment_counts(self) -> tuple[int, dict[str, int]]:
        """
        Returns (global_total, global_embodiment_counts) aggregated across all ranks.
        """
        local: dict[str, int] = dict(self._embodiment_counts)
        local["__total__"] = int(self._embodiment_total_count)

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        global_total = 0
        global_counts: dict[str, int] = {}
        for item in gathered:
            if not item:
                continue
            global_total += int(item.get("__total__", 0))
            for k, v in item.items():
                if k == "__total__":
                    continue
                global_counts[k] = global_counts.get(k, 0) + int(v)
        return global_total, global_counts

    def _build_mode_log_dict(
        self, *, log_prefix: str, global_total: int, global_counts: dict[str, int]
    ) -> dict[str, float]:
        info: dict[str, float] = {}

        denom = float(global_total) if global_total > 0 else 0.0
        for mode in sorted(global_counts.keys()):
            count = float(global_counts.get(mode, 0))
            pct = (100.0 * count / denom) if denom > 0 else 0.0
            info[f"{log_prefix}_stats_mode/{mode}"] = pct

        return info

    def _build_embodiment_log_dict(
        self, *, log_prefix: str, global_total: int, global_counts: dict[str, int]
    ) -> dict[str, float]:
        info: dict[str, float] = {}

        denom = float(global_total) if global_total > 0 else 0.0
        for embodiment in sorted(global_counts.keys()):
            count = float(global_counts.get(embodiment, 0))
            pct = (100.0 * count / denom) if denom > 0 else 0.0
            info[f"{log_prefix}_stats_embodiment/{embodiment}"] = pct

        return info

    def _get_batch_embodiment(self, data_batch: dict[str, torch.Tensor]) -> str | None:
        """Only attribute a batch mean when its embodiment is unambiguous."""
        embodiments = self._action128_embodiments(data_batch)
        if embodiments is not None:
            # A batch-mean loss cannot be attributed to just its first robot.
            return embodiments[0] if embodiments and len(set(embodiments)) == 1 else None
        domain_ids = data_batch.get("domain_id", None)
        if domain_ids is None:
            return None
        if isinstance(domain_ids, torch.Tensor):
            if domain_ids.numel() == 0:
                return None
            domain_id = int(domain_ids.flatten()[0].item())
        elif isinstance(domain_ids, (list, tuple)):
            first = next((d for d in domain_ids if d is not None), None)
            if first is None:
                return None
            domain_id = int(first)
        else:
            domain_id = int(domain_ids)
        return DOMAIN_ID_TO_EMBODIMENT.get(domain_id, f"unknown_{domain_id}")

    def _accumulate_per_embodiment_loss(
        self,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        embodiments = self._action128_embodiments(data_batch)
        sample_count = len(embodiments) if embodiments is not None else len(data_batch.get("sequence_plan", [None]))
        embodiment = self._get_batch_embodiment(data_batch)
        if embodiment is None:
            if embodiments and len(set(embodiments)) > 1:
                self._skipped_mixed_samples += sample_count
            return
        self._per_embodiment_loss.setdefault(embodiment, _SampleLossRecord()).add(loss, sample_count)
        details = self._per_embodiment_sub_loss.setdefault(embodiment, {})
        for key, value in output_batch.items():
            if "loss" in key and "per_instance" not in key and isinstance(value, torch.Tensor) and value.numel() == 1:
                details.setdefault(key, _SampleLossRecord()).add(value, sample_count)

    def _compute_per_embodiment_loss_stats(self, log_prefix: str) -> dict[str, float]:
        """Merge sums/counts, never equally average ranks with unequal coverage."""
        local = {f"{log_prefix}_stats_loss/{emb}": record.pair() for emb, record in self._per_embodiment_loss.items()}
        for emb, details in self._per_embodiment_sub_loss.items():
            local.update(
                {f"{log_prefix}_stats_loss_detail/{emb}_{key}": record.pair() for key, record in details.items()}
            )
        payload = (local, self._skipped_mixed_samples)
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        sums = {}
        skipped = 0
        for records, missing in gathered:
            skipped += missing
            for key, (total, count) in records.items():
                previous = sums.get(key, (0.0, 0.0))
                sums[key] = (previous[0] + total, previous[1] + count)
        result = {key: total / count for key, (total, count) in sums.items() if count > 0}
        result[f"{log_prefix}_stats_loss_coverage/skipped_mixed_samples"] = skipped
        self._per_embodiment_loss = {}
        self._per_embodiment_sub_loss = {}
        self._skipped_mixed_samples = 0
        return result

    @torch.no_grad()
    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        # Includes every microbatch under gradient accumulation, not just the last.
        self._accumulate_mode_counts(data_batch)
        self._accumulate_embodiment_counts(data_batch)
        self._accumulate_per_embodiment_loss(data_batch, output_batch, loss)

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if iteration % self.log_freq != 0:
            return

        # All ranks must participate in collective operations below.
        mode_total, mode_counts = self._gather_global_mode_counts()
        embodiment_total, embodiment_counts = self._gather_global_embodiment_counts()
        per_embodiment_loss_dict = self._compute_per_embodiment_loss_stats(log_prefix="train")

        if not distributed.is_rank0():
            return

        if wandb.run is None:
            return

        log_dict: dict[str, float] = {}
        log_dict.update(
            self._build_mode_log_dict(log_prefix="train", global_total=mode_total, global_counts=mode_counts)
        )
        log_dict.update(
            self._build_embodiment_log_dict(
                log_prefix="train", global_total=embodiment_total, global_counts=embodiment_counts
            )
        )
        log_dict.update(per_embodiment_loss_dict)

        wandb.log({k: float(v) for k, v in log_dict.items()}, step=iteration)
