# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint / resume callbacks for ``CosmosDataLoader``.

Two public classes:

* ``CosmosDataLoaderStateCallback`` — for a single ``CosmosDataLoader`` whose
  distributor is a ``MapDistributor``.  Saves per-worker ``(epoch, index)`` to
  the DCP checkpoint and, on resume, sets ``COSMOS_DL_STATE_*`` env vars so
  that ``MapDistributor.stream`` fast-forwards each worker to the saved
  position.

* ``JointCosmosDataLoaderStateCallback`` — for ``JointCosmosDataLoader``.
  Persists the outer ``global_id`` (dataset-selection sequence cursor) plus
  inner per-dataset per-worker state via one ``CosmosDataLoaderStateCallback``
  per inner loader.

Usage (single loader)::

    exp["trainer"]["callbacks"]["dataloader_state"] = CosmosDataLoaderStateCallback()

Usage (joint loader)::

    joint_loader = JointCosmosDataLoader(dataloaders={...}, seed=42)
    exp["dataloader_train"] = joint_loader
    exp["trainer"]["callbacks"]["dataloader_state"] = JointCosmosDataLoaderStateCallback(
        outer_loader=joint_loader,
    )

Structured configs instantiate callbacks before the training dataloader. In
that flow, register JointCosmosDataLoaderStateCallback without outer_loader.
It discovers dataset names from delivered batches; after DCP load the trainer
restores the actual outer loader's deterministic selection cursor from the
optimizer iteration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


@dataclass
class _WorkerState:
    epoch: int = 0
    index: int = 0


class CosmosDataLoaderStateCallback(Callback):
    """Checkpoint/resume for a single ``CosmosDataLoader(MapDistributor)``.

    Tracks the highest-seen ``(epoch, index)`` per worker from batch metadata
    fields ``sample_worker_id``, ``sample_epoch``, ``sample_index`` (injected
    by ``MapDistributor``).

    On ``state_dict()`` the per-worker positions are serialised into the DCP
    checkpoint (``checkpoint_component = "dataloader"``).

    On ``load_state_dict()`` the positions are written to env vars::

        COSMOS_DL_STATE_WORKER_{id}_EPOCH
        COSMOS_DL_STATE_WORKER_{id}_INDEX

    (or ``COSMOS_DL_STATE_{name}_WORKER_{id}_*`` when ``name`` is set, for
    multi-loader namespacing).  ``MapDistributor.stream`` pops these on first
    iteration and resumes from ``index + 1``.
    """

    checkpoint_component: str = "dataloader"

    def __init__(self, name: str = "", distributor_type: str | None = None) -> None:
        # distributor_type is accepted but unused — it exists only so that Hydra
        # struct-merging over the legacy DataLoaderStateCallback entry (which
        # carries distributor_type="${data_setting.distributor_type}") does not
        # raise an unexpected-keyword-argument error at instantiation time.
        super().__init__()
        self.name = name
        self.config: Any = None
        self.state: dict[int, _WorkerState] = {}

    @property
    def _env_prefix(self) -> str:
        return f"COSMOS_DL_STATE_{self.name}_" if self.name else "COSMOS_DL_STATE_"

    def _update_state_from_batch(self, data_batch: dict[str, torch.Tensor]) -> None:
        if "sample_worker_id" not in data_batch:
            return  # IterableDistributor / no position metadata
        worker_ids = data_batch["sample_worker_id"].tolist()
        epochs = data_batch["sample_epoch"].tolist()
        indices = data_batch["sample_index"].tolist()
        for worker_id, epoch, index in zip(worker_ids, epochs, indices, strict=True):
            cur = self.state.get(worker_id)
            if cur is None:
                self.state[worker_id] = _WorkerState(epoch=epoch, index=index)
            elif epoch > cur.epoch or (epoch == cur.epoch and index > cur.index):
                self.state[worker_id] = _WorkerState(epoch=epoch, index=index)

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.config and iteration % self.config.trainer.logging_iter == 0:
            msg = "\n"
            for wid, s in self.state.items():
                msg += f"worker {wid}: epoch={s.epoch}, index={s.index}\n"
            log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[int, dict[str, int]]:
        result: dict[int, dict[str, int]] = {}
        for worker_id, s in self.state.items():
            result[worker_id] = {"epoch": s.epoch, "index": s.index}
            log.info(f"Saved CosmosDataLoader state for worker {worker_id}: epoch={s.epoch}, index={s.index}")
        return result

    def load_state_dict(self, state_dict: dict[int, dict[str, int]]) -> None:
        if not state_dict:
            log.info("No CosmosDataLoader state found in checkpoint")
            return

        pfx = self._env_prefix
        self.state = {}
        for worker_id, per_worker in state_dict.items():
            epoch = per_worker["epoch"]
            index = per_worker["index"]
            self.state[worker_id] = _WorkerState(epoch=epoch, index=index)
            os.environ[f"{pfx}WORKER_{worker_id}_EPOCH"] = str(epoch)
            os.environ[f"{pfx}WORKER_{worker_id}_INDEX"] = str(index)
            log.info(f"Loaded CosmosDataLoader state for worker {worker_id}: epoch={epoch}, index={index}")


class JointCosmosDataLoaderStateCallback(Callback):
    """Checkpoint/resume for ``JointCosmosDataLoader``.

    Manages two levels of state in a single DCP checkpoint entry:

    1. **Outer** ``global_id`` — how many batches the outer loader has yielded.
       Restored via ``outer_loader.set_start_iteration(global_id)`` so the
       deterministic dataset-selection sequence resumes from the right step.
       Direct restoration applies when outer_loader is bound. In detached
       structured-config mode, Trainer.train restores the actual outer cursor
       from iteration times gradient accumulation.

    2. **Inner** per-dataset, per-worker ``(epoch, index)`` — one
       ``CosmosDataLoaderStateCallback`` per inner loader, keyed by name.

    The ``checkpoint_component = "dataloader"`` class attribute ensures the DCP
    checkpointer's ``_DataloaderWrapper`` discovers exactly this callback.  Do
    **not** also register standalone ``CosmosDataLoaderStateCallback`` instances
    for the inner loaders — this class already handles them all.
    """

    checkpoint_component: str = "dataloader"
    _RESERVED_STATE_KEYS = {"global_id", "__schedule__", "__fingerprints__", "__execution__"}

    def __init__(self, outer_loader: Any | None = None) -> None:
        super().__init__()
        self._outer = outer_loader
        self._global_id = 0
        self._inner: dict[str, CosmosDataLoaderStateCallback] = {}
        if outer_loader is not None:
            self._inner = {name: CosmosDataLoaderStateCallback(name=name) for name in outer_loader._names}
        self.config: Any = None

    def _inner_callback(self, name: str) -> CosmosDataLoaderStateCallback:
        callback = self._inner.get(name)
        if callback is None:
            callback = CosmosDataLoaderStateCallback(name=name)
            self._inner[name] = callback
        return callback

    def _update_state_from_batch(self, batch: dict) -> None:
        if self._outer is not None and hasattr(self._outer, "commit_batch"):
            self._outer.commit_batch(batch)
        name = batch.get("dataset_name")
        if isinstance(name, str):
            self._inner_callback(name)._update_state_from_batch(batch)
        if self._outer is None:
            self._global_id += 1

    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict,
        output_batch: dict,
        loss: Any,
        iteration: int = 0,
    ) -> None:
        if self._outer is not None and hasattr(self._outer, "commit_batch"):
            window = getattr(getattr(self, "trainer", None), "_cp_data_window", None)
            if window is not None and window.active:
                # Each CP slot reuses the same local batch. Commit only after
                # the final slot has released the trainer's data window.
                return
        self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict,
        output_batch: dict,
        loss: Any,
        iteration: int = 0,
    ) -> None:
        if self.config and iteration % self.config.trainer.logging_iter == 0:
            global_id = self._outer._global_id if self._outer is not None else self._global_id
            msg = f"\nJointCosmosDataLoader global_id={global_id}\n"
            for name, cb in self._inner.items():
                for wid, s in cb.state.items():
                    msg += f"  [{name}] worker {wid}: epoch={s.epoch}, index={s.index}\n"
            log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict:
        global_id = self._outer._global_id if self._outer is not None else self._global_id
        if self._outer is not None and hasattr(self._outer, "committed_global_id"):
            window = getattr(getattr(self, "trainer", None), "_cp_data_window", None)
            if window is not None and window.active:
                raise ValueError("Save shared dataloader checkpoints at a completed CP data-window boundary")
            global_id = self._outer.committed_global_id
        state = {
            "global_id": global_id,
            **{name: cb.state_dict() for name, cb in self._inner.items()},
        }
        if self._outer is not None:
            state["__schedule__"] = self._outer.schedule_identity
            state["__fingerprints__"] = self._outer.dataset_fingerprints
            if hasattr(self._outer, "execution_state_dict"):
                state["__execution__"] = self._outer.execution_state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if (
            self._outer is not None
            and state.get("__execution__") is not None
            and not hasattr(self._outer, "restore_execution_state")
        ):
            raise ValueError(
                "Shared loader checkpoint requires shared_prefetch; legacy backend cannot restore its state"
            )
        saved_schedule = state.get("__schedule__")
        if self._outer is not None and saved_schedule is not None and saved_schedule != self._outer.schedule_identity:
            raise ValueError(
                "JointCosmosDataLoader schedule changed since checkpoint: "
                f"saved={saved_schedule}, current={self._outer.schedule_identity}"
            )
        saved_fingerprints = state.get("__fingerprints__")
        if (
            self._outer is not None
            and saved_fingerprints is not None
            and saved_fingerprints != self._outer.dataset_fingerprints
        ):
            raise ValueError(
                "JointCosmosDataLoader datasets changed since checkpoint: "
                f"saved={saved_fingerprints}, current={self._outer.dataset_fingerprints}"
            )
        global_id = state.get("global_id", 0)
        self._global_id = global_id
        if self._outer is not None:
            if hasattr(self._outer, "restore_execution_state"):
                self._outer.restore_execution_state(state.get("__execution__"), global_id)
            else:
                self._outer.set_start_iteration(global_id)
        log.info(f"JointCosmosDataLoaderStateCallback: resumed outer global_id={global_id}")
        for name, inner_state in state.items():
            if name in self._RESERVED_STATE_KEYS or not isinstance(name, str):
                continue
            self._inner_callback(name).load_state_dict(inner_state)

    def validate_action128_extension(self, state: dict, old_spec: dict) -> bool:
        """Validate all existing streams without restoring any worker cursor."""
        if self._outer is None:
            raise ValueError("Action128 extend requires the bound catalog dataloader")
        saved = state.get("__fingerprints__")
        current = self._outer.dataset_fingerprints_for_action128(old_spec)
        if not saved or any(current.get(name) != fingerprint for name, fingerprint in saved.items()):
            raise ValueError(
                "Action128 extend requires every old dataset unchanged and present; append new catalogs last"
            )
        if set(current) != set(self._outer._names):
            raise ValueError("Action128 extend requires fingerprints for all data streams")
        added = set(saved) != set(current)
        if not added and state.get("__schedule__") != self._outer.schedule_identity:
            raise ValueError("Action128 extend permits a new schedule only when appending datasets")
        return added


def attach_cosmos_dataloader_state_callback(trainer: Any, dataloader: Any) -> bool:
    """Attach checkpoint state after the catalog loader has been instantiated.

    Catalog TOML conversion leaves the loader as a LazyCall until after the
    trainer and its callbacks are constructed. Binding here keeps one actual
    loader instance shared by training and its state callback.
    """

    from cosmos_framework.data.generator.dataflow.loader import CosmosDataLoader, JointCosmosDataLoader

    for existing in trainer.callbacks._callbacks:
        if getattr(existing, "checkpoint_component", None) == "dataloader" and existing.has_checkpoint_state():
            if isinstance(existing, JointCosmosDataLoaderStateCallback) and isinstance(
                dataloader, JointCosmosDataLoader
            ):
                if existing._outer is not None and existing._outer is not dataloader:
                    raise ValueError("Dataloader state callback is already bound to a different loader")
                existing._outer = dataloader
            return False

    if isinstance(dataloader, JointCosmosDataLoader):
        state_callback: Callback = JointCosmosDataLoaderStateCallback(dataloader)
    elif isinstance(dataloader, CosmosDataLoader):
        state_callback = CosmosDataLoaderStateCallback()
    else:
        return False

    state_callback.config = trainer.config
    state_callback.trainer = trainer
    trainer.callbacks._callbacks.append(state_callback)
    log.info(f"Attached {type(state_callback).__name__} to the instantiated training dataloader")
    return True
