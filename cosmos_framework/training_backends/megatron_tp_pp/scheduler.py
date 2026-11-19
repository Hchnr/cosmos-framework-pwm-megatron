# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Restore the native LambdaLR position onto canonical MCore LR groups."""

import copy

import torch

from cosmos_framework.utils.generator.optimizer import _lr_scheduler_cls


class NativeScheduler:
    def __init__(self, optimizer, scheduler_type, kwargs):
        self.optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        schedule = _lr_scheduler_cls(scheduler_type, **kwargs)
        self.schedulers = []
        for opt in self.optimizers:
            for group in opt.optimizer.param_groups:
                group.setdefault("initial_lr", group["max_lr"])
            self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt.optimizer, schedule.schedule))

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return {"version": 1, "schedulers": [scheduler.state_dict() for scheduler in self.schedulers]}

    def load_state_dict(self, state):
        if state["version"] != 1 or len(state["schedulers"]) != len(self.schedulers):
            raise ValueError("Scheduler checkpoint topology mismatch")
        for scheduler, saved in zip(self.schedulers, state["schedulers"], strict=True):
            scheduler.load_state_dict(copy.deepcopy(saved))
            for group, value in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr(), strict=True):
                group["lr"] = float(value)

    def import_source(self, reader):
        """Map native group order by its base LR, retaining every scalar field."""
        if any(name.startswith("schedulers.") for name in reader.metadata):
            raise ValueError("The planned source uses a single native scheduler")
        saved = {}
        for name in reader.metadata:
            if name in ("base_lrs", "_last_lr", "lr_lambdas") or name.startswith(("base_lrs.", "_last_lr.")):
                continue
            if name.startswith("lr_lambdas."):
                raise ValueError("Unexpected stateful source LR lambda")
            saved[name] = reader.read(name)
        bases = (
            reader.read("base_lrs")
            if "base_lrs" in reader.metadata
            else [
                reader.read(name)
                for name in sorted(
                    (name for name in reader.metadata if name.startswith("base_lrs.")),
                    key=lambda name: int(name.rsplit(".", 1)[1]),
                )
            ]
        )
        current = (
            reader.read("_last_lr")
            if "_last_lr" in reader.metadata
            else [reader.read(f"_last_lr.{index}") for index in range(len(bases))]
        )
        if len({float(value) for value in bases}) != len(bases):
            raise ValueError("Ambiguous source LR identities")
        for scheduler in self.schedulers:
            indices = []
            for group in scheduler.optimizer.param_groups:
                matches = [
                    index for index, base in enumerate(bases) if abs(float(base) - float(group["initial_lr"])) <= 1e-12
                ]
                if len(matches) != 1:
                    raise ValueError("MCore scheduler group cannot be mapped to its source LR")
                indices.append(matches[0])
            state = {
                **saved,
                "base_lrs": [float(bases[index]) for index in indices],
                "_last_lr": [float(current[index]) for index in indices],
                "lr_lambdas": [None] * len(indices),
            }
            scheduler.load_state_dict(copy.deepcopy(state))
            for group, base, lr in zip(
                scheduler.optimizer.param_groups, state["base_lrs"], state["_last_lr"], strict=True
            ):
                group["initial_lr"], group["lr"] = base, lr
        return int(saved["last_epoch"])
