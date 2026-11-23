# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit mapping from model TP parameters to DP-owned FP32 Adam state.

MCore 0.20 internals are isolated here and pinned by runtime.py. In particular,
master weights are read from the optimizer, never reconstructed from BF16.
"""

import math
from dataclasses import dataclass

import torch


def normalize_adam_step(value):
    """Accept scalar Tensor/Python steps without truncating invalid values."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Invalid Adam step")
        value = value.item()
    if type(value) not in (int, float):
        raise ValueError("Invalid Adam step")
    try:
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Invalid Adam step")
    except OverflowError as error:
        raise ValueError("Invalid Adam step") from error
    return int(value)


@dataclass
class AdamParameterState:
    name: str
    model_parameter: torch.nn.Parameter
    master: torch.Tensor
    state: dict
    start: int
    end: int
    group: dict
    group_step: bool

    @property
    def step(self):
        return int(self.group["step"] if self.group_step else self.state["step"].item())

    def select(self, local_tp_value):
        return local_tp_value.reshape(-1)[self.start : self.end].view_as(self.master)


def adam_parameter_states(stage, optimizer):
    """Yield each optimizer-owned fragment exactly once, without collectives."""
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    by_parameter = {p: name for name, p in stage.named_parameters() if p.requires_grad}
    for opt in optimizers:
        group_step = not isinstance(opt.optimizer, (torch.optim.Adam, torch.optim.AdamW))
        if hasattr(opt, "model_param_group_index_map"):
            for param, (group, order) in opt.model_param_group_index_map.items():
                if param not in by_parameter:
                    raise ValueError("Optimizer contains a parameter outside this stage")
                main = opt.optimizer.param_groups[group]["params"][order]
                interval = opt._get_model_param_range_map(param)["param"]
                yield AdamParameterState(
                    by_parameter[param],
                    param,
                    main,
                    opt.optimizer.state[main],
                    interval.start,
                    interval.end,
                    opt.optimizer.param_groups[group],
                    group_step,
                )
        elif not opt.config.bf16 and not opt.config.fp16:
            for group in opt.optimizer.param_groups:
                for main in group["params"]:
                    yield AdamParameterState(
                        by_parameter[main], main, main, opt.optimizer.state[main], 0, main.numel(), group, group_step
                    )
        else:
            for group_index, (model_group, master_group) in enumerate(
                zip(opt.float16_groups, opt.fp32_from_float16_groups, strict=True)
            ):
                for param, main in zip(model_group, master_group, strict=True):
                    yield AdamParameterState(
                        by_parameter[param],
                        param,
                        main,
                        opt.optimizer.state[main],
                        0,
                        main.numel(),
                        opt.optimizer.param_groups[group_index],
                        group_step,
                    )
            for group_index, parameters in enumerate(opt.fp32_from_fp32_groups):
                for main in parameters:
                    yield AdamParameterState(
                        by_parameter[main],
                        main,
                        main,
                        opt.optimizer.state[main],
                        0,
                        main.numel(),
                        opt.optimizer.param_groups[group_index],
                        group_step,
                    )


def initialize_adam_state(entry, *, step, exp_avg, exp_avg_sq, master):
    if master.dtype != torch.float32 or exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
        raise ValueError("Master weights and Adam moments must be true FP32 state")
    if step < 0 or int(step) != step:
        raise ValueError("Invalid Adam step")
    with torch.no_grad():
        entry.master.copy_(entry.select(master))
    entry.state["exp_avg"] = entry.select(exp_avg).clone()
    entry.state["exp_avg_sq"] = entry.select(exp_avg_sq).clone()
    if entry.group_step:
        # TE/Apex FusedAdam keeps the bias-correction step in the parameter
        # group. Setting a per-parameter PyTorch step silently restarts it.
        entry.group["step"] = int(step)
    else:
        entry.state["step"] = torch.tensor(float(step))
