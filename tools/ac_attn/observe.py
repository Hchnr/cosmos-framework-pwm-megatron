# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""AC experiment observer with explicit checkpoint-start validation."""

import os

from tools.unified_attn.observe import Observer as BaseObserver


class Observer(BaseObserver):
    def audit_parameter(self, name):
        return True

    def on_train_start(self, model, iteration=0):
        expected = int(os.environ["AC_INITIAL_STEP"])
        if iteration != expected or expected != 25000:
            raise ValueError(f"Observer RNG schedule requires start 25000; restored {iteration}, requested {expected}")
        super().on_train_start(model, iteration)
        self.emit(
            {
                "event": "ac_config",
                "mode": model.config.activation_checkpointing.mode,
                "save_ops_regex": list(model.config.activation_checkpointing.save_ops_regex),
                "backend": model.config.joint_attn_backend,
                "trainable_parameter_tensors": sum(p.requires_grad for p in model.parameters()),
            }
        )
