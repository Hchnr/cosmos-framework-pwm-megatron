# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in real Action128 training smoke: assert Action loss and trace global prefetch.

Use only with SFT_TRAINING_MODULE=tools.action128_prefetch_smoke and an isolated
OUTPUT_ROOT. This probe skips checkpoint *writes* to avoid creating large smoke
artifacts. Model/optimizer construction, checkpoint load, forward, backward and
optimizer steps use the real training path. No online logging is enabled here.
"""

import math
import runpy

import torch

from tools import action128_spike_probe as probe

_original_getattr = probe.CallBackGroup.__getattr__


def _traced_getattr(self, name):
    original = _original_getattr(self, name)

    def call(*args, **kwargs):
        result = original(*args, **kwargs)
        if name == "on_training_step_batch_end":
            model, batch, output, loss = args[:4]
            assert model.config.action_gen and model.config.max_action_dim == 128
            assert output["task_type"] == "action"
            action_loss = float(output["flow_matching_loss_action"].detach().item())
            assert math.isfinite(action_loss) and action_loss > 0
            total_loss = float(loss.detach().item())
            assert math.isfinite(total_loss)
            loader = next(
                callback._outer
                for callback in self._callbacks
                if getattr(callback, "checkpoint_component", None) == "dataloader" and hasattr(callback, "_outer")
            )
            probe.emit(
                "action128_contract",
                dataset=batch["dataset_name"],
                action_loss=action_loss,
                vision_loss=float(output["flow_matching_loss_vision"].detach().item()),
                total_loss=total_loss,
                action_tokens=int(output["action_token_length"]),
                vision_tokens=int(output["vision_token_length"]),
                samples=int(output["batch_size"]),
                committed_global_id=loader.committed_global_id,
                prefetch_stats=loader.prefetch_stats,
            )
        return result

    return call


probe.CallBackGroup.__getattr__ = _traced_getattr


def _skip_smoke_checkpoint(self, *args, **kwargs):
    probe.emit("smoke_checkpoint_write_skipped")


probe.DistributedCheckpointer.save = _skip_smoke_checkpoint

if __name__ == "__main__":
    torch.set_num_threads(1)
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
