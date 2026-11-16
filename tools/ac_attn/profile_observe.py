# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Additional CPU pack geometry only in separately timed Nsight runs."""

from tools.ac_attn.observe import Observer


class ProfileObserver(Observer):
    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        super().on_training_step_batch_end(model, data_batch, output_batch, loss, iteration)
        pack = data_batch.get("packed_sequence")
        if pack is not None:
            self.emit(
                {
                    "event": "profile_geometry",
                    "iteration": iteration,
                    "micro": self.microbatches,
                    "sample_lens": pack.sample_lens,
                    "split_lens": pack.split_lens,
                    "attn_modes": pack.attn_modes,
                }
            )
