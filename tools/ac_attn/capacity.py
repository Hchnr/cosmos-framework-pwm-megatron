# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experiment-only selection of whole samples fitting a reduced fixed budget.

Never truncate a latent or change the production batcher's fail-fast default.
Logs describe scanned samples (including prefetch), not consumed training steps.
"""

import json
import os
from pathlib import Path

from cosmos_framework.data.generator.dataflow.batchers import SequentialPackingBatcher


class CapacityFilteredBatcher(SequentialPackingBatcher):
    def admitted(self, samples):
        seen = dropped = 0
        for sample in samples:
            seen += 1
            if not self.fixed_sequence_length_enabled:
                yield sample
                continue
            counts = self.sample_token_counts(sample)
            oversized = counts.und > self.fixed_und_tokens - 1 or counts.gen > self.fixed_gen_tokens - 1
            if self.fixed_action_tokens:
                oversized |= (
                    counts.action > self.fixed_action_tokens
                    or counts.video > self.fixed_gen_tokens - self.fixed_action_tokens
                )
            if oversized:
                dropped += 1
                self.record_drop(sample, counts, seen, dropped)
            else:
                yield sample

    def record_drop(self, sample, counts, seen, dropped):
        directory = Path(os.environ["UNIFIED_ATTN_RUN"]) / "capacity"
        directory.mkdir(exist_ok=True)
        path = directory / f"rank{os.environ['RANK']}_pid{os.getpid()}.jsonl"
        row = {
            "sample_id": str(sample.get("__key__", sample.get("catalog_id", "unknown"))),
            "url": str(sample.get("__url__", "")),
            "und": counts.und,
            "gen": counts.gen,
            "action": counts.action,
            "und_capacity": self.fixed_und_tokens - 1,
            "gen_capacity": self.fixed_gen_tokens - 1,
            "stream_seen": seen,
            "stream_dropped": dropped,
        }
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def batches(self, samples):
        yield from super().batches(self.admitted(samples))


def install():
    from cosmos_framework.data.generator import catalog_processors

    catalog_processors.SequentialPackingBatcher = CapacityFilteredBatcher
