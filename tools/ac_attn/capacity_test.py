# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Boundary/order regression for experiment-only whole-sample admission."""

import unittest
from types import SimpleNamespace

from cosmos_framework.data.generator.dataflow.batchers import SequentialPackingBatcher
from tools.ac_attn.capacity import CapacityFilteredBatcher


class Batcher(CapacityFilteredBatcher):
    def __init__(self, action=0):
        super().__init__(
            max_sequence_length=12,
            fixed_sequence_length_enabled=True,
            fixed_und_tokens=4,
            fixed_gen_tokens=8,
            fixed_action_tokens=action,
            special_tokens={},
            lookahead_limit=1,
        )
        self.dropped = []

    def sample_token_counts(self, sample):
        return sample["counts"]

    def record_drop(self, sample, counts, seen, dropped):
        self.dropped.append(sample["__key__"])


def sample(name, und, gen, action=0):
    return {
        "__key__": name,
        "video_latent_ref": {"path": name},
        "counts": SimpleNamespace(und=und, gen=gen, action=action, video=gen - action, total=und + gen),
    }


class CapacityTests(unittest.TestCase):
    def test_preserves_whole_samples_and_order_at_exact_capacity(self):
        a, b, c = sample("a", 1, 2), sample("b", 2, 5), sample("c", 3, 7)
        batcher = Batcher()
        got = list(batcher.batches(iter([a, sample("too_long", 1, 8), b, sample("too_much_text", 4, 1), c])))
        self.assertEqual([[s["__key__"] for s in batch] for batch in got], [["a", "b"], ["c"]])
        self.assertIs(got[0][0], a)
        self.assertIs(got[0][1], b)
        self.assertIs(got[1][0], c)
        self.assertEqual(batcher.dropped, ["too_long", "too_much_text"])

    def test_production_default_still_raises(self):
        with self.assertRaisesRegex(ValueError, "exceeds an UND/GEN capacity"):
            list(SequentialPackingBatcher.batches(Batcher(), iter([sample("too_long", 1, 8)])))

    def test_empty_and_all_oversized(self):
        self.assertEqual(list(Batcher().batches(iter([]))), [])
        self.assertEqual(list(Batcher().batches(iter([sample("too_long", 1, 8)]))), [])

    def test_partitioned_capacity(self):
        batcher = Batcher(action=2)
        keep = sample("keep", 1, 7, action=2)
        got = list(batcher.admitted([keep, sample("action", 1, 7, action=3), sample("video", 1, 7)]))
        self.assertEqual(got, [keep])
        self.assertEqual(batcher.dropped, ["action", "video"])


if __name__ == "__main__":
    unittest.main()
