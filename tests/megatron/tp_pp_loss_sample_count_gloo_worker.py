# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Four-rank Gloo check for the production denominator preflight protocol."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.training_backends.megatron_tp_pp import planning
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("tp2pp1dp2", "tp1pp2dp2", "tp2pp2dp1"), required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=40))
    rank = dist.get_rank()
    if args.topology == "tp2pp1dp2":
        tp, pp, dp, model_groups, dp_groups = 2, 1, 2, [(0, 1), (2, 3)], [(0, 2), (1, 3)]
    elif args.topology == "tp1pp2dp2":
        tp, pp, dp, model_groups, dp_groups = 1, 2, 2, [(0, 2), (1, 3)], [(0, 1), (2, 3)]
    else:
        tp, pp, dp, model_groups, dp_groups = 2, 2, 1, [(0, 1, 2, 3)], [(0,), (1,), (2,), (3,)]
    model_group = data_group = None
    for members in model_groups:
        group = dist.new_group(list(members), backend="gloo", timeout=timedelta(seconds=40))
        if rank in members:
            model_members, model_group = members, group
    for members in dp_groups:
        group = dist.new_group(list(members), backend="gloo", timeout=timedelta(seconds=40))
        if rank in members:
            data_members, data_group = members, group
    owner = rank == model_members[0]
    data_rank = data_members.index(rank)
    topology = ModelParallelConfig(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        logical_data_parallel_size=4,
        global_packed_microbatches_per_update=8,
        reference_micro_steps_per_update=2,
    )
    slots = logical_slots(topology, update=7, data_parallel_size=dp, data_parallel_rank=data_rank)
    assert [(s.micro_step, s.logical_rank) for s in slots] == [
        (index // 4, index % 4) for index in range(data_rank, 8, dp)
    ]
    # (is_image, samples, actual vision items). Item count intentionally differs
    # from sample count on several ranks and each reference microstep.
    specs = [
        [(True, 1, 2), (True, 1, 1), (False, 2, 3), (False, 1, 2)],
        [(True, 1, 1), (False, 1, 1), (False, 1, 2), (False, 1, 1)],
    ]

    class Loader:
        template = None
        num_workers = 0
        prefetch_factor = None
        _runner_factory = SimpleNamespace(keywords={"max_live_streams": 1})

        def __init__(self):
            self.slots = iter(slots)

        def __next__(self):
            slot = next(self.slots)
            is_image, samples, items = specs[slot.micro_step][slot.logical_rank]
            vision = ModalityData(tokens=[torch.ones(1)] * items, sequence_indexes=torch.arange(items))
            return {
                "logical_data_rank": slot.logical_rank,
                "dataset_name": "source",
                "is_image": is_image,
                "packed_sequence": PackedSequence(sample_lens=[1] * samples, vision=vision),
            }

        def logical_state_dict(self):
            return {}

        def fingerprint_batch(self, raw):
            return raw

        def commit_batch(self, raw):
            pass

        def close(self):
            pass

    config = SimpleNamespace(
        vision_gen=True, rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=True)
    )
    preparation = SimpleNamespace(is_image_batch=lambda raw: raw["is_image"], config=config)
    preparation.prepare = lambda raw, update, *, logical_rank=None: {
        "sample_count": len(raw["packed_sequence"].sample_lens),
        "packed_sequence": raw["packed_sequence"],
        "loss_inputs": {"is_image_batch": raw["is_image"]},
    }
    # Only substitute device placement: real Gloo broadcasts/all-reduces run.
    for name in ("zeros", "empty", "tensor"):
        original = getattr(torch, name)
        setattr(planning.torch, name, lambda *a, _original=original, **kw: _original(*a, **{**kw, "device": "cpu"}))
    planning.LogicalDataLoader = lambda *args, **kwargs: Loader()
    planning.batch_fingerprint = lambda raw: "same"
    replay, records, totals = planning.preflight_update(
        Loader() if owner else None,
        slots,
        topology,
        preparation if owner else None,
        owner=owner,
        model_group=model_group,
        source_rank=model_members[0],
        dp_group=data_group,
    )
    assert totals["samples"] == [[2, 3], [1, 3]]
    assert totals["loss_samples"] == [[3, 5], [1, 4]]
    if owner:
        total = 0.0
        for slot, record in zip(slots, records, strict=True):
            batch = planning.prepare_slot(
                replay, slot, record, totals, topology, preparation, SimpleNamespace(activate=lambda _: nullcontext())
            )
            image, _, items = specs[slot.micro_step][slot.logical_rank]
            expected = 4 * items / totals["loss_samples"][slot.micro_step][0 if image else 1]
            assert abs(batch["weight"] - expected) < 1e-12
            total += batch["weight"] * (1 + slot.logical_rank + 4 * slot.micro_step)
        contribution = torch.tensor([total], dtype=torch.float64)
        dist.all_reduce(contribution, group=data_group)
        weighted = contribution.item() / 8
        # Compute the native objective directly by modality, then average GA.
        native = 0.0
        for step in range(2):
            for logical in range(4):
                image, _, items = specs[step][logical]
                denom = totals["loss_samples"][step][0 if image else 1]
                native += items / denom * (1 + logical + 4 * step) / 2
        assert abs(weighted - native) < 1e-12
    dist.barrier()
    if rank == 0:
        print(f"{args.topology} passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
