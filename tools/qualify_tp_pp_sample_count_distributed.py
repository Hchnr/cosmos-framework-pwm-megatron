# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare native sample-level loss with TP/PP preflight on four ranks.

Synthetic inputs exercise production preflight, prepare, decoder and loss
functions. This does not run the 36-layer network, Megatron schedule, or a real
dataset loader. ``gloo`` is for contract tests; ``nccl`` runs on four GPUs.
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.training_backends.megatron_tp_pp import planning
from cosmos_framework.training_backends.megatron_tp_pp.batch import logical_slots
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig
from cosmos_framework.training_backends.megatron_tp_pp.logical_loader import LogicalDataLoader
from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationLoss
from cosmos_framework.training_backends.megatron_tp_pp.rng import LogicalTrainingRNG
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, preserve_training_rng

# image, packed samples, vision item count (None means no vision), noisy
SPECS = (
    ((True, 1, 2, True), (True, 1, 1, True), (False, 2, 3, True), (False, 1, None, False)),
    ((True, 1, 1, True), (False, 1, 2, False), (False, 1, None, False), (False, 1, 1, True)),
)


def topology_groups(topology):
    if topology == "tp2pp1dp2":
        return 2, 1, 2, [(0, 1), (2, 3)], [(0, 2), (1, 3)]
    if topology == "tp2pp2dp1":
        return 2, 2, 1, [(0, 1, 2, 3)], [(0,), (1,), (2,), (3,)]
    raise ValueError(f"Unsupported topology: {topology}")


def fixture(slot):
    image, samples, items, noisy = SPECS[slot.micro_step][slot.logical_rank]
    vision = None
    if items is not None:
        vision = ModalityData(
            tokens=[torch.ones(1, 1, 1, 1, 1)] * items,
            sequence_indexes=torch.arange(items),
            mse_loss_indexes=torch.arange(items) if noisy else torch.empty(0, dtype=torch.long),
            token_shapes=[(1, 1, 1)] * items,
            noisy_frame_indexes=[torch.tensor([0])] * items,
            condition_mask=[torch.zeros(1, 1, 1)] * items,
        )
    return {
        "logical_data_rank": slot.logical_rank,
        "micro_step": slot.micro_step,
        "dataset_name": "synthetic",
        "is_image": image,
        "packed_sequence": PackedSequence(sample_lens=[1] * samples, vision=vision),
    }


class Loader:
    num_workers = 0
    prefetch_factor = None
    _runner_factory = SimpleNamespace(keywords={"max_live_streams": 1})

    def __init__(self, template, state=None, **_):
        self.template = template
        self.slots = template
        self.cursor = 0 if state is None else state["cursor"]
        self.commits = [] if state is None else list(state["commits"])
        self.closed = False

    def __next__(self):
        if self.cursor >= len(self.slots):
            raise StopIteration
        return fixture(self.slots[self.cursor])

    def commit_batch(self, raw):
        self.commits.append((raw["micro_step"], raw["logical_data_rank"]))
        self.cursor += 1

    def logical_state_dict(self):
        return {"cursor": self.cursor, "commits": list(self.commits)}

    def fingerprint_batch(self, batch):
        return batch

    def close(self):
        self.closed = True


class RealLoaderAdapter:
    """Decorate real logical batches with deterministic synthetic packed payloads."""

    num_workers = 0
    prefetch_factor = None

    def __init__(self, template, state, slots):
        self.template = template
        self.pool = LogicalDataLoader(template, state, num_workers=0)
        self._runner_factory = self.pool._runner_factory
        self.slots = slots
        self.read_index = 0
        self.commits = []
        self.closed = False

    def __next__(self):
        raw = next(self.pool)
        slot = self.slots[self.read_index]
        if raw["logical_data_rank"] != slot.logical_rank:
            raise ValueError("Logical loader order does not match expected slot")
        # Preserve native loader metadata, samples, augmentation and cursor;
        # add only deterministic synthetic packed inputs for this experiment.
        extra = fixture(slot)
        raw.update({key: value for key, value in extra.items() if key != "dataset_name"})
        self.read_index += 1
        return raw

    def commit_batch(self, raw):
        self.pool.commit_batch(raw)
        self.commits.append((raw["micro_step"], raw["logical_data_rank"]))

    def logical_state_dict(self):
        return self.pool.logical_state_dict()

    def fingerprint_batch(self, raw):
        return self.pool.fingerprint_batch(raw)

    def close(self):
        self.pool.close()
        self.closed = True


def make_real_loader(slots):
    """Use the repository's synthetic native streams and actual logical loader."""
    from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/megatron"))
    from tp_pp_logical_loader_test import native

    states = {}
    with preserve_training_rng():
        for logical_rank in sorted({slot.logical_rank for slot in slots}):
            source = native(logical_rank)
            callback = JointCosmosDataLoaderStateCallback(source)
            callback._update_state_from_batch(next(source))
            states[logical_rank] = copy.deepcopy(callback.state_dict())
            source.close()
        template = native(0)
    return RealLoaderAdapter(template, states, slots)


class LogicalRng:
    def __init__(self, device):
        self.device = device
        self.production = None
        if device.type == "cuda":
            states = {}
            with preserve_training_rng():
                for rank in range(4):
                    torch.random.default_generator.manual_seed(9010 + rank)
                    torch.cuda.manual_seed(8040 + rank)
                    np.random.seed(7000 + rank)
                    random.seed(6000 + rank)
                    states[rank] = get_rand_state_dict()
            self.production = LogicalTrainingRNG(states)
            return
        self.cpu = {rank: torch.Generator().manual_seed(9010 + rank).get_state() for rank in range(4)}
        self.accel = (
            {rank: torch.Generator(device=device).manual_seed(8040 + rank).get_state() for rank in range(4)}
            if device.type == "cuda"
            else None
        )

    @contextmanager
    def activate(self, logical_rank):
        if self.production is not None:
            with self.production.activate(logical_rank):
                yield
            return
        outer_cpu = torch.random.get_rng_state()
        outer_accel = torch.cuda.get_rng_state(self.device) if self.accel is not None else None
        torch.random.set_rng_state(self.cpu[logical_rank])
        if self.accel is not None:
            torch.cuda.set_rng_state(self.accel[logical_rank], self.device)
        try:
            yield
        finally:
            self.cpu[logical_rank] = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(outer_cpu)
            if self.accel is not None:
                self.accel[logical_rank] = torch.cuda.get_rng_state(self.device).clone()
                torch.cuda.set_rng_state(outer_accel, self.device)

    def digest(self, logical_rank):
        if self.production is not None:
            state = self.production.states[logical_rank]
            return [hashlib.sha256(state[key].contiguous().numpy().tobytes()).hexdigest() for key in sorted(state)]
        values = [self.cpu[logical_rank]]
        if self.accel is not None:
            values.append(self.accel[logical_rank])
        return [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in values]


def make_config(vision_gen):
    return SimpleNamespace(
        predict_text_tokens=False,
        vision_gen=vision_gen,
        action_gen=True,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            sample_level_loss_averaging=True,
            normalize_loss_by_active=False,
            image_loss_scale=1.3,
            loss_scale=2.0,
            action_loss_weight=3.0,
        ),
    )


def make_preparation(config, device):
    def prepare(raw, update):
        del update
        packed = copy.deepcopy(raw["packed_sequence"])
        # The native loss expects a vision modality object even for its dummy
        # vision prediction. The static preflight deliberately sees None.
        if config.vision_gen and packed.vision is None:
            packed.vision = ModalityData(tokens=None, condition_mask=[])
        packed.action = ModalityData(
            tokens=[torch.ones(1, 1, device=device)],
            mse_loss_indexes=torch.tensor([0], device=device),
            condition_mask=[torch.zeros(1, 1, device=device)],
            raw_action_dim=None,
        )
        if packed.vision is not None:
            packed.vision.mse_loss_indexes = packed.vision.mse_loss_indexes.to(device)
            packed.vision.condition_mask = [item.to(device) for item in packed.vision.condition_mask]
            packed.vision.noisy_frame_indexes = [item.to(device) for item in packed.vision.noisy_frame_indexes]
            if packed.vision.tokens is not None:
                packed.vision.tokens = [item.to(device) for item in packed.vision.tokens]
        cpu_noise = torch.rand(())
        device_noise = torch.rand((), device=device)
        noise = device_noise + cpu_noise.to(device)
        if device.type == "cuda":
            noise = noise + random.random() + float(np.random.random())
        item_count = 0 if packed.vision is None or packed.vision.tokens is None else len(packed.vision.tokens)
        target = [noise.expand(1, 1, 1, 1) for _ in range(item_count)]
        return {
            "sample_count": len(packed.sample_lens),
            "packed_sequence": packed,
            "noise": noise,
            "loss_inputs": {
                "gen_data_noised": SimpleNamespace(vt_target_vision=target, vt_target_action=[noise.expand(1, 1)]),
                "timesteps": torch.ones(max(1, item_count), 1, device=device),
                "timesteps_action": torch.ones(1, 1, device=device),
                "timesteps_sound": None,
                "is_image_batch": raw["is_image"],
            },
        }

    return SimpleNamespace(config=config, is_image_batch=lambda raw: raw["is_image"], prepare=prepare)


def make_model(config, device):
    model = GenerationLoss.__new__(GenerationLoss)
    torch.nn.Module.__init__(model)
    model.config, model.parallel_dims = config, None
    model.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}
    flow = SimpleNamespace(train_time_weight=lambda ts, _: torch.ones_like(ts))
    model.rectified_flow_image = model.rectified_flow_video = model.rectified_flow_action = flow
    return model


def outputs(batch, parameter):
    output = {"preds_action": [parameter.reshape(1, 1) * 1.7]}
    vision = batch["packed_sequence"].vision
    if vision is not None:
        count = 1 if vision.tokens is None else max(1, len(vision.tokens))
        decoder = SimpleNamespace(
            patch_latent_dim=1,
            latent_patch_size=1,
            latent_channel=1,
            vae2llm=torch.nn.Identity(),
            llm2vae=torch.nn.Identity(),
        )
        decoder.unpatchify_and_unpack_latents = lambda *a, **kw: Cosmos3VFMNetwork.unpatchify_and_unpack_latents(
            decoder, *a, **kw
        )
        hidden = torch.arange(1, count + 1, device=parameter.device, dtype=torch.float32).reshape(-1, 1) * parameter
        Cosmos3VFMNetwork._decode_vision(decoder, batch["packed_sequence"], hidden, output)
    return output


def override_cpu_placement():
    # Only test-time placement is changed; the production distributed calls
    # still execute on actual four-rank Gloo groups.
    for name in ("zeros", "empty", "tensor"):
        original = getattr(torch, name)

        def wrapped(*args, _original=original, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        setattr(planning.torch, name, wrapped)


def run_case(args, device, topology, model_group, data_group, model_members, data_members, vision_gen):
    rank = dist.get_rank()
    owner = rank == model_members[0]
    data_rank = data_members.index(rank)
    dp = len(data_members)
    slots = logical_slots(topology, update=7, data_parallel_size=dp, data_parallel_rank=data_rank)
    config = make_config(vision_gen)
    preparation = make_preparation(config, device)
    model = make_model(config, device)
    baseline_rng, candidate_rng = LogicalRng(device), LogicalRng(device)
    outer_cpu = torch.random.get_rng_state().clone()
    outer_cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    outer_python = pickle.dumps(random.getstate())
    outer_numpy = pickle.dumps(np.random.get_state())

    # Native: each physical rank stands in for one of the four original logical
    # DP ranks. _sample_level_loss_scale uses a real WORLD collective.
    native_parameter = torch.tensor(0.6, device=device, requires_grad=True)
    native_total = torch.zeros((), device=device)
    native_noise = {}
    for micro in range(2):
        slot = SimpleNamespace(update=7, micro_step=micro, logical_rank=rank)
        raw = fixture(slot)
        with baseline_rng.activate(rank):
            batch = preparation.prepare(raw, slot.update)
        native_noise[(micro, rank)] = float(batch["noise"].item())
        loss, _ = OmniMoTModel._compute_losses(
            model,
            out_net=outputs(batch, native_parameter),
            data_batch_packed=batch["packed_sequence"],
            **batch["loss_inputs"],
        )
        native_total = native_total + loss.detach() / 8
        (loss / 8).backward()
    native_grad = native_parameter.grad.detach().clone()
    dist.all_reduce(native_total)
    dist.all_reduce(native_grad)
    native_rng = [None] * 4
    native_noises = [None] * 4
    dist.all_gather_object(native_rng, baseline_rng.digest(rank))
    dist.all_gather_object(native_noises, native_noise)

    initial_loader = make_real_loader(slots) if owner else None
    if owner:
        starting_loader_state = initial_loader.logical_state_dict()
    planning.LogicalDataLoader = lambda template, state, **_: RealLoaderAdapter(template, state, slots)
    if vision_gen:
        replay, records, totals = planning.preflight_update(
            initial_loader,
            slots,
            topology,
            preparation if owner else None,
            owner=owner,
            model_group=model_group,
            source_rank=model_members[0],
            dp_group=data_group,
        )
        assert totals["samples"] == [[2, 3], [1, 3]]
        assert totals["loss_samples"] == [[3, 4], [1, 4]]
        if owner:
            assert initial_loader.closed and len(initial_loader.commits) == len(slots)
            assert replay.read_index == 0 and not replay.commits
            assert planning.batch_fingerprint(replay.logical_state_dict()) == planning.batch_fingerprint(
                starting_loader_state
            )
    else:
        replay, records, totals = initial_loader, ([{} for _ in slots] if owner else []), None

    parameter = torch.tensor(0.6, device=device, requires_grad=True)
    local_total = torch.zeros((), device=device)
    owner_result = None
    if owner:
        if vision_gen:
            # A changed prepared item count must fail before a real logical
            # loader commit. This uses an independent replay at the same cursor.
            rejected = RealLoaderAdapter(initial_loader.template, starting_loader_state, slots)
            broken_preparation = copy.copy(preparation)

            def changed_count(raw, update):
                batch = preparation.prepare(raw, update)
                batch["packed_sequence"].vision.tokens.append(torch.ones(1, device=device))
                return batch

            broken_preparation.prepare = changed_count
            try:
                planning.prepare_slot(
                    rejected,
                    slots[0],
                    copy.deepcopy(records[0]),
                    totals,
                    topology,
                    broken_preparation,
                    LogicalRng(device),
                )
            except ValueError as error:
                if "prediction count" not in str(error):
                    raise
            else:
                raise AssertionError("Changed prediction count was not rejected")
            assert not rejected.commits
            assert planning.batch_fingerprint(rejected.logical_state_dict()) == planning.batch_fingerprint(
                starting_loader_state
            )
            rejected.close()
        noises = {}
        for slot, record in zip(slots, records, strict=True):
            batch = planning.prepare_slot(
                replay,
                slot,
                record,
                totals,
                topology,
                preparation,
                candidate_rng,
                hash_raw=vision_gen,
            )
            noises[(slot.micro_step, slot.logical_rank)] = float(batch["noise"].item())
            expected_noise = native_noises[slot.logical_rank][(slot.micro_step, slot.logical_rank)]
            assert noises[(slot.micro_step, slot.logical_rank)] == expected_noise
            if vision_gen:
                image, _, items, _ = SPECS[slot.micro_step][slot.logical_rank]
                n = 1 if items is None else items
                assert batch["weight"] == 4 * n / totals["loss_samples"][slot.micro_step][0 if image else 1]
            else:
                assert "weight" not in batch
            loss, _ = model(outputs(batch, parameter), batch)
            local_total = local_total + loss.detach() / len(slots) / dp
            (loss / len(slots) / dp).backward()
        assert replay.commits == [(slot.micro_step, slot.logical_rank) for slot in slots]
        for logical_rank, state in replay.logical_state_dict().items():
            assert state["global_id"] == starting_loader_state[logical_rank]["global_id"] + 2
        for logical_rank in {slot.logical_rank for slot in slots}:
            assert candidate_rng.digest(logical_rank) == native_rng[logical_rank]
        replay.close()
        local_grad = parameter.grad.detach().clone()
        dist.all_reduce(local_total, group=data_group)
        dist.all_reduce(local_grad, group=data_group)
        torch.testing.assert_close(local_total, native_total, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(local_grad, native_grad, rtol=2e-6, atol=2e-6)
        owner_result = {"loss": float(local_total), "grad": float(local_grad), "slots": len(slots)}
    assert torch.equal(torch.random.get_rng_state(), outer_cpu)
    if outer_cuda is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), outer_cuda)
    assert pickle.dumps(random.getstate()) == outer_python
    assert pickle.dumps(np.random.get_state()) == outer_numpy
    result = [owner_result]
    dist.broadcast_object_list(result, src=model_members[0], group=model_group, device=device)
    assert result[0] is not None
    dist.barrier()
    return {
        "vision_gen": vision_gen,
        "native_loss": float(native_total),
        "native_grad": float(native_grad),
        **result[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("tp2pp1dp2", "tp2pp2dp1"), required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cpu")
        override_cpu_placement()
    dist.init_process_group(args.backend, timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("The qualification requires four ranks")
    tp, pp, dp, model_groups, dp_groups = topology_groups(args.topology)
    model_group = data_group = None
    for members in model_groups:
        group = dist.new_group(list(members), backend=args.backend, timeout=timedelta(seconds=180))
        if rank in members:
            model_members, model_group = members, group
    for members in dp_groups:
        group = dist.new_group(list(members), backend=args.backend, timeout=timedelta(seconds=180))
        if rank in members:
            data_members, data_group = members, group
    topology = ModelParallelConfig(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        logical_data_parallel_size=4,
        global_packed_microbatches_per_update=8,
        reference_micro_steps_per_update=2,
    )
    assert topology.data_parallel_size(4) == dp
    cases = [
        run_case(args, device, topology, model_group, data_group, model_members, data_members, vision_gen)
        for vision_gen in (True, False)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{args.topology}.rank{rank}.json"
    with target.open("x") as handle:
        json.dump(
            {
                "passed": True,
                "topology": args.topology,
                "backend": args.backend,
                "rank": rank,
                "scope": "synthetic production preflight, decoder and loss; not full-model acceptance",
                "loader": "actual LogicalDataLoader with synthetic native streams and packed payloads",
                "production_logical_rng": args.backend == "nccl",
                "covered": [
                    "multiple vision items",
                    "absent vision dummy",
                    "no noisy vision",
                    "vision_gen false",
                    "uncommitted mismatch",
                    "loss",
                    "gradient",
                    "RNG",
                ],
                "cases": cases,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
