# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Preserve native vision-item denominators and separate reporting sample counts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.training_backends.megatron_tp_pp import planning
from cosmos_framework.training_backends.megatron_tp_pp.batch import LogicalSlot, reference_loss_weight
from cosmos_framework.training_backends.megatron_tp_pp.config import ModelParallelConfig


def vision(items, *, noisy):
    return ModalityData(
        tokens=[torch.ones(1, 1, 1, 1, 1) for _ in range(items)],
        sequence_indexes=torch.arange(items),
        mse_loss_indexes=torch.arange(items) if noisy else torch.empty(0, dtype=torch.long),
        token_shapes=[(1, 1, 1)] * items,
        noisy_frame_indexes=[torch.tensor([0])] * items,
    )


def native_prediction_count(vision_data):
    decoder = SimpleNamespace(
        patch_latent_dim=1,
        latent_patch_size=1,
        latent_channel=1,
        vae2llm=torch.nn.Identity(),
        llm2vae=torch.nn.Identity(),
    )
    decoder.unpatchify_and_unpack_latents = lambda *args, **kwargs: Cosmos3VFMNetwork.unpatchify_and_unpack_latents(
        decoder, *args, **kwargs
    )
    output = {}
    Cosmos3VFMNetwork._decode_vision(
        decoder,
        PackedSequence(vision=vision_data),
        torch.ones(max(1, len(vision_data.tokens) if vision_data is not None else 1), 1),
        output,
    )
    return len(output["preds_vision"])


def test_absent_vision_payload_uses_native_dummy_count():
    vision_data = vision(2, noisy=False)
    vision_data.tokens = None
    decoder = SimpleNamespace(
        patch_latent_dim=1,
        vae2llm=torch.nn.Identity(),
        llm2vae=torch.nn.Identity(),
    )
    output = {}
    Cosmos3VFMNetwork._decode_vision(decoder, PackedSequence(vision=vision_data), torch.ones(1, 1), output)
    assert (
        planning._native_vision_prediction_count(PackedSequence(vision=vision_data)) == len(output["preds_vision"]) == 1
    )


@pytest.mark.parametrize("items,noisy", [(1, True), (2, True), (1, False), (3, False), (None, False)])
def test_preflight_count_matches_native_vision_decoder(items, noisy):
    vision_data = None if items is None else vision(items, noisy=noisy)
    packed = PackedSequence(sample_lens=[2], vision=vision_data)
    slot = LogicalSlot(update=0, micro_step=0, logical_rank=0)
    raw = {"logical_data_rank": 0, "packed_sequence": packed, "dataset_name": "source"}
    preparation = SimpleNamespace(
        is_image_batch=lambda _: False,
        config=SimpleNamespace(
            vision_gen=True, rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=True)
        ),
    )
    loader = SimpleNamespace(fingerprint_batch=lambda batch: batch)
    record = planning._raw_record(loader, raw, slot, preparation, hash_raw=False)
    assert record["samples"] == 1
    assert record["loss_samples"] == native_prediction_count(vision_data)


def test_preflight_uses_native_item_denominator_for_each_microstep_and_modality(monkeypatch):
    # Four logical ranks, two reference microsteps. Two image ranks have one
    # sample each but 2 and 1 vision items, respectively; video totals differ.
    topology = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        global_packed_microbatches_per_update=8,
        logical_data_parallel_size=4,
        reference_micro_steps_per_update=2,
    )
    descriptions = [
        (0, 0, True, 1, 2),
        (0, 1, True, 1, 1),
        (0, 2, False, 2, 3),
        (0, 3, False, 1, 2),
        (1, 0, True, 1, 1),
        (1, 1, False, 1, 1),
        (1, 2, False, 1, 2),
        (1, 3, False, 1, 1),
    ]
    slots = [LogicalSlot(0, micro, rank) for micro, rank, _, _, _ in descriptions]
    raws = [
        {
            "logical_data_rank": rank,
            "dataset_name": "source",
            "is_image": image,
            "packed_sequence": PackedSequence(sample_lens=[2] * samples, vision=vision(items, noisy=True)),
        }
        for _, rank, image, samples, items in descriptions
    ]

    class Loader:
        num_workers = 0
        prefetch_factor = None
        template = None
        _runner_factory = SimpleNamespace(keywords={"max_live_streams": 1})

        def __init__(self):
            self.iterator = iter(raws)

        def __next__(self):
            return next(self.iterator)

        def logical_state_dict(self):
            return {}

        def fingerprint_batch(self, raw):
            return raw

        def commit_batch(self, raw):
            pass

        def close(self):
            pass

    preparation = SimpleNamespace(
        is_image_batch=lambda raw: raw["is_image"],
        config=SimpleNamespace(
            vision_gen=True, rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=True)
        ),
    )
    preparation.prepare = lambda raw, update, *, logical_rank=None: {
        "sample_count": len(raw["packed_sequence"].sample_lens),
        "packed_sequence": raw["packed_sequence"],
        "loss_inputs": {"is_image_batch": raw["is_image"]},
    }
    for name in ("zeros", "empty", "tensor"):
        original = getattr(torch, name)
        monkeypatch.setattr(
            planning.torch, name, lambda *a, _original=original, **kw: _original(*a, **{**kw, "device": "cpu"})
        )
    monkeypatch.setattr(planning.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(planning.dist, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(planning, "LogicalDataLoader", lambda *args, **kwargs: Loader())
    monkeypatch.setattr(planning, "batch_fingerprint", lambda raw: "unchanged")
    replay, records, totals = planning.preflight_update(
        Loader(), slots, topology, preparation, owner=True, model_group=None, source_rank=0, dp_group=None
    )
    sample_totals = totals["samples"] if isinstance(totals, dict) else totals
    assert sample_totals == [[2, 3], [1, 3]]
    expected_weights = [8 / 3, 4 / 3, 12 / 5, 8 / 5, 4, 1, 2, 1]
    rng = SimpleNamespace(activate=lambda rank: nullcontext())
    weights = [
        planning.prepare_slot(replay, slot, record, totals, topology, preparation, rng)["weight"]
        for slot, record in zip(slots, records, strict=True)
    ]
    assert weights == pytest.approx(expected_weights)
    assert totals["loss_samples"] == [[3, 5], [1, 4]]
    # Undo schedule/physical-DP averaging and compare with native image weights.
    assert (weights[0] * 2 + weights[1] * 1) / 8 == pytest.approx((4 / 3 * 2 + 2 / 3 * 1) / 2 / 2)
    assert reference_loss_weight(topology, local_samples=2, reference_modality_samples=3) == pytest.approx(8 / 3)


@pytest.mark.parametrize("static_items,prepared_items", [(2, 1), (None, 2)])
def test_prepare_refuses_changed_native_loss_count_before_committing(monkeypatch, static_items, prepared_items):
    slot = LogicalSlot(update=0, micro_step=0, logical_rank=0)
    static = PackedSequence(sample_lens=[2], vision=None if static_items is None else vision(static_items, noisy=False))
    changed = PackedSequence(sample_lens=[2], vision=vision(prepared_items, noisy=False))
    raw = {"logical_data_rank": 0, "packed_sequence": static, "dataset_name": "source"}

    class Loader:
        committed = False

        def __next__(self):
            return raw

        def fingerprint_batch(self, batch):
            return batch

        def commit_batch(self, batch):
            self.committed = True

    loader = Loader()
    preparation = SimpleNamespace(
        is_image_batch=lambda _: False,
        config=SimpleNamespace(
            vision_gen=True, rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=True)
        ),
        prepare=lambda raw, update, *, logical_rank=None: {
            "sample_count": 1,
            "packed_sequence": changed,
            "loss_inputs": {"is_image_batch": False},
        },
    )
    record = planning._raw_record(loader, raw, slot, preparation, hash_raw=False)
    record["sha256"] = "same"
    monkeypatch.setattr(planning, "batch_fingerprint", lambda _: "same")
    topology = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        global_packed_microbatches_per_update=1,
        logical_data_parallel_size=1,
        reference_micro_steps_per_update=1,
    )
    with pytest.raises(ValueError, match="prediction count"):
        planning.prepare_slot(
            loader,
            slot,
            record,
            {"samples": [[0, 1]], "loss_samples": [[0, 2]]},
            topology,
            preparation,
            SimpleNamespace(activate=lambda _: nullcontext()),
        )
    assert not loader.committed


def test_no_vision_sample_averaging_uses_native_unscaled_loss_and_no_preflight():
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationLoss

    config = SimpleNamespace(
        vision_gen=False,
        action_gen=True,
        sound_gen=False,
        predict_text_tokens=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False, sample_level_loss_averaging=True
        ),
    )
    packed = PackedSequence(sample_lens=[2], vision=None, action=None)
    raw = {"logical_data_rank": 0, "packed_sequence": packed, "dataset_name": "source"}

    class Loader:
        committed = False

        def __next__(self):
            return raw

        def commit_batch(self, batch):
            self.committed = True

    preparation = SimpleNamespace(
        config=config,
        is_image_batch=lambda _: False,
        prepare=lambda raw, update, *, logical_rank=None: {
            "sample_count": 1,
            "packed_sequence": packed,
            "loss_inputs": {
                "gen_data_noised": None,
                "timesteps": None,
                "timesteps_action": None,
                "timesteps_sound": None,
                "is_image_batch": False,
            },
        },
    )
    topology = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        global_packed_microbatches_per_update=1,
        logical_data_parallel_size=1,
        reference_micro_steps_per_update=1,
    )
    loader = Loader()
    batch = planning.prepare_slot(
        loader,
        LogicalSlot(0, 0, 0),
        {},
        None,
        topology,
        preparation,
        SimpleNamespace(activate=lambda _: nullcontext()),
        hash_raw=False,
    )
    assert loader.committed
    assert "weight" not in batch
    model = GenerationLoss.__new__(GenerationLoss)
    torch.nn.Module.__init__(model)
    model.config = config
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model._get_load_balancing_loss_meshes = lambda: (None, None)
    outputs = {"preds_action": [torch.ones(1, requires_grad=True)]}
    native, _ = OmniMoTModel._compute_losses(model, out_net=outputs, data_batch_packed=packed, **batch["loss_inputs"])
    actual, _ = model(outputs, batch)
    torch.testing.assert_close(actual, native, atol=0, rtol=0)
