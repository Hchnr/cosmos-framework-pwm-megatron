# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import torch
from torch import nn

from cosmos_framework.data.generator.dataflow.batchers import SequentialPackingBatcher
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.sequence_packing import (
    SequencePlan,
    pack_input_sequence,
    pack_understanding_sequence,
)
from cosmos_framework.data.generator.sequence_packing.static_layout import (
    PACKED_SEQUENCE_KEY,
    StaticPackingConfig,
    bind_static_packed_sequence,
)
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class _FakeVisual(nn.Module):
    spatial_merge_size = 1
    dtype = torch.float32

    def forward(self, pixels: torch.Tensor, *, grid_thw: torch.Tensor):
        count = int(grid_thw.prod(-1).sum().item())
        base = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) + 100
        return base, [base + 1000]


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(256, 4))
        self.visual = _FakeVisual()
        self.config = SimpleNamespace(
            image_token_id=10,
            video_token_id=12,
            vision_start_token_id=9,
            vision_config=SimpleNamespace(spatial_merge_size=1),
        )


def test_vfm_collator_keeps_per_sample_vlm_inputs_aligned() -> None:
    samples = [
        {
            "text_token_ids": torch.tensor([1, 2]),
            "sequence_plan": SequencePlan(has_text=True),
            "pixel_values": torch.ones(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {"text_token_ids": torch.tensor([3]), "sequence_plan": SequencePlan(has_text=True)},
    ]
    batch = VFMListCollator().collate(samples)
    assert len(batch["vlm_inputs"]) == 2
    assert set(batch["vlm_inputs"][0]) == {"pixel_values", "image_grid_thw"}
    assert batch["vlm_inputs"][1] == {}
    assert "pixel_values" not in batch


def test_packed_vlm_scatter_mrope_and_deepstack_stay_in_causal_stream() -> None:
    plans = [
        SequencePlan(has_text=True, text_loss_mask=[False] * 6),
        SequencePlan(has_text=True, text_loss_mask=[False, True]),
    ]
    packed = pack_understanding_sequence(
        plans,
        [[9, 10, 10, 10, 10, 11], [20, 21]],
        {"eos_token_id": 2, "start_of_generation": 3},
    )
    network = Cosmos3VFMNetwork.__new__(Cosmos3VFMNetwork)
    nn.Module.__init__(network)
    network.language_model = _FakeLanguageModel()
    network.hidden_size = 4

    encoded, dtype, deepstack = network._encode_text(
        packed,
        [
            {
                "pixel_values": torch.ones(4, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            {},
        ],
    )
    visual_positions = torch.tensor([1, 2, 3, 4])
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100
    assert dtype == torch.float32
    assert torch.equal(encoded[visual_positions], expected)
    assert len(deepstack) == 1
    assert torch.equal(deepstack[0][visual_positions], expected + 1000)
    assert torch.count_nonzero(deepstack[0][torch.tensor([0, 5, 6, 7])]) == 0
    assert packed.position_ids.shape == (3, 8)


def test_sequential_packing_keeps_model_routing_families_homogeneous() -> None:
    text = torch.arange(4)
    image = {"text_token_ids": text, "images": torch.zeros(3, 32, 32)}
    video = {"text_token_ids": text, "video": torch.zeros(3, 5, 32, 32)}
    forward = {**video, "mode": "forward_dynamics", "action": torch.zeros(4, 8)}
    inverse = {**video, "mode": "inverse_dynamics", "action": torch.zeros(4, 8)}
    batcher = SequentialPackingBatcher(max_samples_per_batch=2)

    batches = list(batcher.batches(iter([image, video, image, video, forward, inverse, forward, inverse])))

    assert [len(batch) for batch in batches] == [2, 2, 2, 2]
    assert all(len({batcher.packing_key(sample) for sample in batch}) == 1 for batch in batches)


def test_sequential_packing_with_single_lookahead_keeps_stream_positions_contiguous() -> None:
    batcher = SequentialPackingBatcher(max_sequence_length=11, lookahead_limit=1)
    batcher.sample_size = lambda sample: sample["size"]
    samples = [
        {"size": 6, "_dp_stream_pos": 0, "video": torch.empty(0)},
        {"size": 6, "_dp_stream_pos": 1, "video": torch.empty(0)},
        {"size": 4, "_dp_stream_pos": 2, "video": torch.empty(0)},
    ]

    batches = list(batcher.batches(iter(samples)))

    assert [[sample["_dp_stream_pos"] for sample in batch] for batch in batches] == [[0], [1, 2]]


def test_attention_topology_accepts_pure_understanding_without_generation_span() -> None:
    hidden = torch.randn(5, 4)
    input_pack, attention_meta, natten = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=["causal"],
        split_lens=[5],
        sample_lens=[5],
        packed_und_token_indexes=torch.arange(5),
        packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
        num_heads=1,
        head_dim=4,
        num_layers=1,
    )

    assert input_pack["causal_seq"].shape == (5, 4)
    assert input_pack["full_only_seq"].shape == (0, 4)
    assert attention_meta.attn_modes == ["causal"]
    assert natten is None


def test_collator_builds_same_t2v_topology_as_model_side_packer() -> None:
    special_tokens = {"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}
    config = StaticPackingConfig(
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        latent_patch_size=2,
        unified_3d_mrope_temporal_modality_margin=15_000,
        enable_fps_modulation=True,
    )
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        condition_frame_indexes_vision=[0],
        supervise_text=True,
        text_loss_mask=[False, True, True],
    )
    sample = {
        "text_token_ids": torch.tensor([10, 11, 12]),
        "video": torch.zeros(3, 9, 32, 48, dtype=torch.uint8),
        "image_size": torch.tensor([32, 48, 32, 48]),
        "conditioning_fps": 24.0,
        "sequence_plan": plan,
    }
    collated = VFMListCollator(static_packing_config=config, special_tokens=special_tokens).collate([sample])
    static = collated[PACKED_SEQUENCE_KEY]

    latent = torch.randn(1, 4, 3, 2, 3)
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[latent],
        x0_tokens_action=[],
        fps_vision=torch.tensor([24.0]),
        fps_action=torch.tensor([24.0]),
    )
    timestep = torch.tensor([[0.2, 0.3, 0.4]])
    bound = bind_static_packed_sequence(static, clean, timestep, latent_patch_size=2)
    legacy = pack_input_sequence(
        [plan],
        [[10, 11, 12]],
        clean,
        timestep,
        special_tokens,
        latent_patch_size=2,
        unified_3d_mrope_temporal_modality_margin=15_000,
        enable_fps_modulation=True,
        temporal_compression_factor=4,
    )

    assert bound.sample_lens == legacy.sample_lens
    assert bound.split_lens == legacy.split_lens
    assert bound.attn_modes == legacy.attn_modes
    assert torch.equal(bound.text_ids, legacy.text_ids)
    assert torch.equal(bound.text_indexes, legacy.text_indexes)
    assert torch.equal(bound.position_ids, legacy.position_ids)
    assert torch.equal(bound.ce_loss_indexes, legacy.ce_loss_indexes)
    assert torch.equal(bound.label_ids, legacy.label_ids)
    assert bound.vision is not None and legacy.vision is not None
    assert torch.equal(bound.vision.sequence_indexes, legacy.vision.sequence_indexes)
    assert torch.equal(bound.vision.mse_loss_indexes, legacy.vision.mse_loss_indexes)
    assert torch.equal(bound.vision.timesteps, legacy.vision.timesteps)
    assert len(bound.vision.tokens) == 1 and bound.vision.tokens[0] is latent


def test_collator_plans_action_and_binds_only_dynamic_payloads() -> None:
    special_tokens = {"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}
    config = StaticPackingConfig()
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=[0],
        condition_frame_indexes_action=[0],
        supervise_text=False,
    )
    sample = {
        "text_token_ids": torch.tensor([10, 11]),
        "video": torch.zeros(3, 9, 32, 32, dtype=torch.uint8),
        "image_size": torch.tensor([32, 32, 32, 32]),
        "conditioning_fps": 24.0,
        "action": torch.randn(8, 7),
        "domain_id": torch.tensor([5]),
        "raw_action_dim": torch.tensor(7),
        "sequence_plan": plan,
    }
    static = VFMListCollator(static_packing_config=config, special_tokens=special_tokens).collate([sample])[
        PACKED_SEQUENCE_KEY
    ]
    latent = torch.randn(1, 4, 3, 2, 2)
    action = torch.randn(8, 64)
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[latent],
        x0_tokens_action=[action],
        fps_vision=torch.tensor([24.0]),
        fps_action=torch.tensor([24.0]),
        action_domain_id=[torch.tensor([5])],
        raw_action_dim=[torch.tensor(7)],
    )
    bound = bind_static_packed_sequence(static, clean, torch.tensor([0.25]), latent_patch_size=2)

    assert bound.action is not None
    assert len(bound.action.tokens) == 1 and bound.action.tokens[0] is action
    assert bound.action.sample_indexes == [0]
    assert bound.action.timesteps.shape == (7,)
    assert torch.all(bound.action.timesteps == 0.25)
    assert bound.action.domain_id[0].item() == 5
