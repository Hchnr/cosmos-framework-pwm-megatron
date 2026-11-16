# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.omni_mot_model import (
    OmniMoTModel,
    _configure_reasoner_visual_trainability,
    _prepare_full_model_instance,
)


class _FakeQwenModel:
    pass


class _FakeEdgeModel:
    @staticmethod
    def initialize_reasoner_vision_tower() -> None:
        pass


@pytest.mark.parametrize("batched_metadata", [False, True])
@pytest.mark.parametrize(
    ("image_size", "expected_hw"),
    [
        ([80, 112], (5, 7)),
        ([80, 112, 80, 112], (5, 7)),
        ([80, 112, 48, 64], (3, 4)),
    ],
    ids=["action128_canvas", "legacy_unpadded", "legacy_padded"],
)
def test_latent_crop_accepts_canvas_and_padding_metadata(image_size, expected_hw, batched_metadata) -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.tokenizer_vision_gen = SimpleNamespace(spatial_compression_factor=16)
    latent = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(1, 2, 3, 5, 7).requires_grad_()
    metadata = torch.tensor(image_size)
    if batched_metadata:
        metadata = metadata.unsqueeze(0)

    (cropped,) = model._remove_padding_from_latent([latent], [metadata])

    height, width = expected_hw
    torch.testing.assert_close(cropped, latent[..., :height, :width])
    assert cropped.is_contiguous()
    cropped.sum().backward()
    expected_grad = torch.zeros_like(latent)
    expected_grad[..., :height, :width] = 1
    torch.testing.assert_close(latent.grad, expected_grad)


def test_latent_crop_handles_mixed_sample_dimensions() -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.tokenizer_vision_gen = SimpleNamespace(spatial_compression_factor=16)
    canvas = torch.randn(1, 2, 3, 5, 7)
    padded = torch.randn(1, 2, 3, 4, 6)

    cropped = model._remove_padding_from_latent(
        [canvas, padded], [torch.tensor([[80, 112]]), torch.tensor([64, 96, 32, 48])]
    )

    torch.testing.assert_close(cropped[0], canvas)
    torch.testing.assert_close(cropped[1], padded[..., :2, :3])


@pytest.mark.parametrize("shape", [(1,), (3,), (5,), (2, 2)])
def test_latent_crop_rejects_invalid_size_metadata(shape) -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.tokenizer_vision_gen = SimpleNamespace(spatial_compression_factor=16)

    with pytest.raises(ValueError, match="image_size must contain"):
        model._remove_padding_from_latent([torch.zeros(1, 2, 3, 4, 5)], [torch.ones(shape)])


def test_action_metadata_device_transfer_accepts_published_integer_width(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor.clone())
    legacy_width = torch.tensor(64)
    action = ModalityData(raw_action_dim=[128, legacy_width, None])
    action.to_cuda()
    assert action.raw_action_dim[0] == 128
    assert action.raw_action_dim[2] is None
    torch.testing.assert_close(action.raw_action_dim[1], legacy_width)
    assert action.raw_action_dim[1] is not legacy_width


@pytest.mark.parametrize("enabled", [True, False])
def test_full_model_visual_construction_follows_single_load_flag(enabled: bool) -> None:
    qwen, _, external = _prepare_full_model_instance(
        {"_target_": _FakeQwenModel, "config": {"include_visual": not enabled}},
        load_vlm_visual=enabled,
        include_generation_tower=False,
    )
    assert external is False
    assert qwen["config"]["include_visual"] is enabled
    assert qwen["config"]["include_generation_tower"] is False

    edge, _, external = _prepare_full_model_instance(
        {"_target_": _FakeEdgeModel, "config": {"include_visual": None}},
        load_vlm_visual=enabled,
        include_generation_tower=True,
    )
    assert external is True
    assert edge["config"]["include_visual"] is None
    assert edge["config"]["include_generation_tower"] is True


def test_full_model_specialization_accepts_structured_hydra_config() -> None:
    config = OmegaConf.create(
        {"_target_": _FakeQwenModel, "config": {"include_visual": True}},
        flags={"allow_objects": True},
    )
    OmegaConf.set_struct(config, True)
    prepared, _, _ = _prepare_full_model_instance(
        config,
        load_vlm_visual=True,
        include_generation_tower=False,
    )
    assert prepared.config.include_generation_tower is False
    assert prepared.config.include_visual is True


def test_full_model_visual_construction_supports_structured_qwen_lazy_config() -> None:
    config = OmegaConf.create(
        {"_target_": _FakeQwenModel, "config": {"base_config": {}}}, flags={"allow_objects": True}
    )
    OmegaConf.set_struct(config, True)

    prepared, _, external = _prepare_full_model_instance(
        config,
        load_vlm_visual=True,
        include_generation_tower=True,
    )

    assert external is False
    assert prepared["config"]["include_visual"] is True
    assert "include_visual" not in config["config"]


@pytest.mark.parametrize("trainable", [False, True])
def test_reasoner_visual_trainability_is_explicit(trainable: bool) -> None:
    language_model = nn.Module()
    language_model.visual = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))

    _configure_reasoner_visual_trainability(
        language_model,
        load_vlm_visual=True,
        train_vlm_visual=trainable,
    )

    assert all(parameter.requires_grad is trainable for parameter in language_model.visual.parameters())
    assert language_model._reasoner_visual_artifact is True
    assert language_model._reasoner_visual_trainable is trainable


def test_training_visual_requires_loading_visual() -> None:
    with pytest.raises(ValueError, match="train_vlm_visual=True requires load_vlm_visual=True"):
        _configure_reasoner_visual_trainability(
            nn.Module(),
            load_vlm_visual=False,
            train_vlm_visual=True,
        )


def test_loading_visual_requires_family_native_visual_tower() -> None:
    with pytest.raises(ValueError, match="family-native model to expose a visual tower"):
        _configure_reasoner_visual_trainability(
            nn.Module(),
            load_vlm_visual=True,
            train_vlm_visual=False,
        )


def test_unified_checkpoint_keeps_visual_tower_weights() -> None:
    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.visual = nn.Linear(2, 2)
            self.language_model._reasoner_visual_artifact = True

    model = OmniMoTModel.__new__(OmniMoTModel)
    model.config = SimpleNamespace(exclude_reasoner_weights_from_checkpoint=False)
    state = model._net_state_dict(_Net())
    assert "language_model.visual.weight" in state
    assert "language_model.visual.bias" in state


def test_pure_understanding_denoise_does_not_require_vision_predictions() -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.config = SimpleNamespace(vision_gen=False, action_gen=False, sound_gen=False)
    ce_preds = torch.randn(2, 3)
    packed_sequence = Mock()
    net = Mock(return_value={"last_hidden_state": torch.randn(2, 3), "ce_preds": ce_preds})

    result = model.denoise(
        net=net,
        data_batch_packed=packed_sequence,
        vlm_inputs=[],
    )

    assert set(result) == {"ce_preds"}
    assert result["ce_preds"] is ce_preds
    net.assert_called_once_with(
        packed_seq=packed_sequence,
        memory=None,
        video_temporal_causal=None,
        vlm_inputs=[],
    )


def test_fixed_pure_understanding_step_reports_real_and_physical_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.config = SimpleNamespace(
        und_loss_scale=1.0,
        fixed_sequence_length=SimpleNamespace(enabled=True, understanding_tokens=16),
    )
    packed_sequence = PackedSequence(
        sample_lens=[2, 1],
        split_lens=[2, 1],
        attn_modes=["causal", "causal"],
        sequence_length=3,
        text_ids=torch.tensor([4, 5, 6]),
        text_indexes=torch.arange(3),
        label_ids=torch.tensor([0, 1]),
        ce_loss_weights=torch.ones(2),
    )
    model._get_understanding_inputs = Mock(return_value=(None, None, [], packed_sequence))
    model.denoise = Mock(return_value={"ce_preds": torch.tensor([[2.0, 0.0], [0.0, 2.0]])})
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)

    output, loss = model._training_step_und({}, iteration=0)

    assert output["loss"] is loss
    assert output["task_type"] == "understanding"
    assert output["fixed_profile"] == "understanding"
    assert output["und_token_length"] == 3
    assert output["gen_token_length"] == 0
    assert output["physical_und_token_length"] == 16
    assert output["physical_gen_token_length"] == 0
    assert output["batch_size"] == 2


def test_reasoner_only_setup_skips_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=False,
        sound_gen=False,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )
    instantiated = []

    def _instantiate(candidate):
        instantiated.append(candidate)
        return vlm_processor

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert instantiated == [vlm_config.tokenizer]
    assert model.tokenizer_vision_gen is None
    assert model.tokenizer_sound_gen is None


def test_default_setup_loads_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vision_tokenizer = SimpleNamespace(latent_ch=48, reset_dtype=Mock())
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=True,
        sound_gen=False,
        state_ch=48,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )

    def _instantiate(candidate):
        if candidate == vlm_config.tokenizer:
            return vlm_processor
        assert candidate is vision_config
        return vision_tokenizer

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert model.tokenizer_vision_gen is vision_tokenizer
    vision_tokenizer.reset_dtype.assert_called_once_with()


def test_generation_text_ce_is_data_driven_and_weighted() -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.config = SimpleNamespace(
        predict_text_tokens=False,
        text_loss_weight=2.0,
        vision_gen=False,
        action_gen=False,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False,
            sample_level_loss_averaging=False,
        ),
    )
    model.tensor_kwargs_fp32 = {"device": torch.device("cpu"), "dtype": torch.float32}
    packed = PackedSequence(
        label_ids=torch.tensor([1, 0]),
        ce_loss_weights=torch.ones(2),
    )
    logits = torch.tensor([[0.0, 2.0], [3.0, 0.0]], requires_grad=True)

    total, losses = model._compute_losses(
        {"ce_preds": logits},
        packed,
        gen_data_noised=SimpleNamespace(),
        timesteps=torch.empty(0),
        is_image_batch=False,
    )
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    assert torch.allclose(losses["cross_entropy_loss_text"], expected)
    assert torch.allclose(total, expected * 2.0)


def test_generation_reasoning_ce_and_vision_flow_share_one_loss() -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.config = SimpleNamespace(
        predict_text_tokens=False,
        text_loss_weight=2.0,
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False,
            sample_level_loss_averaging=False,
            image_loss_scale=None,
            loss_scale=10.0,
        ),
    )
    model.tensor_kwargs_fp32 = {"device": torch.device("cpu"), "dtype": torch.float32}
    model.rectified_flow_video = object()
    flow_loss = torch.tensor(0.25, requires_grad=True)
    model._compute_flow_matching_loss = Mock(return_value=(flow_loss, torch.tensor([0.25])))
    packed = PackedSequence(
        label_ids=torch.tensor([1, 0]),
        ce_loss_weights=torch.ones(2),
        vision=ModalityData(
            mse_loss_indexes=torch.tensor([2]),
            tokens=[torch.zeros(1)],
            condition_mask=[torch.zeros(1, dtype=torch.bool)],
        ),
    )
    logits = torch.tensor([[0.0, 2.0], [3.0, 0.0]], requires_grad=True)

    total, losses = model._compute_losses(
        {"ce_preds": logits, "preds_vision": [torch.zeros(1)]},
        packed,
        gen_data_noised=SimpleNamespace(vt_target_vision=[torch.zeros(1)]),
        timesteps=torch.ones(1, 1),
        is_image_batch=False,
    )
    expected_ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    assert torch.allclose(losses["cross_entropy_loss_text"], expected_ce)
    assert torch.equal(losses["flow_matching_loss_vision"], flow_loss)
    assert torch.allclose(total, expected_ce * 2.0 + flow_loss * 10.0)


def test_pure_generation_model_step_has_zero_text_ce() -> None:
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.config = SimpleNamespace(
        predict_text_tokens=True,
        text_loss_weight=1.0,
        vision_gen=False,
        action_gen=False,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False,
            sample_level_loss_averaging=False,
        ),
    )
    model.tensor_kwargs_fp32 = {"device": torch.device("cpu"), "dtype": torch.float32}
    empty_logits = torch.empty(0, 2, requires_grad=True)
    total, losses = model._compute_losses(
        {"ce_preds": empty_logits},
        PackedSequence(label_ids=None, ce_loss_weights=None),
        gen_data_noised=SimpleNamespace(),
        timesteps=torch.empty(0),
        is_image_batch=False,
    )
    assert total.item() == 0.0
    assert losses["cross_entropy_loss_text"].item() == 0.0


@pytest.mark.parametrize("empty_action", [False, True])
def test_action128_model_losses_preserve_spatial_temporal_and_dimension_masks(empty_action):
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model.config = SimpleNamespace(
        predict_text_tokens=False,
        vision_gen=True,
        action_gen=True,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False,
            sample_level_loss_averaging=False,
            image_loss_scale=None,
            loss_scale=2.0,
            action_loss_weight=3.0,
        ),
    )
    flow = SimpleNamespace(train_time_weight=lambda ts, kwargs: torch.ones_like(ts))
    model.rectified_flow_video = model.rectified_flow_action = flow
    video = torch.ones(2, 3, 2, 2, requires_grad=True)
    action = torch.ones(5, 128, requires_grad=True)
    spatial_mask = torch.zeros(3, 2, 2, dtype=torch.bool)
    spatial_mask[:2, 0, 0] = True  # last frame and canvas padding are invalid
    action_mask = torch.zeros(5, 128, dtype=torch.bool)
    if not empty_action:
        action_mask[:3, :2] = True  # independent temporal length and native dimensions
    condition = torch.zeros(3, 1, 1)
    condition[0] = 1
    packed = PackedSequence(
        vision=ModalityData(
            tokens=[video],
            mse_loss_indexes=torch.tensor([1]),
            condition_mask=[condition],
            loss_valid_mask=[spatial_mask],
        ),
        action=ModalityData(
            tokens=[action],
            mse_loss_indexes=torch.arange(5),
            condition_mask=[torch.zeros(5, 1)],
            loss_valid_mask=[action_mask],
            raw_action_dim=[128],
        ),
    )
    loss, metrics = model._compute_losses(
        {"preds_vision": [video], "preds_action": [action]},
        packed,
        SimpleNamespace(vt_target_vision=[torch.zeros_like(video)], vt_target_action=[torch.zeros_like(action)]),
        torch.ones(1, 1),
        False,
        timesteps_action=torch.full((1, 1), 0.5),
    )
    assert metrics["flow_matching_loss_vision"].item() == 1
    assert metrics["flow_matching_loss_action"].item() == (0 if empty_action else 1)
    assert loss.item() == (2 if empty_action else 5)
    loss.backward()
    expected_video = torch.zeros_like(video, dtype=torch.bool)
    expected_video[:, 1, 0, 0] = True
    assert torch.equal(video.grad != 0, expected_video)
    assert torch.equal(action.grad != 0, action_mask)
