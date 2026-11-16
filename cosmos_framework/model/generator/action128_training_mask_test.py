"""Regressions for State conditioning through collating, binding and training noising."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.sequence_packing.static_layout import (
    StaticPackingConfig,
    bind_static_packed_sequence,
)
from cosmos_framework.data.test.action128_test import media_stub as media_stub
from cosmos_framework.data.test.action128_test import source as source
from cosmos_framework.data.test.pretrain_state_test import state_source as state_source
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def _training_input(sample, device):
    model = OmniMoTModel.__new__(OmniMoTModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_gen=True)
    model.parallel_dims = None
    model.tensor_kwargs_fp32 = {"device": device, "dtype": torch.float32}
    model.tensor_kwargs = {"device": device, "dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    model.rectified_flow_video = model.rectified_flow_action = RectifiedFlow(velocity_field=lambda *args: None)
    collator = VFMListCollator(
        static_packing_config=StaticPackingConfig(enable_fps_modulation=True, action_dim=128),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
    )
    packed = collator.collate([sample])["packed_sequence"]
    frames, height, width = packed.vision.token_shapes[0]
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, frames, height * 2, width * 2, device=device)],
        x0_tokens_action=[sample["action"].to(device).clone()],
        action_domain_id=[sample["domain_id"].to(device)],
        raw_action_dim=[sample["raw_action_dim"]],
    )
    sigma = torch.tensor([[0.5]], device=device)
    bind_static_packed_sequence(packed, clean, sigma, latent_patch_size=2)
    if device == "cuda":
        packed.to_cuda()
    noised = model._add_noise_to_input(clean, packed, sigma, sigmas_action=sigma)
    model._replace_clean_with_noised(packed, noised)
    return model, packed, clean, noised, sigma


@pytest.mark.parametrize("state_value", [0.25, 0.75])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpus(1))])
def test_training_preserves_state_and_unlabelled_inputs(state_source, media_stub, state_value, device):
    root, _ = state_source
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    sample = dataset[1]  # Includes a padded last Action row.
    sample["action"][0, 0] = state_value
    sample["action_loss_mask"][1] = False  # Valid input, but not a supervised target.
    model, packed, clean, noised, sigma = _training_input(sample, device)
    action = packed.action.tokens[0]

    # Check the payload actually read by the network, after noise and replacement.
    torch.testing.assert_close(action[0], clean.x0_tokens_action[0][0].to(action.dtype))
    assert action[0, 0].item() == state_value
    input_mask = sample["action_dim_mask"].to(device)
    assert not action[~input_mask].any()
    assert not input_mask[-1].any()
    assert packed.action.input_valid_mask[0].device.type == device
    assert packed.action.robot_state_token_indexes.tolist() == [0]
    assert packed.action.domain_id[0].tolist() == sample["domain_id"].tolist()
    torch.testing.assert_close(
        action[1][input_mask[1]],
        ((clean.x0_tokens_action[0][1] + noised.epsilon_action[0][1]) * 0.5).to(action.dtype)[input_mask[1]],
    )
    # Loss eligibility must not alter the flow target either.
    torch.testing.assert_close(noised.vt_target_action[0], noised.epsilon_action[0] - clean.x0_tokens_action[0])

    loss_mask = packed.action.loss_valid_mask[0]
    pred = noised.vt_target_action[0].clone()
    pred[~loss_mask] += 100  # State, unlabelled targets and padding do not contribute.

    def loss():
        return compute_flow_matching_loss(
            pred=[pred],
            target=noised.vt_target_action,
            condition_mask=packed.action.condition_mask,
            timesteps=sigma,
            has_valid_tokens=True,
            rectified_flow=model.rectified_flow_action,
            tensor_kwargs_fp32=model.tensor_kwargs_fp32,
            loss_valid_mask=packed.action.loss_valid_mask,
        )[0]

    assert loss().item() == 0
    pred[loss_mask] += 2
    torch.testing.assert_close(loss(), torch.tensor(4.0, device=device))


def test_training_without_slot_mask_keeps_legacy_prefix(state_source, media_stub):
    root, _ = state_source
    sample = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)[1]
    for key in ("action_dim_mask", "action_loss_mask", "vision_loss_mask"):
        sample.pop(key)
    sample["raw_action_dim"] = 2
    _, packed, clean, noised, _ = _training_input(sample, "cpu")
    torch.testing.assert_close(
        packed.action.tokens[0][:, :2], (clean.x0_tokens_action[0][:, :2] + noised.epsilon_action[0][:, :2]) * 0.5
    )
    assert not packed.action.tokens[0][:, 2:].any()
