"""Prepared logical weights reuse the native video/action loss and masks."""

from types import SimpleNamespace

import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def test_mcore_schedule_scaling_does_not_modify_reported_loss():
    from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss

    from cosmos_framework.training_backends.megatron_tp_pp.preparation import GenerationLoss

    model = GenerationLoss.__new__(GenerationLoss)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(rectified_flow_training_config=SimpleNamespace(sample_level_loss_averaging=False))
    model._compute_losses = lambda **kwargs: (kwargs["out_net"].square().sum(), {})
    prediction = torch.tensor([2.0, 3.0], requires_grad=True)
    batch = {"packed_sequence": None, "loss_inputs": {}}
    records = []
    scaled, _ = forward_step_calc_loss(
        model,
        prediction,
        lambda output: model(output, batch),
        SimpleNamespace(timers=None, grad_scale_func=None, calculate_per_token_loss=False),
        None,
        False,
        12,
        records,
        cp_group_size=1,
        is_last_stage=True,
    )
    assert records[0]["loss"].item() == 13.0
    torch.testing.assert_close(scaled, torch.tensor(13.0 / 12))
    scaled.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([4.0, 6.0]) / 12)


def test_prepared_sample_weight_preserves_native_loss_and_gradients_without_collective():
    model = OmniMoTModel.__new__(OmniMoTModel)
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model.config = SimpleNamespace(
        predict_text_tokens=False,
        vision_gen=True,
        action_gen=True,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False,
            sample_level_loss_averaging=True,
            image_loss_scale=None,
            loss_scale=2.0,
            action_loss_weight=3.0,
        ),
    )
    model.rectified_flow_video = model.rectified_flow_action = SimpleNamespace(
        train_time_weight=lambda ts, kwargs: torch.ones_like(ts)
    )
    collective_calls = []

    def native_count_scale(**kwargs):
        collective_calls.append(kwargs)
        return torch.tensor(0.75, dtype=torch.float64)

    model._sample_level_loss_scale = native_count_scale

    def compute(override):
        video = torch.ones(2, 3, 2, 2, requires_grad=True)
        action = torch.ones(2, 128, requires_grad=True)
        video_mask = torch.zeros(3, 2, 2, dtype=torch.bool)
        video_mask[:2, 0, 0] = True
        action_mask = torch.zeros(2, 128, dtype=torch.bool)
        action_mask[:, :2] = True
        condition = torch.zeros(3, 1, 1)
        condition[0] = 1
        packed = PackedSequence(
            vision=ModalityData(
                tokens=[video],
                mse_loss_indexes=torch.tensor([1]),
                condition_mask=[condition],
                loss_valid_mask=[video_mask],
            ),
            action=ModalityData(
                tokens=[action],
                mse_loss_indexes=torch.arange(2),
                condition_mask=[torch.zeros(2, 1)],
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
            sample_level_scale_override=override,
        )
        return loss, torch.autograd.grad(loss, (video, action)), metrics

    expected, expected_grads, expected_metrics = compute(None)
    assert len(collective_calls) == 1
    actual, actual_grads, actual_metrics = compute(torch.tensor(0.75, dtype=torch.float64))
    assert len(collective_calls) == 1  # no physical WORLD/DP count reduction
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.item() == 3.75
    for grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)
    for key in expected_metrics:
        torch.testing.assert_close(actual_metrics[key], expected_metrics[key], rtol=0, atol=0)
