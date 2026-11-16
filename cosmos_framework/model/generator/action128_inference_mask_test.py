"""CPU regressions for sparse Action128 inference, through real model methods."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan, prepare_runtime_sequence
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

ACTION_STEPS = 3
ACTION_DIM = 128


def _sparse_mask(*slots):
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    mask[list(slots)] = True
    return mask


def _batch(mask=None, *, raw_dim=ACTION_DIM, nested=False):
    batch = {
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                supervise_text=False,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=True,
            )
        ],
        "action": [[torch.zeros(ACTION_STEPS, ACTION_DIM)]],
        "raw_action_dim": [raw_dim],
    }
    if mask is not None:
        batch["action_dim_mask"] = [[mask]] if nested else [mask]
    return batch


def _action(flat):
    return flat[-ACTION_STEPS * ACTION_DIM :].reshape(ACTION_STEPS, ACTION_DIM)


@pytest.fixture
def cpu_model(monkeypatch):
    model = OmniMoTModel.__new__(OmniMoTModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_gen=True, sound_gen=False)
    model.input_video_key, model.input_image_key = "video", "images"
    model.parallel_dims = None
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model.tensor_kwargs = dict(model.tensor_kwargs_fp32)
    model.llm_special_tokens = {"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}
    model._get_inference_text_tokens = lambda *args, **kwargs: ([[1, 2]], [[]])
    model._derive_include_end_of_generation_token = lambda: False
    model._input_sequence_options = lambda: {
        "special_tokens": model.llm_special_tokens,
        "latent_patch_size": 1,
        "enable_fps_modulation": True,
        "temporal_compression_factor": 4,
        "action_dim": ACTION_DIM,
    }

    def clean_data(batch, **kwargs):
        action = batch["action"][0][0].clone()
        return GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[torch.zeros(1, 1, 2, 1, 1)],
            fps_vision=torch.tensor([30.0]),
            x0_tokens_action=[action],
            raw_action_dim=list(batch["raw_action_dim"]),
            fps_action=torch.tensor([30.0]),
            action_domain_id=[torch.tensor([0])],
        )

    model.get_data_and_condition = clean_data
    model.network_action_inputs = []

    def denoise(*, data_batch_packed, **kwargs):
        actions = data_batch_packed.action.tokens
        model.network_action_inputs.append([action.clone() for action in actions])
        return {
            "preds_vision": [torch.ones_like(item) for item in data_batch_packed.vision.tokens],
            "preds_action": [2 * action + 0.125 for action in actions],
        }

    model.denoise = denoise
    # Keep the real CPU packing/topology, replacing only the device transfer.
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    return model


def test_initial_noise_masks_sparse_slots_without_changing_valid_noise(cpu_model):
    mask = _sparse_mask(0, 3, 9, 66, 127)
    reference = cpu_model._prepare_inference_data(_batch(), [42])
    prepared = cpu_model._prepare_inference_data(_batch(mask, nested=True), [42])
    noise, clean_reference, condition = map(_action, (prepared[4][0], prepared[5][0], prepared[6][0]))
    expanded = mask.expand_as(noise)
    assert noise[~expanded].count_nonzero() == 0
    torch.testing.assert_close(noise[expanded], _action(reference[4][0])[expanded], rtol=0, atol=0)
    assert clean_reference[~expanded].count_nonzero() == 0
    assert condition[~expanded].eq(1).all()
    assert condition[expanded].eq(0).all()


@pytest.mark.parametrize("seed", [0, 42, 12345678, 2**32 - 1])
def test_inference_noise_modalities_are_distinct_and_repeatable(cpu_model, seed):
    prepared = cpu_model._prepare_inference_data(_batch(), [seed])
    repeated = cpu_model._prepare_inference_data(_batch(), [seed])
    flat = prepared[4][0]
    vision_size = prepared[1].x0_tokens_vision[0].numel()
    action = _action(flat).reshape(-1)
    free_vision = ~prepared[6][0][:vision_size].bool()
    # An independent NumPy stream freezes the existing video-seed contract.
    expected_vision = torch.from_numpy(np.random.RandomState(seed).standard_normal(vision_size).astype(np.float32))
    torch.testing.assert_close(flat[:vision_size][free_vision], expected_vision[free_vision], rtol=0, atol=0)
    assert free_vision.any()
    assert not torch.equal(flat[:vision_size][free_vision], action[:vision_size][free_vision])
    torch.testing.assert_close(flat, repeated[4][0], rtol=0, atol=0)


def test_temporal_slot_mask_does_not_use_training_loss_mask(cpu_model):
    mask = torch.stack([_sparse_mask(0, 127), _sparse_mask(3, 66), _sparse_mask(9)])
    batch = _batch(mask)
    batch["action_loss_mask"] = [torch.zeros_like(mask)]
    prepared = cpu_model._prepare_inference_data(batch, [123])
    unmasked = cpu_model._prepare_inference_data(_batch(), [123])
    noise = _action(prepared[4][0])
    assert noise[~mask].count_nonzero() == 0
    torch.testing.assert_close(noise[mask], _action(unmasked[4][0])[mask], rtol=0, atol=0)
    assert noise[mask].count_nonzero() == mask.sum()


def test_training_collator_temporal_slot_mask_is_accepted(cpu_model):
    mask = torch.stack([_sparse_mask(0, 127), _sparse_mask(3, 66), _sparse_mask(9)])
    raw_batch = _batch(mask)
    batch = VFMListCollator().collate(
        [
            {
                "sequence_plan": raw_batch["sequence_plan"][0],
                "action": raw_batch["action"][0][0],
                "raw_action_dim": ACTION_DIM,
                "action_dim_mask": mask,
            }
        ]
    )
    assert batch["action_dim_mask"][0].shape == (1, ACTION_STEPS, ACTION_DIM)
    prepared = cpu_model._prepare_inference_data(batch, [42])
    reference = cpu_model._prepare_inference_data(_batch(mask), [42])
    torch.testing.assert_close(prepared[4][0], reference[4][0], rtol=0, atol=0)
    velocity = _action(_velocity(cpu_model, prepared, prepared[4])[0])
    assert velocity[~mask].count_nonzero() == 0


def _velocity(model, prepared, state, *, template=None):
    return model._get_velocity(
        noise_x=state,
        timestep=torch.tensor([[500.0]]),
        text_tokens=prepared[2],
        sequence_plans=prepared[0],
        gen_data_clean=prepared[1],
        has_noisy_actions=prepared[7],
        packed_sequence_template=template,
    )


@pytest.mark.parametrize("cached_template", [False, True])
def test_velocity_masks_network_input_and_output_and_keeps_valid_values(cpu_model, cached_template):
    mask = _sparse_mask(0, 3, 9, 66, 127).expand(ACTION_STEPS, -1)
    prepared = cpu_model._prepare_inference_data(_batch(mask), [42])
    corrupted_state = [prepared[4][0].clone()]
    action = _action(corrupted_state[0])
    action.copy_(torch.arange(action.numel()).reshape_as(action) / 100)
    action[~mask] = 99
    expected_valid = 2 * action[mask] + 0.125
    before = corrupted_state[0].clone()
    template = None
    if cached_template:
        template = prepare_runtime_sequence(
            sequence_plans=prepared[0],
            input_text_indexes=prepared[2],
            gen_data_clean=prepared[1],
            input_timesteps=torch.tensor([0.0]),
            **cpu_model._input_sequence_options(),
        )
    velocity = _action(_velocity(cpu_model, prepared, corrupted_state, template=template)[0])
    network_input = cpu_model.network_action_inputs[-1][0]
    assert network_input[~mask].count_nonzero() == 0
    torch.testing.assert_close(network_input[mask], action[mask], rtol=0, atol=0)
    assert velocity[~mask].count_nonzero() == 0
    torch.testing.assert_close(velocity[mask], expected_valid, rtol=0, atol=0)
    # Do not mutate a caller's sampler state while enforcing the model-input contract.
    torch.testing.assert_close(corrupted_state[0], before, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["unipc", "fixed_step"])
def test_real_multistep_sampler_keeps_invalid_slots_zero(cpu_model, kind):
    mask = _sparse_mask(0, 3, 9, 66, 127).expand(ACTION_STEPS, -1)

    def sample(batch):
        prepared = cpu_model._prepare_inference_data(batch, [19])
        observed_states = []

        def velocity_fn(state, timestep):
            observed_states.append(_action(state[0]).clone())
            return cpu_model._get_velocity(
                noise_x=state,
                timestep=timestep,
                text_tokens=prepared[2],
                sequence_plans=prepared[0],
                gen_data_clean=prepared[1],
                has_noisy_actions=prepared[7],
            )

        kwargs = {}
        if kind == "fixed_step":
            sampler = FixedStepSampler([1.0, 0.7, 0.3, 0.0])
            kwargs = {"condition_reference": prepared[5], "condition_mask": prepared[6]}
        else:
            sampler = UniPCSampler(UniPCSamplerConfig(), {"device": "cpu", "dtype": torch.float32})
        result = sampler(velocity_fn, prepared[4], seed=[23], num_steps=3, shift=1.0, **kwargs)
        return _action(result[0]), observed_states

    baseline, _ = sample(_batch())
    cpu_model.network_action_inputs.clear()
    masked, observed_states = sample(_batch(mask))
    assert len(observed_states) == 3
    # FixedStep generates fresh full-width noise at intermediate steps. Check
    # the actual states seen by the next step, not only the final output.
    for state in observed_states:
        assert state[~mask].count_nonzero() == 0
    for network_inputs in cpu_model.network_action_inputs:
        assert network_inputs[0][~mask].count_nonzero() == 0
    assert masked[~mask].count_nonzero() == 0
    torch.testing.assert_close(masked[mask], baseline[mask], rtol=0, atol=0)


def test_request_masks_do_not_leak_between_requests(cpu_model):
    first_mask, second_mask = _sparse_mask(0, 3), _sparse_mask(66, 127)
    first = cpu_model._prepare_inference_data(_batch(first_mask), [42])
    second = cpu_model._prepare_inference_data(_batch(second_mask), [42])
    repeated = cpu_model._prepare_inference_data(_batch(first_mask), [42])
    first_noise, second_noise = _action(first[4][0]), _action(second[4][0])
    assert first_noise[:, first_mask].count_nonzero() == ACTION_STEPS * 2
    assert first_noise[:, second_mask].count_nonzero() == 0
    assert second_noise[:, second_mask].count_nonzero() == ACTION_STEPS * 2
    assert second_noise[:, first_mask].count_nonzero() == 0
    torch.testing.assert_close(first[4][0], repeated[4][0], rtol=0, atol=0)
    # Neither preparation rewrites a previously prepared request's metadata.
    torch.testing.assert_close(first[1].action_dim_mask[0], repeated[1].action_dim_mask[0])


def test_without_slot_mask_legacy_prefix_masking_is_preserved(cpu_model):
    prepared = cpu_model._prepare_inference_data(_batch(raw_dim=7), [42])
    noise = _action(prepared[4][0])
    assert noise[:, :7].count_nonzero() == ACTION_STEPS * 7
    assert noise[:, 7:].count_nonzero() == 0
    velocity = _action(_velocity(cpu_model, prepared, [prepared[4][0].clone()])[0])
    assert velocity[:, 7:].count_nonzero() == 0
    torch.testing.assert_close(velocity[:, :7], 2 * noise[:, :7] + 0.125, rtol=0, atol=0)


def test_slot_mask_intersects_legacy_prefix(cpu_model):
    prepared = cpu_model._prepare_inference_data(_batch(_sparse_mask(0, 9, 127), raw_dim=7), [42])
    noise = _action(prepared[4][0])
    velocity = _action(_velocity(cpu_model, prepared, prepared[4])[0])
    assert noise[:, 0].count_nonzero() == ACTION_STEPS
    assert noise[:, 1:].count_nonzero() == 0
    assert velocity[:, 1:].count_nonzero() == 0
    torch.testing.assert_close(velocity[:, 0], 2 * noise[:, 0] + 0.125, rtol=0, atol=0)


def test_condition_only_action_is_masked_without_becoming_a_denoising_target(cpu_model):
    mask = _sparse_mask(0, 9, 127).expand(ACTION_STEPS, -1)
    batch = _batch(mask)
    batch["sequence_plan"][0].condition_frame_indexes_action = list(range(ACTION_STEPS))
    batch["action"][0][0].fill_(3)
    prepared = cpu_model._prepare_inference_data(batch, [42])
    assert prepared[7] is False
    assert prepared[4][0].numel() == 2  # Only vision enters the sampler state.
    clean_action = prepared[1].x0_tokens_action[0]
    assert clean_action[~mask].count_nonzero() == 0
    assert clean_action[mask].eq(3).all()
    velocity = _velocity(cpu_model, prepared, prepared[4])[0]
    assert velocity.numel() == 2
    network_input = cpu_model.network_action_inputs[-1][0]
    assert network_input[~mask].count_nonzero() == 0
    assert network_input[mask].eq(3).all()


@pytest.mark.parametrize("misplaced", [False, True])
def test_slot_masks_align_with_action_samples_in_mixed_batch(cpu_model, misplaced):
    mask = _sparse_mask(0, 9, 127)
    action = torch.zeros(ACTION_STEPS, ACTION_DIM)
    video_plan = SequencePlan(has_text=True, supervise_text=False, has_vision=True)
    action_plan = _batch()["sequence_plan"][0]
    batch = {
        "sequence_plan": [video_plan, action_plan],
        "action_dim_mask": [mask, None] if misplaced else [None, [mask]],
    }
    cpu_model._get_inference_text_tokens = lambda *args, **kwargs: ([[1, 2], [1, 2]], [[], []])
    cpu_model.get_data_and_condition = lambda *args, **kwargs: GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, 2, 1, 1), torch.zeros(1, 1, 2, 1, 1)],
        fps_vision=torch.tensor([30.0, 30.0]),
        x0_tokens_action=[action.clone()],
        raw_action_dim=[ACTION_DIM],
        fps_action=torch.tensor([30.0]),
        action_domain_id=[torch.tensor([0])],
    )
    if misplaced:
        with pytest.raises(ValueError, match="align with action-bearing"):
            cpu_model._prepare_inference_data(batch, [42, 43])
    else:
        prepared = cpu_model._prepare_inference_data(batch, [42, 43])
        assert len(prepared[1].action_dim_mask) == 1
        torch.testing.assert_close(prepared[1].action_dim_mask[0], mask)
        assert prepared[4][0].numel() == 2
        assert _action(prepared[4][1])[:, ~mask].count_nonzero() == 0


@pytest.mark.parametrize(
    "invalid_mask",
    [
        torch.ones(128),
        torch.ones(127, dtype=torch.bool),
        torch.ones(2, 128, dtype=torch.bool),
        torch.ones(2, ACTION_STEPS, 128, dtype=torch.bool),
        torch.ones(1, 1, ACTION_STEPS, 128, dtype=torch.bool),
    ],
)
def test_invalid_slot_mask_is_rejected(cpu_model, invalid_mask):
    with pytest.raises((ValueError, TypeError)):
        cpu_model._prepare_inference_data(_batch(invalid_mask), [42])


def test_slice_clean_data_preserves_request_slot_masks(cpu_model):
    masks = [_sparse_mask(0, 3), _sparse_mask(66, 127)]
    clean = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, 2, 1, 1), torch.zeros(1, 1, 2, 1, 1)],
        x0_tokens_action=[torch.zeros(ACTION_STEPS, ACTION_DIM), torch.zeros(ACTION_STEPS, ACTION_DIM)],
        action_dim_mask=masks,
    )
    sliced = cpu_model._slice_gen_data_clean(clean, 1, 2)
    assert len(sliced.action_dim_mask) == 1
    torch.testing.assert_close(sliced.action_dim_mask[0], masks[1])


def test_generate_remasks_hook_velocity_before_cfg_normalization(cpu_model):
    mask = _sparse_mask(0, 3, 9, 66, 127).expand(ACTION_STEPS, -1)
    batch = _batch(mask)
    batch["action_processing_record"] = [ActionProcessingRecord(ACTION_DIM, None)]
    cpu_model.config.rectified_flow_inference_config = SimpleNamespace(scheduler_type="unipc")
    cpu_model._can_reuse_inference_pack_templates = lambda *args, **kwargs: False
    cpu_model._can_reuse_inference_text_kv = lambda *args, **kwargs: False
    hook_states = []

    def hook_builder(**kwargs):
        def corrupt_absent_slots(velocity, state, timestep):
            hook_states.append(_action(state[0]).clone())
            result = [item.clone() for item in velocity]
            _action(result[0])[~mask] = 999
            return result

        return corrupt_absent_slots

    def generate(builder):
        return cpu_model.generate_samples_from_batch(
            batch,
            sampler=UniPCSampler(UniPCSamplerConfig(), {"device": "cpu", "dtype": torch.float32}),
            seed=[42],
            num_steps=3,
            shift=1.0,
            guidance=2.0,
            normalize_cfg=True,
            velocity_postprocess_builder=builder,
        )

    reference = generate(None)
    with_hook = generate(hook_builder)
    assert len(hook_states) == 3
    for state in hook_states:
        assert state[~mask].count_nonzero() == 0
    assert with_hook["action"][0][~mask].count_nonzero() == 0
    # Invalid hook output must not affect the norm used to rescale valid action
    # or vision velocities. This runs the actual CFG closure and real sampler.
    for modality in ("vision", "action"):
        torch.testing.assert_close(with_hook[modality][0], reference[modality][0], rtol=0, atol=0)
