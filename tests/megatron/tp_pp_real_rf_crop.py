# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small real RF replay preserving native masks, positions, domains and targets."""

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence


def original_rf_inputs(path):
    """Retain every original sample/token/target, casting floating inputs to FP32."""
    path = Path(path)

    def fp32(value):
        if isinstance(value, torch.Tensor):
            return value.float() if value.is_floating_point() else value
        if is_dataclass(value):
            result = copy.copy(value)
            for field in fields(value):
                object.__setattr__(result, field.name, fp32(getattr(value, field.name)))
            if isinstance(result, PackedSequence):
                # torch.load maps cached CUDA tensors to CPU, but the cache's
                # torch.device descriptor can still name CUDA. Rebuild derived
                # packing metadata at the actual execution device.
                result._sequence_pack_metadata = None
            return result
        if isinstance(value, dict):
            return {key: fp32(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(fp32(child) for child in value)
        return value

    return fp32(
        {
            "packed_sequence": torch.load(path, map_location="cpu", weights_only=False),
            "loss_inputs": torch.load(path.with_suffix(".loss.pt"), map_location="cpu", weights_only=False),
        }
    )


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        result = copy.copy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, move_batch(getattr(value, field.name), device))
        return result
    if isinstance(value, dict):
        return {key: move_batch(child, device) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_batch(child, device) for child in value)
    return value


def cropped_times(timesteps, noisy):
    values = []
    for sample, indexes in enumerate(noisy):
        row = timesteps[sample].reshape(-1)
        values.append(row.expand(indexes.numel()) if row.numel() == 1 else row[indexes])
    return torch.cat(values).float()


def real_rf_crop(path):
    path = Path(path)
    source = torch.load(path, map_location="cpu", weights_only=False)
    loss_inputs = torch.load(path.with_suffix(".loss.pt"), map_location="cpu", weights_only=False)
    if source.vision is None or source.action is None or len(source.vision.tokens) < 2:
        raise ValueError("RF math replay requires two native video/action samples")
    for modality in (source.vision, source.action):
        if modality.sample_indexes[:2] != [0, 1]:
            raise ValueError("RF crop requires one native item for each of its first two samples")
    videos = [tensor[:, :, :2, :2, :2].float().clone() for tensor in source.vision.tokens[:2]]
    actions = [tensor[:2].float().clone() for tensor in source.action.tokens[:2]]
    if any(tensor.shape != (1, 48, 2, 2, 2) for tensor in videos) or any(
        tensor.shape != (2, 128) for tensor in actions
    ):
        raise ValueError("Native RF crop payload shapes do not match Nano/Action128")
    vnoisy = [indexes[indexes < 2].clone() for indexes in source.vision.noisy_frame_indexes[:2]]
    anoisy = [indexes[indexes < 2].clone() for indexes in source.action.noisy_frame_indexes[:2]]
    vtimes = loss_inputs["timesteps"][:2].clone()
    if vtimes.shape[-1] > 1:
        vtimes = vtimes[:, :2]
    atimes = loss_inputs["timesteps_action"]
    if atimes is not None:
        atimes = atimes[:2].clone()
        if atimes.shape[-1] > 1:
            atimes = atimes[:, :2]
    vision = ModalityData(
        tokens=videos,
        token_shapes=[(2, 1, 1)] * 2,
        sample_indexes=[0, 1],
        sequence_indexes=torch.tensor([2, 3, 8, 9]),
        mse_loss_indexes=torch.cat([indexes + 2 + sample * 6 for sample, indexes in enumerate(vnoisy)]),
        timesteps=cropped_times(vtimes, vnoisy),
        noisy_frame_indexes=vnoisy,
        condition_mask=[tensor[:2].clone() for tensor in source.vision.condition_mask[:2]],
        loss_valid_mask=None
        if source.vision.loss_valid_mask is None
        else [tensor[:2, :2, :2].clone() for tensor in source.vision.loss_valid_mask[:2]],
    )
    robot_states = []
    offset = 0
    for sample in range(2):
        for index in range(2):
            if bool((source.action.robot_state_token_indexes == offset + index).any()):
                robot_states.append(sample * 2 + index)
        offset += source.action.token_shapes[sample][0]
    action = ModalityData(
        tokens=actions,
        token_shapes=[(2,)] * 2,
        sample_indexes=[0, 1],
        sequence_indexes=torch.tensor([4, 5, 10, 11]),
        mse_loss_indexes=torch.cat([indexes + 4 + sample * 6 for sample, indexes in enumerate(anoisy)]),
        robot_state_token_indexes=torch.tensor(robot_states, dtype=torch.long),
        domain_id=[tensor[:2].clone() for tensor in source.action.domain_id[:2]],
        timesteps=cropped_times(vtimes if atimes is None else atimes, anoisy),
        noisy_frame_indexes=anoisy,
        condition_mask=[tensor[:2].clone() for tensor in source.action.condition_mask[:2]],
        raw_action_dim=None if source.action.raw_action_dim is None else source.action.raw_action_dim[:2],
        loss_valid_mask=None
        if source.action.loss_valid_mask is None
        else [tensor[:2].clone() for tensor in source.action.loss_valid_mask[:2]],
        input_valid_mask=None
        if source.action.input_valid_mask is None
        else [tensor[:2].clone() for tensor in source.action.input_valid_mask[:2]],
    )
    texts, positions = [], []
    sample_offset = vision_offset = action_offset = 0
    for sample in range(2):
        indexes = torch.where(
            (source.text_indexes >= sample_offset) & (source.text_indexes < sample_offset + source.sample_lens[sample])
        )[0][:2]
        if indexes.numel() != 2:
            raise ValueError("Each RF crop sample must retain two real text tokens")
        texts.append(source.text_ids[indexes])
        spatial = source.vision.token_shapes[sample][1] * source.vision.token_shapes[sample][2]
        selected = torch.cat(
            (
                source.text_indexes[indexes],
                source.vision.sequence_indexes[vision_offset + torch.tensor([0, spatial])],
                source.action.sequence_indexes[action_offset : action_offset + 2],
            )
        )
        positions.append(source.position_ids[:, selected])
        sample_offset += source.sample_lens[sample]
        vision_offset += spatial * source.vision.token_shapes[sample][0]
        action_offset += source.action.token_shapes[sample][0]
    sequence = PackedSequence(
        sample_lens=[6, 6],
        split_lens=[2, 4] * 2,
        attn_modes=["causal", "full"] * 2,
        sequence_length=12,
        text_ids=torch.cat(texts),
        text_indexes=torch.tensor([0, 1, 6, 7]),
        position_ids=torch.cat(positions, dim=1).clone(),
        vision=vision,
        action=action,
        uses_single_timestep=source.uses_single_timestep,
        vision_spatial_position_mode=source.vision_spatial_position_mode,
    )
    noised = copy.copy(loss_inputs["gen_data_noised"])
    noised.batch_size = 2
    for name in ("epsilon_vision", "xt_tokens_vision", "vt_target_vision"):
        setattr(noised, name, [tensor[:, :, :2, :2, :2].clone() for tensor in getattr(noised, name)[:2]])
    for name in ("epsilon_action", "xt_tokens_action", "vt_target_action"):
        setattr(noised, name, [tensor[:2].clone() for tensor in getattr(noised, name)[:2]])
    if noised.sigmas_vision is not None:
        noised.sigmas_vision = [tensor[:2].clone() for tensor in noised.sigmas_vision[:2]]
    if noised.sigmas_action is not None:
        noised.sigmas_action = [tensor[:2].clone() for tensor in noised.sigmas_action[:2]]
    if noised.raw_action_dim is not None:
        noised.raw_action_dim = noised.raw_action_dim[:2]
    return {
        "packed_sequence": sequence,
        "sample_count": 2,
        "loss_inputs": {**loss_inputs, "gen_data_noised": noised, "timesteps": vtimes, "timesteps_action": atimes},
    }


def full_text_rf_crop(path):
    """Keep complete native text conditioning while cropping only video/action."""
    batch = real_rf_crop(path)
    source = torch.load(path, map_location="cpu", weights_only=False)
    old = batch["packed_sequence"]
    texts, text_indexes, positions, lengths = [], [], [], []
    vision_indexes, action_indexes, vision_losses, action_losses = [], [], [], []
    original_offset = new_offset = vision_offset = action_offset = 0
    for sample in range(2):
        selected_text = torch.where(
            (source.text_indexes >= original_offset)
            & (source.text_indexes < original_offset + source.sample_lens[sample])
        )[0]
        count = selected_text.numel()
        if count < 2:
            raise ValueError("Native sample must include text conditioning")
        texts.append(source.text_ids[selected_text])
        text_indexes.append(torch.arange(new_offset, new_offset + count))
        v_indexes = torch.arange(new_offset + count, new_offset + count + 2)
        a_indexes = v_indexes + 2
        vision_indexes.append(v_indexes)
        action_indexes.append(a_indexes)
        vision_losses.append(v_indexes[old.vision.noisy_frame_indexes[sample]])
        action_losses.append(a_indexes[old.action.noisy_frame_indexes[sample]])
        spatial = source.vision.token_shapes[sample][1] * source.vision.token_shapes[sample][2]
        original_indexes = torch.cat(
            (
                source.text_indexes[selected_text],
                source.vision.sequence_indexes[vision_offset + torch.tensor([0, spatial])],
                source.action.sequence_indexes[action_offset : action_offset + 2],
            )
        )
        positions.append(source.position_ids[:, original_indexes])
        lengths.append(count + 4)
        original_offset += source.sample_lens[sample]
        new_offset += count + 4
        vision_offset += spatial * source.vision.token_shapes[sample][0]
        action_offset += source.action.token_shapes[sample][0]
    old.vision.sequence_indexes = torch.cat(vision_indexes)
    old.vision.mse_loss_indexes = torch.cat(vision_losses)
    old.action.sequence_indexes = torch.cat(action_indexes)
    old.action.mse_loss_indexes = torch.cat(action_losses)
    batch["packed_sequence"] = PackedSequence(
        sample_lens=lengths,
        split_lens=[value for length in lengths for value in (length - 4, 4)],
        attn_modes=["causal", "full"] * 2,
        sequence_length=sum(lengths),
        text_ids=torch.cat(texts),
        text_indexes=torch.cat(text_indexes),
        position_ids=torch.cat(positions, dim=1).clone(),
        vision=old.vision,
        action=old.action,
        uses_single_timestep=source.uses_single_timestep,
        vision_spatial_position_mode=source.vision_spatial_position_mode,
    )
    return batch
