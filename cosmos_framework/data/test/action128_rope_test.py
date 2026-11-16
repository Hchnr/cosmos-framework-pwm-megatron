"""Verify Action128 seam coordinates from published reader through model binding."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tomllib
import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset, view_layout
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.spatial import action128_seam_position
from cosmos_framework.data.generator.sequence_packing.static_layout import (
    StaticPackingConfig,
    bind_static_packed_sequence,
)
from cosmos_framework.data.test.action128_test import media_stub as media_stub
from cosmos_framework.data.test.action128_test import source as source
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@pytest.fixture
def packed_canvas(source, media_stub, tmp_path, monkeypatch):
    # The shared media fixture normally shrinks the canvas. This audit needs
    # real production layouts; stub only video decoding and text tokenization.
    monkeypatch.setattr("cosmos_framework.data.generator.action.unified_action128.view_layout", view_layout)

    def build(view_count, aspect, *, parts=False):
        root, catalog, _ = source
        path = root / "meta/episodes/chunk-000/episodes.parquet"
        episodes = pq.read_table(path).to_pylist()
        roles = ["main", "left_wrist", "right_wrist"][:view_count]
        for episode in episodes:
            reference = episode["video_views"][0]
            episode["view_mapping"] = {role: role for role in roles}
            episode["video_views"] = [
                dict(reference, role=role, source_key=role, width=aspect[0], height=aspect[1]) for role in roles
            ]
        pq.write_table(pa.Table.from_pylist(episodes), path)
        monkeypatch.setenv("HF_MODEL_PATH", "/tmp/tokenizer")
        monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/converted-three-tower")
        monkeypatch.setenv("WAN_VAE_PATH", "/tmp/vae.pth")
        recipe_text = Path("examples/toml/sft_config/action128_wam_nano.toml").read_text()
        for dataset in tomllib.loads(recipe_text)["dataloader_train"]["datasets"]:
            recipe_text = recipe_text.replace(dataset["path"], str(catalog))
        recipe = tmp_path / "action128_rope.toml"
        recipe.write_text(recipe_text)
        config = load_experiment_from_toml(recipe)
        geometry = StaticPackingConfig(**config.dataloader_train.static_packing_config)
        dataset = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
        sample = dataset[0]
        _, boxes, width, height = view_layout(episodes[0])
        assert sample["video"].shape[-2:] == (height, width)
        assert len(sample["decode_metadata"]["selected_roles"]) == view_count
        caption = json.loads(sample["ai_caption"])
        if view_count > 1:
            assert caption["camera_layout"][0][3] == boxes[0][3]
        if parts:
            return sample, geometry
        packed = VFMListCollator(
            static_packing_config=geometry,
            special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
        ).collate([sample])["packed_sequence"]
        position_ids = packed.position_ids.clone()
        frames, grid_h, grid_w = packed.vision.token_shapes[0]
        # Same geometry as online VAE; pixel values cannot change spatial IDs.
        latent = torch.zeros(
            1, 48, frames, height // geometry.spatial_compression_factor, width // geometry.spatial_compression_factor
        )
        bound = bind_static_packed_sequence(
            packed,
            SimpleNamespace(
                x0_tokens_vision=[latent],
                spatial_seams_vision=[
                    action128_seam_position(
                        sample["decode_metadata"],
                        (height, width),
                        geometry.spatial_compression_factor * geometry.latent_patch_size,
                    )
                ],
                x0_tokens_action=[sample["action"]],
                action_domain_id=[sample["domain_id"]],
                raw_action_dim=[128],
                is_image_batch=False,
                num_vision_items_per_sample=None,
                num_views_per_vision_item=None,
                control_weights=None,
            ),
            torch.tensor([0.5]),
            latent_patch_size=geometry.latent_patch_size,
            vision_spatial_position_mode="action128_seam",
        )
        torch.testing.assert_close(bound.position_ids, position_ids)
        ids = bound.position_ids[:, bound.vision.sequence_indexes].reshape(3, frames, grid_h, grid_w)
        return ids, boxes[0][3], geometry.spatial_compression_factor * geometry.latent_patch_size

    return build


@pytest.mark.parametrize("aspect", [(16, 9), (3, 2), (4, 3)])
def test_single_camera_keeps_image_origin(packed_canvas, aspect):
    ids, _, _ = packed_canvas(1, aspect)
    assert ids[1, 0, 0, 0] == ids[2, 0, 0, 0] == 0
    assert torch.all(torch.diff(ids[1, 0, :, 0]) > 0)
    assert torch.all(torch.diff(ids[2, 0, 0, :]) > 0)


@pytest.mark.parametrize("view_count", [2, 3])
@pytest.mark.parametrize("aspect", [(16, 9), (3, 2), (4, 3)])
def test_multiview_origin_at_seam_with_positive_above_negative_below(packed_canvas, view_count, aspect):
    ids, seam_pixels, pixel_stride = packed_canvas(view_count, aspect)
    h_ids = ids[1, 0, :, 0]
    row_starts = torch.arange(h_ids.numel()) * pixel_stride
    # Exclude cells touching/crossing the seam; both corner- and center-based
    # discretizations must satisfy the signs strictly inside each camera region.
    above = row_starts + pixel_stride < seam_pixels
    below = row_starts > seam_pixels
    assert above.any() and below.any()
    checks = {
        "upper_positive": bool(torch.all(h_ids[above] > 0)),
        "lower_negative": bool(torch.all(h_ids[below] < 0)),
        "height_decreases_downward": bool(torch.all(torch.diff(h_ids) < 0)),
    }
    assert all(checks.values()), {
        "checks": checks,
        "seam_pixels": seam_pixels,
        "pixel_stride": pixel_stride,
        "height_ids": h_ids.tolist(),
    }

    # Exact patch-corner coordinates, including fractional seams and padding rows.
    torch.testing.assert_close(h_ids, seam_pixels / pixel_stride - torch.arange(h_ids.numel()).float())
    torch.testing.assert_close(ids[2, 0, 0], torch.arange(ids.shape[-1]).float())
    for frame in range(ids.shape[1]):
        torch.testing.assert_close(ids[1:, frame], ids[1:, 0])


def _mixed_canvas(packed_canvas, *, zero_margin=False):
    samples = [packed_canvas(n, (16, 9), parts=True)[0] for n in (1, 2, 3)]
    _, geometry = packed_canvas(3, (16, 9), parts=True)
    if zero_margin:
        geometry = replace(geometry, unified_3d_mrope_temporal_modality_margin=0)
    special = {"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4}
    batch = VFMListCollator(static_packing_config=geometry, special_tokens=special).collate(samples)
    packed = batch["packed_sequence"]
    clean = GenerationDataClean(
        batch_size=3,
        is_image_batch=False,
        x0_tokens_vision=[
            torch.empty(1, 0, shape[0], sample["video"].shape[-2] // 16, sample["video"].shape[-1] // 16)
            for sample, shape in zip(samples, packed.vision.token_shapes, strict=True)
        ],
        raw_state_vision=[s["video"].unsqueeze(0) for s in samples],
        x0_tokens_action=[s["action"] for s in samples],
        fps_vision=torch.tensor([float(s["conditioning_fps"]) for s in samples]),
        fps_action=torch.tensor([float(s["conditioning_fps_action"]) for s in samples]),
        action_domain_id=[s["domain_id"] for s in samples],
        raw_action_dim=[128] * 3,
    )
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    model = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(spatial_compression_factor=16),
            diffusion_expert_config=SimpleNamespace(vision_spatial_position_mode="action128_seam", patch_spatial=2),
        ),
        input_video_latent_key="video_latent",
    )
    clean.spatial_seams_vision = OmniMoTModel._get_spatial_seams_vision(model, batch, clean.raw_state_vision)
    return samples, geometry, special, batch, clean


@pytest.mark.parametrize("zero_margin", [False, True])
def test_mixed_views_static_runtime_binding_and_other_axes_unchanged(packed_canvas, zero_margin):
    samples, geometry, special, batch, clean = _mixed_canvas(packed_canvas, zero_margin=zero_margin)
    packed = batch["packed_sequence"]
    assert clean.spatial_seams_vision == [None, 11.25, 13.34375]
    legacy = VFMListCollator(
        static_packing_config=replace(geometry, vision_spatial_position_mode="canvas"), special_tokens=special
    ).collate(samples)["packed_sequence"]
    # Include subsequent samples and action/text spatial axes, not just vision T.
    torch.testing.assert_close(packed.position_ids[0], legacy.position_ids[0].float())
    torch.testing.assert_close(packed.position_ids[2], legacy.position_ids[2].float())
    nonvision = torch.ones(packed.sequence_length, dtype=torch.bool)
    nonvision[packed.vision.sequence_indexes] = False
    torch.testing.assert_close(packed.position_ids[:, nonvision], legacy.position_ids[:, nonvision].float())
    opts = asdict(geometry)
    opts.pop("spatial_compression_factor")
    runtime = pack_input_sequence(
        sequence_plans=[s["sequence_plan"] for s in samples],
        input_text_indexes=[s["text_token_ids"].tolist() for s in samples],
        gen_data_clean=clean,
        input_timesteps=torch.zeros(3),
        special_tokens=special,
        **opts,
    )
    torch.testing.assert_close(runtime.position_ids, packed.position_ids)
    bound = bind_static_packed_sequence(
        packed, clean, torch.ones(3), latent_patch_size=2, vision_spatial_position_mode="action128_seam"
    )
    torch.testing.assert_close(bound.position_ids, runtime.position_ids)
    with pytest.raises(ValueError, match="mode differs"):
        bind_static_packed_sequence(
            legacy, clean, torch.ones(3), latent_patch_size=2, vision_spatial_position_mode="action128_seam"
        )
    clean.spatial_seams_vision[1] += 0.5
    with pytest.raises(ValueError, match="seams differ"):
        bind_static_packed_sequence(packed, clean, torch.ones(3), latent_patch_size=2)


def test_sampling_slice_preserves_spatial_seams(packed_canvas):
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    *_, clean = _mixed_canvas(packed_canvas)
    subset = OmniMoTModel._slice_gen_data_clean(None, clean, 1, 3)
    assert subset.spatial_seams_vision == [11.25, 13.34375]


@pytest.mark.parametrize("fault", ["missing", "roles", "locations", "width", "height", "float", "single"])
def test_invalid_layout_rejected(packed_canvas, fault):
    sample, _ = packed_canvas(3, (16, 9), parts=True)
    metadata = copy.deepcopy(sample["decode_metadata"])
    if fault == "missing":
        metadata.pop("camera_layout")
    elif fault == "roles":
        metadata["selected_roles"].pop()
    elif fault == "locations":
        metadata["camera_layout"][1][1] = "top"
    elif fault == "width":
        metadata["camera_layout"][1][2] += 1
    elif fault == "height":
        metadata["camera_layout"][2][3] += 1
    elif fault == "float":
        metadata["camera_layout"][0][3] = 427.5
    else:
        metadata["selected_roles"] = ["main"]
    with pytest.raises(ValueError):
        action128_seam_position(metadata, tuple(sample["video"].shape[-2:]), 32)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(vision_spatial_position_mode="unknown"),
        dict(vision_spatial_position_mode="action128_seam", unified_3d_mrope_reset_spatial_ids=False),
        dict(vision_spatial_position_mode="action128_seam", video_temporal_causal=True),
    ],
)
def test_invalid_spatial_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        StaticPackingConfig(**kwargs)


def test_fractional_seam_without_fps_preserves_following_text_cursor():
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan

    clean = GenerationDataClean(
        batch_size=1, is_image_batch=False, x0_tokens_vision=[torch.empty(1, 0, 1, 30, 2)], spatial_seams_vision=[15.25]
    )
    options = dict(
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        input_text_indexes=[[1]],
        gen_data_clean=clean,
        input_timesteps=torch.zeros(1),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
        include_end_of_generation_token=True,
    )
    old = pack_input_sequence(**options)
    new = pack_input_sequence(**options, vision_spatial_position_mode="action128_seam")
    assert new.position_ids.dtype == torch.float32
    torch.testing.assert_close(new.position_ids[:, new.text_indexes], old.position_ids[:, old.text_indexes].float())
    torch.testing.assert_close(new.position_ids[0], old.position_ids[0].float())
    assert new.position_ids[1, new.vision.sequence_indexes[0]] == 15.25
    assert new.position_ids[1, new.vision.sequence_indexes[-1]] == -13.75
