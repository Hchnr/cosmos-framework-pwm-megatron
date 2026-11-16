"""Condition-only photometric augmentation and deterministic replay."""

from unittest.mock import patch

import pytest
import torch

from cosmos_framework.data.generator.dataflow.processors import IdentityProcessor
from cosmos_framework.data.generator.first_frame_color_jitter import FirstFrameColorJitterProcessor
from cosmos_framework.data.generator.sequence_packing import SequencePlan


def sample(epoch=0, *, conditioned=True):
    generator = torch.Generator().manual_seed(7)
    return {
        "__key__": "episode1/window10",
        "__url__": "/dataset",
        "_dp_epoch": epoch,
        "video": torch.randint(0, 256, (3, 17, 32, 48), dtype=torch.uint8, generator=generator),
        "action": torch.arange(17 * 128).reshape(17, 128).float(),
        "sequence_plan": SequencePlan(
            has_text=True, has_vision=True, condition_frame_indexes_vision=[0] if conditioned else []
        ),
    }


def test_only_conditioning_first_frame_changes():
    source = sample()
    original = source["video"].clone()
    result = FirstFrameColorJitterProcessor(IdentityProcessor()).process(source)
    assert not torch.equal(result["video"][:, 0], original[:, 0])
    torch.testing.assert_close(result["video"][:, 1:], original[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(source["video"], original, atol=0, rtol=0)
    assert result["action"] is source["action"]
    assert result["sequence_plan"] is source["sequence_plan"]
    assert result["video"].dtype == torch.uint8
    assert result["video"].shape == original.shape


def test_replay_is_independent_of_global_rng_and_changes_between_epochs():
    processor = FirstFrameColorJitterProcessor(IdentityProcessor(), seed=42)
    rng = torch.random.get_rng_state()
    first = processor.process(sample())
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.rand(100)
    replay = processor.process(sample())
    torch.testing.assert_close(first["video"], replay["video"], rtol=0, atol=0)
    assert not torch.equal(first["video"][:, 0], processor.process(sample(1))["video"][:, 0])


def test_default_parameters_follow_ordinary_reference_order():
    source = sample()
    from cosmos_framework.data.generator import first_frame_color_jitter as module

    with (
        patch.object(module.F, "adjust_brightness", wraps=module.F.adjust_brightness) as brightness,
        patch.object(module.F, "adjust_contrast", wraps=module.F.adjust_contrast) as contrast,
        patch.object(module.F, "adjust_saturation", wraps=module.F.adjust_saturation) as saturation,
        patch.object(module.F, "adjust_hue", wraps=module.F.adjust_hue) as hue,
    ):
        FirstFrameColorJitterProcessor(IdentityProcessor()).process(source)
        for transform in (brightness, contrast, saturation):
            assert 0.8 <= transform.call_args.args[1] <= 1.2
        assert -0.05 <= hue.call_args.args[1] <= 0.05


@pytest.mark.parametrize("views", [2, 3])
def test_camera_seams_remain_black(views):
    source = sample()
    source["video"][:, :, 14:18] = 0
    if views == 3:
        source["video"][:, :, 16:, 22:26] = 0
    source["decode_metadata"] = {"camera_layout": [["main", "top", 48, 16]] * views}
    result = FirstFrameColorJitterProcessor(IdentityProcessor()).process(source)
    assert not result["video"][:, 0, 14:18].any()
    if views == 3:
        assert not result["video"][:, 0, 16:, 22:26].any()


def test_unconditioned_and_offline_inputs_are_unchanged():
    processor = FirstFrameColorJitterProcessor(IdentityProcessor())
    source = sample(conditioned=False)
    assert processor.process(source) is source
    source = sample()
    source.pop("video")
    source["video_latent_ref"] = object()
    assert processor.process(source) is source


def test_side_by_side_camera_seam_remains_black():
    source = sample()
    source["video"][:, :, :, 22:26] = 0
    source["decode_metadata"] = {"camera_layout": [["auxiliary_0", "left", 24, 32], ["left_wrist", "right", 24, 32]]}
    result = FirstFrameColorJitterProcessor(IdentityProcessor()).process(source)
    assert not result["video"][:, 0, :, 22:26].any()
    torch.testing.assert_close(result["video"][:, 1:], source["video"][:, 1:], atol=0, rtol=0)
