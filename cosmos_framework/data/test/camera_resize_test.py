"""Preserve RGB array layouts and pixels at the camera resize boundary."""

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.camera_resize import copy_resized_camera


@pytest.mark.parametrize("layout", ["vertical_flip", "horizontal_flip", "transpose", "readonly"])
def test_camera_resize_accepts_numpy_views_without_mutating_inputs(layout):
    rng = np.random.default_rng(704)
    frames = [rng.integers(0, 256, (32, 48, 3), dtype=np.uint8) for _ in range(3)]
    if layout == "vertical_flip":
        frames = [frame[::-1] for frame in frames]
    elif layout == "horizontal_flip":
        frames = [frame[:, ::-1] for frame in frames]
    elif layout == "transpose":
        frames = [frame.transpose(1, 0, 2) for frame in frames]
    else:
        for frame in frames:
            frame.flags.writeable = False
    before = [frame.copy() for frame in frames]
    height, width = 17, 29
    source_height, source_width = frames[0].shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_height, resized_width = math.ceil(source_height * scale), math.ceil(source_width * scale)
    # Independent legacy full-window reference also normalizes array layouts.
    pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    resized = F.interpolate(pixels, (resized_height, resized_width), mode="bilinear", align_corners=False)
    resized = resized.round().clamp(0, 255).byte()
    y, x = (resized_height - height) // 2, (resized_width - width) // 2
    canvas = torch.full((3, len(frames), height + 4, width + 6), 17, dtype=torch.uint8)
    expected = canvas.clone()
    expected[:, :, 2 : 2 + height, 3 : 3 + width] = resized[:, :, y : y + height, x : x + width].permute(1, 0, 2, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        copy_resized_camera(frames, canvas[:, :, 2 : 2 + height, 3 : 3 + width])
    assert torch.equal(canvas, expected)
    for frame, original in zip(frames, before, strict=True):
        np.testing.assert_array_equal(frame, original)
