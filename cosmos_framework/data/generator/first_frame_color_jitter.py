"""Training-only color jitter for the first RGB conditioning frame.

Ranges and operation order follow the ordinary augmentation in NVlabs/cosmos-policy,
commit 18a2accadf4e7a3531e56754102af5a24d2316da, datasets/dataset_utils.py.
"""

import hashlib
import json

import torch
from torchvision.transforms import functional as F

from cosmos_framework.data.generator.dataflow.base import RawItemProcessor


class FirstFrameColorJitterProcessor(RawItemProcessor):
    """Augment decoded conditioning pixels without changing sample ordering."""

    def __init__(self, processor: RawItemProcessor, *, seed: int = 42):
        self.processor = processor
        self.seed = seed

    def process(self, item: dict) -> dict:
        sample = self.processor.process(item)
        if sample is None:
            return sample
        plan = sample.get("sequence_plan")
        conditions = (
            plan.get("condition_frame_indexes_vision", [])
            if isinstance(plan, dict)
            else getattr(plan, "condition_frame_indexes_vision", [])
        )
        video = sample.get("video")
        # Offline latents have no RGB pixels to augment. Pure T2V has no
        # conditioning frame, and Understanding media uses a different path.
        if 0 not in (conditions or []) or video is None:
            return sample
        if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("First-frame color jitter requires RGB video [3,T,H,W]")
        if video.dtype != torch.uint8 or video.shape[1] < 1:
            raise ValueError("First-frame color jitter requires nonempty uint8 video")
        identity = json.dumps(
            [
                "first-frame-color-jitter-v1",
                self.seed,
                str(sample.get("__url__", "")),
                str(sample.get("__key__", item.get("id", ""))),
                int(item.get("_dp_epoch", 0)),
            ],
            separators=(",", ":"),
        )
        seed = int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little")
        generator = torch.Generator().manual_seed(seed)
        factors = torch.rand(4, generator=generator).tolist()
        frame = F.adjust_brightness(video[:, 0], 0.8 + factors[0] * 0.4)
        frame = F.adjust_contrast(frame, 0.8 + factors[1] * 0.4)
        frame = F.adjust_saturation(frame, 0.8 + factors[2] * 0.4)
        frame = F.adjust_hue(frame, -0.05 + factors[3] * 0.1)
        layout = (sample.get("decode_metadata") or {}).get("camera_layout")
        if isinstance(layout, list) and len(layout) in (2, 3):
            if layout[0][1] == "left":
                boundary = int(layout[0][2])
                frame[:, :, boundary - 2 : boundary + 2] = 0
            else:
                boundary = int(layout[0][3])
                frame[:, boundary - 2 : boundary + 2] = 0
                if len(layout) == 3:
                    width = frame.shape[-1]
                    frame[:, boundary:, width // 2 - 2 : width // 2 + 2] = 0
        result = dict(sample)
        result["video"] = video.clone()
        result["video"][:, 0] = frame
        return result
