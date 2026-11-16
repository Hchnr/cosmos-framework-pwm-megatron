"""Load the complete trained HF export and run the original Action128 WAM sampler."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import json
import time
from pathlib import Path

import torch

from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.inference.action_serving import PreparedAction128Request, assemble_batch
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.utils import misc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    config = Cosmos3OmniConfig.from_pretrained(args.checkpoint)
    config.model["config"]["fixed_sequence_length"]["enabled"] = False
    config.model["config"]["action_tower_und_short_tokens"] = 0
    config.model["config"]["action_tower_init"] = "checkpoint"
    started = time.perf_counter()
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(
        args.checkpoint,
        config=config,
        parallelism_config=ParallelismConfig(data_parallel_shard_degree=1, data_parallel_replicate_degree=1),
    )
    model = hf_model.model.eval()
    load_seconds = time.perf_counter() - started
    spec = model.config.action128_spec
    if not isinstance(spec, dict):
        from omegaconf import OmegaConf

        spec = OmegaConf.to_container(spec, resolve=True)
    request = PreparedAction128Request(
        action_domain=spec["registry"]["domains"][0],
        task_title="export smoke",
        prompt="Move the robotic arm forward.",
        action_fps=30,
        duration_seconds=4 / 30,
        action_dim_mask=[True] * 6 + [False] * 122,
        num_steps=2,
        guidance=0,
        seed=42,
    )
    canvas = torch.zeros(3, 5, 256, 256, dtype=torch.uint8)
    # A nonconstant observation; one camera requires no spatial stitching seam.
    canvas[:, 0] = torch.arange(256, dtype=torch.uint8).view(1, 1, 256).expand(3, 256, 256)
    batch = assemble_batch(request, spec, canvas, None, ["main"], 30, model.input_video_key)
    batch = misc.to(batch, device="cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        samples = model.generate_samples_from_batch(batch, seed=[42], num_steps=2, guidance=0.0, shift=5.0)
        video = ((model.decode(samples["vision"][0]) + 1) / 2).clamp(0, 1).squeeze(0)
        assert list(video.shape) == [3, 5, 256, 256], video.shape
        assert torch.isfinite(video).all()
        physical = model.net.action_normalization.denormalize(samples["action"][0].float(), 0)
        transform = spec["transforms"][0]
        normalized_mask = torch.tensor(transform["normalization_mask"], device="cuda", dtype=torch.bool)
        expected = samples["action"][0].float().clone()
        scale = torch.tensor(transform["scale"], device="cuda")
        bias = torch.tensor(transform["bias"], device="cuda")
        expected[:, normalized_mask] = (expected[:, normalized_mask] - bias[normalized_mask]) / scale[normalized_mask]
        torch.testing.assert_close(physical, expected, rtol=2e-5, atol=2e-5)
        assert torch.isfinite(physical).all()
    from cosmos_framework.tools.visualize.video import save_img_or_video

    video_path = args.checkpoint.parent / "export_verify_consumption" / "vision"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    save_img_or_video(video, str(video_path), fps=30, quality=8)
    import cv2

    capture = cv2.VideoCapture(str(video_path.with_suffix(".mp4")))
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (256, 256, 3)
        decoded_frames += 1
    capture.release()
    assert decoded_frames == 5, decoded_frames
    torch.cuda.synchronize()
    action = samples["action"][0]
    assert list(action.shape) == [4, 128], action.shape
    assert torch.isfinite(action).all()
    assert not action[:, 6:].count_nonzero(), "Inactive Action128 slots must remain zero"
    result = dict(
        passed=True,
        checkpoint=str(args.checkpoint.resolve()),
        model_parameters=sum(p.numel() for p in model.net.parameters()),
        load_seconds=load_seconds,
        generation_seconds=time.perf_counter() - started,
        num_steps=2,
        seed=42,
        outputs={
            k: [list(v.shape) for v in values]
            for k, values in samples.items()
            if isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values)
        },
        video_shape=list(video.shape),
        video_finite=True,
        video_file=str(video_path.with_suffix(".mp4").resolve()),
        encoded_video_verified=dict(decoded_frames=decoded_frames, height=256, width=256),
        inverse_transform_verified=True,
        action_finite=True,
        inactive_slots_zero=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        exported_weight_bytes=json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["metadata"][
            "total_size"
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
