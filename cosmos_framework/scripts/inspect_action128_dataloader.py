"""CPU-only TOML -> real Action128 dataloader acceptance and inspectable samples."""

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import torch
from PIL import Image

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.projection_routing import projection_head_id
from cosmos_framework.utils.lazy_config import instantiate


def unwrap(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", default="examples/toml/sft_config/action128_wam_nano.toml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=14)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_experiment_from_toml(args.toml)
    config.dataloader_train.num_workers = args.num_workers
    config.dataloader_train.pin_memory = False
    loader = instantiate(config.dataloader_train)
    processor = instantiate(config.dataloader_train.processor)
    tokenizer = getattr(processor, "tokenizer", processor)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spec = config.model.config.action128_spec
    records = []
    iterator = iter(loader)
    for step in range(args.batches):
        batch = next(iterator)
        key = unwrap(batch["__key__"])
        caption = json.loads(unwrap(batch["ai_caption"]))
        dataset = loader._loaders[loader._names.index(batch["dataset_name"])].dataset._distributor._dataset
        episode_index = int(key.rsplit("/", 2)[1])
        episodes = dataset.episodes.filter(pc.equal(dataset.episodes["episode_index"], episode_index))
        assert episodes.num_rows == 1
        episode = episodes.to_pylist()[0]
        source_caption = dataset.reader.caption(dataset.root / episode["caption_path"], episode["caption_index"])
        assert caption["command"] == source_caption["source_task_description"]
        assert caption["task_title"] == source_caption["task_title"]
        assert caption["embodiment"] == episode["embodiment_name"]
        assert caption["action_frequency_hz"] == episode["fps"]
        assert caption["action_frequency_hz"] == float(unwrap(batch["conditioning_fps_action"]).item())
        assert caption["fps"] == float(unwrap(batch["conditioning_fps"]).item())
        ids = unwrap(batch["text_token_ids"])
        decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        decoded_caption = decoded.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        assert json.loads(decoded_caption) == caption
        video = unwrap(batch["video"])
        action = unwrap(batch["action"])
        assert action.ndim == 2 and action.shape[-1] == 128
        assert torch.isfinite(action).all()
        dim_mask = unwrap(batch["action_dim_mask"]).reshape_as(action).bool()
        control = unwrap(batch["action_control_mask"]).reshape_as(action).bool()
        temporal = unwrap(batch["temporal_valid_mask"]).flatten().bool()
        valid = unwrap(batch["action_valid"]).flatten().bool()
        loss_mask = unwrap(batch["action_loss_mask"]).reshape_as(action).bool()
        assert not (control & ~dim_mask).any()
        assert torch.equal(loss_mask, temporal[:, None] & valid[:, None] & dim_mask)
        assert not action[~dim_mask].any()
        normalization_id = int(unwrap(batch["normalization_domain_id"]).item())
        head_id = int(unwrap(batch["domain_id"]).item())
        domain = spec.registry.domains[normalization_id]
        assert head_id == spec.registry.domain_projection_ids[normalization_id]
        assert head_id == projection_head_id(domain.split("/")[1], caption["embodiment"])
        normalized = torch.tensor(list(spec.transforms[normalization_id].normalization_mask), dtype=torch.bool)
        if normalized.any():
            assert action[:, normalized].abs().max() <= 1
        _, frames, height, width = video.shape
        assert frames % 4 == 1
        assert caption["resolution"] == {"W": width, "H": height}
        layout = caption.get("camera_layout")
        if layout:
            assert isinstance(layout, list)
            boundary = layout[0][3]
            assert not video[:, :, boundary - 2 : boundary + 2].any()
            if len(layout) == 3:
                assert not video[:, :, boundary:, width // 2 - 2 : width // 2 + 2].any()
        packed = batch["packed_sequence"]
        action_positions = packed.position_ids[0, packed.action.sequence_indexes]
        action_times = unwrap(batch["action_timestamps"]).flatten().to(action_positions)
        video_start = packed.position_ids[0, packed.vision.sequence_indexes[0]]
        geometry = config.model.config.diffusion_expert_config
        # Wan VAE temporal compression is 4; action retains native timestamps.
        torch.testing.assert_close(action_positions, action_times * (float(geometry.base_fps) / 4) + video_start)
        assert int(packed.action.domain_id[0].item()) == head_id
        assert torch.equal(packed.action.loss_valid_mask[0], loss_mask)
        assert not unwrap(batch["sequence_plan"]).supervise_text
        filename = f"{step:02d}_{key.replace('/', '_')}"
        image_path = output / f"{filename}.png"
        Image.fromarray(video[:, 0].permute(1, 2, 0).numpy()).save(image_path)
        record = {
            "sample_key": key,
            "dataset": batch["dataset_name"],
            "caption": caption,
            "command_source": "source_task_description",
            "detokenized_text": decoded,
            "video_shape_cthw": list(video.shape),
            "action_shape": list(action.shape),
            "action_domain": domain,
            "normalization_domain_id": normalization_id,
            "projection_head_id": head_id,
            "valid_action_targets": int(loss_mask.sum()),
            "first_action": action[0].tolist(),
            "first_action_dim_mask": dim_mask[0].tolist(),
            "first_frame": str(image_path.resolve()),
        }
        (output / f"{filename}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        records.append(record)
        print(f"PASS {key} video={tuple(video.shape)} action={tuple(action.shape)} head={head_id}", flush=True)
        del batch, video, action, packed
    assert len({record["dataset"] for record in records}) == len(config.catalog_datasets)
    summary = {"passed_batches": len(records), "datasets": len(config.catalog_datasets), "records": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"PASS {len(records)} batches, all {len(config.catalog_datasets)} datasets; artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
