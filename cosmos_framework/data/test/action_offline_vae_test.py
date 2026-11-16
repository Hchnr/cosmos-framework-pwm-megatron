"""Offline action keeps dynamic caption RNG and the online action/plan contract."""

import copy
import json
import random

import pytest
import torch
from safetensors.torch import save_file

from cosmos_framework.data.generator.action.offline_vae import OfflineActionDataset
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.offline_vae import SafetensorsLatentReader


@pytest.mark.level(0)
def test_cached_action_preserves_dynamic_caption_and_action_contract(tmp_path):
    transform_config = dict(
        max_action_dim=64,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    tensors = {
        "0.action": torch.arange(32 * 8).view(32, 8).float(),
        "0.image_size": torch.tensor([256, 320, 256, 303]),
        "0.conditioning_fps": torch.tensor(15),
        "0.domain_id": torch.tensor(0),
        "0.latent": torch.zeros(48, 9, 16, 18, dtype=torch.bfloat16),
    }
    save_file(tensors, str(tmp_path / "data.safetensors"))
    metadata = dict(mode="wam", viewpoint="concat_view", ai_caption="a", _offline_caption_choices=["a", "b", "c"])
    record = dict(
        shard="data.safetensors",
        key="0",
        source_index=17,
        metadata=metadata,
        tensor_keys=[k[2:] for k in tensors if k != "0.latent"],
        video_shape=[3, 33, 256, 320],
        latent_shape=[48, 9, 16, 18],
    )
    manifest = dict(
        format="cosmos_action_latents_v1",
        complete=True,
        blocks=[[0, 1]],
        records=[record],
        resolution="256",
        transform=transform_config,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    dataset = OfflineActionDataset(path)
    transform = ActionTransformPipeline(**transform_config)
    observed = set()
    for seed in range(8):
        random.seed(seed)
        expected = copy.deepcopy(metadata)
        expected["ai_caption"] = random.choice(expected.pop("_offline_caption_choices"))
        expected.update({k[2:]: v.clone() for k, v in tensors.items() if k != "0.latent"})
        expected["video"] = torch.zeros((), dtype=torch.uint8).expand(record["video_shape"])
        expected["_offline_action_media"] = True
        expected = transform(expected, "256")
        state = random.getstate()
        random.seed(seed)
        actual = dataset[0]
        assert random.getstate() == state
        assert actual["ai_caption"] == expected["ai_caption"]
        observed.add(actual["ai_caption"])
        torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
        assert actual["sequence_plan"] == expected["sequence_plan"]
        assert "video" not in actual
    assert len(observed) > 1
    batch = VFMListCollator(latent_reader=SafetensorsLatentReader()).collate([dataset[0]])
    assert "video_latent" in batch and "action" in batch
    assert dataset.get_shuffle_blocks() == [(0, 1)]


@pytest.mark.level(0)
def test_incomplete_manifest_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(format="cosmos_action_latents_v1", complete=False)))
    with pytest.raises(ValueError, match="complete"):
        OfflineActionDataset(path)
