import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.pretrain_state import current_state, load_state_transforms
from cosmos_framework.data.generator.action.unified_action128 import UnifiedAction128Dataset
from cosmos_framework.data.generator.catalog import CatalogSchemaError
from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.sequence_packing.static_layout import StaticPackingConfig
from cosmos_framework.data.test.action128_test import media_stub as media_stub
from cosmos_framework.data.test.action128_test import source as source


@pytest.fixture
def state_source(source):
    root, catalog, _ = source
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["state_normalization"] = {
        "path": "meta/state_normalization.json",
        "field": "observation.state",
        "mask": "observation.state_mask",
        "authoritative_key": "action_domain",
    }
    path.write_text(json.dumps(info))
    doc = json.loads((root / "meta/action_normalization.json").read_text())
    doc["application"]["field"] = "observation.state"
    next(iter(doc["domains"].values()))["transform"]["scale"][0] = 0.25
    (root / "meta/state_normalization.json").write_text(json.dumps(doc))
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    values = np.zeros((len(table), 128), dtype=np.float32)
    values[:, 0] = np.arange(len(table))
    masks = np.zeros_like(values, dtype=bool)
    masks[:, 0] = True
    table = table.append_column("observation.state", pa.array(values.tolist(), type=pa.list_(pa.float32(), 128)))
    table = table.append_column("observation.state_mask", pa.array(masks.tolist(), type=pa.list_(pa.bool_(), 128)))
    pq.write_table(table, path, row_group_size=3)
    return root, catalog


def test_state_is_clean_and_uses_current_timestamp(state_source, media_stub):
    root, catalog = state_source
    dataset = build_action_dataset_from_catalog(catalog, tokenizer_config=object(), action_use_state=True)
    dataset.decoder = media_stub
    sample = dataset[1]
    plain = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)[1]
    assert sample["action"].shape == (5, 128)
    assert sample["domain_id"].tolist() == [8] * 5
    assert plain["domain_id"].tolist() == [8]
    assert sample["action"][0, 0] == 1  # state scaling, not action's 0.5
    torch.testing.assert_close(sample["action"][1:], plain["action"])
    torch.testing.assert_close(sample["action_loss_mask"][1:], plain["action_loss_mask"])
    assert not sample["action_loss_mask"][0].any()
    assert not sample["action_control_mask"][0].any()
    assert sample["action_timestamps"][:2].tolist() == [0, 0]
    collator = VFMListCollator(
        static_packing_config=StaticPackingConfig(enable_fps_modulation=True),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
    )
    packed = collator.collate([sample, dataset[0]])["packed_sequence"]
    assert packed.action.robot_state_token_indexes.tolist() == [0, 5]
    assert torch.cat(packed.action.condition_mask).flatten()[[0, 5]].tolist() == [1, 1]
    times = packed.position_ids[0, packed.action.sequence_indexes]
    assert times[0] == times[1]
    assert times[5] == times[6]
    assert len(packed.action.mse_loss_indexes) == 8
    assert dataset.fingerprint != UnifiedAction128Dataset(root).fingerprint


def test_sample_timing_preserves_state_action_video_and_identity(state_source, media_stub, monkeypatch):
    root, _ = state_source
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    monkeypatch.delenv("COSMOS_DATALOADER_TIMING", raising=False)
    expected = dataset[1]
    monkeypatch.setenv("COSMOS_DATALOADER_TIMING", "1")
    actual = dataset[1]
    assert actual["__key__"] == expected["__key__"]
    assert actual["ai_caption"] == expected["ai_caption"]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    timing = actual["decode_metadata"].pop("timing_seconds")
    assert all(value >= 0 for value in timing.values())
    assert timing["sample"] >= timing["numeric_read"] + timing["video"] + timing["caption_read"]
    for view in actual["decode_metadata"]["views"]:
        assert view.pop("timing_seconds")["canvas_copy"] >= 0
    assert actual["decode_metadata"] == expected["decode_metadata"]


@pytest.mark.parametrize("window,masked_row", [(0, 0), (1, 4)])
def test_masked_state_omitted_without_changing_targets(state_source, media_stub, window, masked_row):
    root, _ = state_source
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    masks = table["observation.state_mask"].to_pylist()
    masks[masked_row] = [False] * 128  # Never borrow a previous/future valid observation.
    table = table.set_column(
        table.schema.get_field_index("observation.state_mask"),
        "observation.state_mask",
        pa.array(masks, type=pa.list_(pa.bool_(), 128)),
    )
    pq.write_table(table, path)
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    sample = dataset[window]
    plain = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)[window]
    assert sample["sequence_plan"].robot_state_frame_indexes == []
    assert sample["sequence_plan"].condition_frame_indexes_action == []
    for key in ("action", "action_loss_mask", "action_dim_mask", "action_timestamps", "video"):
        torch.testing.assert_close(sample[key], plain[key])
    collator = VFMListCollator(
        static_packing_config=StaticPackingConfig(enable_fps_modulation=True),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
    )
    other = dataset[1 - window]
    packed = collator.collate([sample, other])["packed_sequence"]
    indexes = packed.action.robot_state_token_indexes
    assert indexes.tolist() == [len(sample["action"])]
    assert not torch.cat(packed.action.loss_valid_mask)[indexes].any()


def test_eef_state_routes_independently_through_packing(state_source, media_stub):
    root, _ = state_source
    path = root / "meta/action_normalization.json"
    doc = json.loads(path.read_text())
    next(iter(doc["domains"].values()))["action_space"] = "eef20"
    path.write_text(json.dumps(doc))
    dataset = UnifiedAction128Dataset(root, use_state=True, tokenizer_config=object(), video_decoder=media_stub)
    sample = dataset[0]
    assert sample["domain_id"].tolist() == [32, 0, 0, 0, 0]
    collator = VFMListCollator(
        static_packing_config=StaticPackingConfig(enable_fps_modulation=True),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
    )
    packed = collator.collate([sample, dataset[1]])["packed_sequence"]
    assert [ids.tolist() for ids in packed.action.domain_id] == [[32, 0, 0, 0, 0]] * 2
    assert not torch.cat(packed.action.loss_valid_mask)[[0, 5]].any()


@pytest.mark.parametrize("fps", [15, 20, 25, 30, 50])
def test_single_causal_observation(fps):
    t = np.arange(fps * 3 + 1, dtype=np.float32) / fps
    values = np.zeros((len(t), 128), np.float32)
    values[:, 0] = t
    mask = np.zeros_like(values, bool)
    mask[:, 0] = True
    table = pa.table({"timestamp": t, "observation.state": values.tolist(), "observation.state_mask": mask.tolist()})
    transform = {"normalization_mask": [False] * 128, "scale": [1.0] * 128, "bias": [0.0] * 128}
    state, dims, times = current_state(table, start=2.01, transform=transform)
    assert state.shape == (1, 128)
    assert state[0, 0] == 2
    assert not dims[0, 1:].any()
    assert times[0] == pytest.approx(-0.01)


def test_missing_contract_is_error(source):
    root, _, _ = source
    with pytest.raises(CatalogSchemaError, match="State128"):
        UnifiedAction128Dataset(root, use_state=True)


def test_explicit_normalization_column_declaration(state_source):
    root, _ = state_source
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info.pop("state_storage", None)
    path.write_text(json.dumps(info))
    assert load_state_transforms(root)["domains"]
    del info["state_normalization"]["mask"]
    path.write_text(json.dumps(info))
    with pytest.raises(CatalogSchemaError, match="storage contract"):
        load_state_transforms(root)


def test_conflicting_storage_declaration_rejected(state_source):
    root, _ = state_source
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["state_storage"] = {"value": "action", "loader_mask": "action_dim_mask"}
    path.write_text(json.dumps(info))
    with pytest.raises(CatalogSchemaError, match="storage contract"):
        load_state_transforms(root)


def test_normalization_changes_fingerprint(state_source):
    root, _ = state_source
    before = UnifiedAction128Dataset(root, use_state=True).fingerprint
    path = root / "meta/state_normalization.json"
    doc = json.loads(path.read_text())
    next(iter(doc["domains"].values()))["transform"]["scale"][0] *= 2
    path.write_text(json.dumps(doc))
    assert UnifiedAction128Dataset(root, use_state=True).fingerprint != before


def test_explicit_state_snapshot_restores_values_and_fingerprint(state_source, media_stub, tmp_path):
    root, catalog = state_source
    before = build_action_dataset_from_catalog(catalog, tokenizer_config=object(), action_use_state=True)
    snapshot = tmp_path / "old_state_normalization.json"
    snapshot.write_text((root / "meta/state_normalization.json").read_text())
    doc = json.loads(snapshot.read_text())
    next(iter(doc["domains"].values()))["transform"]["scale"][0] = 0.1
    (root / "meta/state_normalization.json").write_text(json.dumps(doc))
    changed = build_action_dataset_from_catalog(catalog, tokenizer_config=object(), action_use_state=True)
    restored = build_action_dataset_from_catalog(
        catalog,
        tokenizer_config=object(),
        action_use_state=True,
        action_state_normalization_paths={str(root): str(snapshot)},
    )
    assert changed.fingerprint != before.fingerprint
    assert restored.fingerprint == before.fingerprint
    before.decoder = changed.decoder = restored.decoder = media_stub
    torch.testing.assert_close(restored[1]["action"], before[1]["action"])
    assert changed[1]["action"][0, 0] != before[1]["action"][0, 0]
    # The explicit path still validates the published State contract.
    doc["application"]["field"] = "action"
    snapshot.write_text(json.dumps(doc))
    with pytest.raises(CatalogSchemaError):
        load_state_transforms(root, normalization_path=snapshot)
