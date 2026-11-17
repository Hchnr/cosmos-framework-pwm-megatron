"""Action128 source values, windows, masks, caption and fresh-head contracts."""

import json
import math
import multiprocessing as mp
import pickle
import shutil
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tomllib
import torch

from cosmos_framework.data.generator.action.unified_action128 import (
    BoundedRows,
    UnifiedAction128Dataset,
    _copy_resized_camera,
    build_action128_spec,
    build_registry,
    render_caption,
    view_layout,
)
from cosmos_framework.data.generator.catalog import CatalogSchemaError
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_CONTEXT_KEY, DROP_SAMPLE_KEY
from cosmos_framework.data.generator.sequence_packing.static_layout import StaticPackingConfig
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.mot.action128 import Action128DomainLinear, Action128Normalization


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "robot"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data").mkdir()
    domain = "franka_panda__parallel_gripper/joint128/observed_absolute_state/test_v1"
    (root / "meta/domains.json").write_text(json.dumps({"domains": {domain: 99}}))
    transform = {
        "kind": ["robust_quantile"] + ["inactive"] * 127,
        "normalization_mask": [True] + [False] * 127,
        "scale": [0.5] + [1.0] * 127,
        "bias": [0.0] * 128,
        "clip_min": -1.0,
        "clip_max": 1.0,
    }
    normalization = {
        "schema_version": "action-normalization-v1",
        "policy_version": "domain-slot-q01-q99-v1",
        "dataset_name": "robot",
        "action_storage": "raw_physical_or_source_units",
        "fit_scope": {
            "data_role": "training",
            "training_eligible": True,
            "row_mask": "action_valid",
            "slot_mask": "action_dim_mask",
            "control_mask_used_for_fit": False,
        },
        "application": {
            "performed_by": "training dataloader",
            "parquet_values_pre_scaled": False,
            "do_not_apply_additional_mean_std": True,
            "authoritative_key": "action_domain",
        },
        "domains": {
            domain: {
                "action_space": "joint128",
                "slot_count": 128,
                "statistics": {
                    key: [0] * 128 for key in ("count", "min", "max", "mean", "std", "q01", "q99", "sampled")
                },
                "transform": transform,
            }
        },
    }
    (root / "meta/action_normalization.json").write_text(json.dumps(normalization))
    (root / "meta/a1_structural_gate.json").write_text(
        json.dumps(
            {
                "schema_version": "a1-structural-action128-v1",
                "dataset_name": "robot",
                "status": "passed",
                "failures": {},
            }
        )
    )
    normalization_reference = {
        "path": "meta/action_normalization.json",
        "schema_version": "action-normalization-v1",
        "policy_version": "domain-slot-q01-q99-v1",
        "storage": "raw_action_plus_dataloader_transform",
        "authoritative_key": "action_domain",
    }
    gate_reference = {"path": "meta/a1_structural_gate.json", "status": "passed"}
    (root / "meta/info.json").write_text(
        json.dumps({"action_normalization": normalization_reference, "a1_structural_gate": gate_reference})
    )
    (root / "manifest.json").write_text(
        json.dumps({"action_normalization": normalization_reference, "a1_structural_gate": gate_reference})
    )
    n, fps = 7, 2.0
    action = np.zeros((n, 128), dtype=np.float32)
    action[:, 0] = np.arange(n) - 3
    action[:, 14] = 0.5
    dim = np.zeros((n, 128), dtype=bool)
    dim[:, [0, 14]] = True
    control = dim.copy()
    control[1] = False
    table = pa.table(
        {
            "index": np.arange(n),
            "episode_index": np.zeros(n, dtype=np.int64),
            "timestamp": np.arange(n) / fps,
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 128)),
            "action_dim_mask": pa.array(dim.tolist(), type=pa.list_(pa.bool_(), 128)),
            "action_control_mask": pa.array(control.tolist(), type=pa.list_(pa.bool_(), 128)),
            "action_valid": [True, True, False, True, True, True, True],
        }
    )
    pq.write_table(table, root / "data/frames.parquet", row_group_size=3)
    ep = dict(
        episode_index=0,
        length=n,
        duration_seconds=n / fps,
        fps=fps,
        action_domain=domain,
        action_space="joint128",
        embodiment_name="franka_panda__parallel_gripper",
        data_role="training",
        training_eligible=True,
        data_path="data/frames.parquet",
        dataset_from_index=0,
        dataset_to_index=n,
        caption_path="captions.jsonl",
        caption_index=0,
        view_mapping={"main": "camera"},
        video_views=[
            dict(
                role="main",
                fps=fps,
                source_key="camera",
                path="video.mp4",
                width=32,
                height=32,
                from_timestamp=0.0,
                to_timestamp=n / fps,
            )
        ],
    )
    excluded = dict(ep, episode_index=1, data_role="failure_archive", training_eligible=False)
    pq.write_table(pa.Table.from_pylist([ep, excluded]), root / "meta/episodes/chunk-000/episodes.parquet")
    caption = dict(
        task_title="Put cup",
        command="Put the cup on its coaster.",
        source_task_description="The robot carries the cup to its coaster and releases it.",
        outcome="failure",
        subtasks=[dict(start_time=0, end_time=3.5, command="Place cup", is_idle=False)],
    )
    (root / "captions.jsonl").write_text(json.dumps(caption) + "\n")
    config = dict(
        root=str(root),
        adapter="unified_action128",
        format="lerobot",
        mode="wam",
        source_schema_version="unified-lerobot-action128-v2",
    )
    catalog = tmp_path / "robot.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["robot"],
                "type": ["action"],
                "dataset": ["robot"],
                "data": [json.dumps(config)],
                "provenance": ["{}"],
            }
        ),
        catalog,
    )
    return root, catalog, action


@pytest.fixture
def media_stub(monkeypatch):
    class Decoder:
        def decode(self, path, *, timestamps_sec, camera_output_size=None):
            return [np.zeros((32, 32, 3), dtype=np.uint8) for _ in timestamps_sec], {"actual": timestamps_sec}

    monkeypatch.setattr(
        "cosmos_framework.data.generator.action.unified_action128.view_layout",
        lambda ep: (ep["video_views"], [(0, 0, 32, 32)], 32, 32),
    )
    monkeypatch.setattr(
        "cosmos_framework.model.generator.reasoner.qwen3_vl.utils.tokenize_generation_caption",
        lambda *args, **kwargs: ([1, 2, 3], [False] * 3),
    )
    return Decoder()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("no_rows", "no_rows"),
        ("invalid", "all_action_invalid"),
        ("no_dimensions", "no_valid_action_dimensions"),
        ("no_control", "no_controlled_action_dimensions"),
    ],
)
def test_no_target_drop_preserves_identity_and_explains_mask(source, mode, expected):
    root, _, _ = source
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    if mode == "no_rows":
        values = pa.array(np.arange(table.num_rows) / 2 + 10)
        table = table.set_column(table.column_names.index("timestamp"), "timestamp", values)
    elif mode == "invalid":
        table = table.set_column(table.column_names.index("action_valid"), "action_valid", pa.array([False] * 7))
    else:
        name = "action_dim_mask" if mode == "no_dimensions" else "action_control_mask"
        values = pa.array([[False] * 128] * 7, type=pa.list_(pa.bool_(), 128))
        table = table.set_column(table.column_names.index(name), name, values)
        if mode == "no_dimensions":
            table = table.set_column(table.column_names.index("action_control_mask"), "action_control_mask", values)
    pq.write_table(table, path, row_group_size=3)
    dataset = UnifiedAction128Dataset(root, control_only=mode == "no_control")

    result = dataset.get_item(0, epoch=3)

    assert result[DROP_SAMPLE_KEY]
    context = result[DROP_SAMPLE_CONTEXT_KEY]
    assert (context["root"], context["episode_index"], context["window_index"], context["sample_index"]) == (
        str(root),
        0,
        0,
        0,
    )
    assert context["epoch"] == 3
    assert context["category"] == expected
    assert context["rows"] == (0 if mode == "no_rows" else 4)


def test_episode_rows_ignores_nullable_auxiliary_columns(source):
    root, _, _ = source
    path = root / "data/frames.parquet"
    expected = pq.read_table(path)
    table = expected
    for name in ("previous_action_valid", "shared_eef_training_eligible", "eef_geometry_qualified"):
        table = table.append_column(name, pa.nulls(table.num_rows, type=pa.bool_()))
    pq.write_table(table, path, row_group_size=3)
    episode = pq.read_table(root / "meta/episodes/chunk-000/episodes.parquet").to_pylist()[0]

    actual = BoundedRows().episode_rows(path, episode)

    assert actual.equals(expected)


def test_episode_cache_releases_unrelated_frames_and_reuses_windows(source, monkeypatch):
    root, _, _ = source
    path = root / "data/frames.parquet"
    expected = pq.read_table(path)
    count = expected.num_rows
    second = expected.set_column(0, "index", pa.array(range(count, 2 * count)))
    second = second.set_column(1, "episode_index", pa.array([1] * count))
    pq.write_table(pa.concat_tables([expected, second]), path)
    episode = pq.read_table(root / "meta/episodes/chunk-000/episodes.parquet").to_pylist()[0]
    reader = BoundedRows()
    original = reader.row_group
    groups = []

    def read(*args, **kwargs):
        table = original(*args, **kwargs)
        groups.append(weakref.ref(table))
        return table

    monkeypatch.setattr(reader, "row_group", read)
    first = reader.episode_rows(path, episode)
    assert first.equals(expected)
    assert reader.group is None and reader.group_key is None
    assert all(group() is None for group in groups)
    assert reader.episode_rows(path, episode) is first
    assert len(groups) == 1, "another window of the same episode reread its row group"
    assert reader.episode_rows(
        path, dict(episode, episode_index=1, dataset_from_index=count, dataset_to_index=2 * count)
    ).equals(second)
    assert len(groups) == 2
    reader.max_row_group_bytes = 1
    with pytest.raises(CatalogSchemaError, match="oversized Action128 row group"):
        reader.episode_rows(path, dict(episode, episode_index=1, dataset_from_index=count, dataset_to_index=2 * count))


def test_episode_cache_identity_includes_native_pts_columns(source):
    root, _, _ = source
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    table = table.append_column("camera_pts", table["timestamp"])
    pq.write_table(table, path)
    episode = pq.read_table(root / "meta/episodes/chunk-000/episodes.parquet").to_pylist()[0]
    reader = BoundedRows()
    legacy = reader.episode_rows(path, episode)
    assert "camera_pts" not in legacy.column_names
    native = {**episode, "video_views": [{**episode["video_views"][0], "pts_column": "camera_pts"}]}
    mapped = reader.episode_rows(path, native)
    assert mapped["camera_pts"].equals(table["timestamp"])
    assert reader.episode_rows(path, native) is mapped
    assert "camera_pts" not in reader.episode_rows(path, episode).column_names


def test_large_episode_read_reclaims_free_pages_without_invalidating_cached_rows(source, monkeypatch):
    root, _, _ = source
    path = root / "data/frames.parquet"
    episode = pq.read_table(root / "meta/episodes/chunk-000/episodes.parquet").to_pylist()[0]
    count = 125_000
    table = pa.table(
        {
            "index": np.arange(count),
            "episode_index": np.r_[
                np.zeros(episode["length"], dtype=np.int64), np.ones(count - episode["length"], dtype=np.int64)
            ],
            "timestamp": np.arange(count, dtype=np.float64),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(count * 128, dtype=np.float32)), 128),
            "action_dim_mask": pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(count * 128, dtype=bool)), 128),
            "action_control_mask": pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(count * 128, dtype=bool)), 128),
            "action_valid": np.ones(count, dtype=bool),
        }
    )
    pq.write_table(table, path, row_group_size=count)
    reader = BoundedRows()
    pool = pa.default_memory_pool()
    reclaimed = []

    def reclaim():
        assert reader.group is None
        pool.release_unused()
        reclaimed.append(True)

    monkeypatch.setattr(pa, "default_memory_pool", lambda: SimpleNamespace(release_unused=reclaim))
    actual = reader.episode_rows(path, episode)
    assert reclaimed == [True]
    assert actual.equals(table.slice(0, episode["length"]))
    assert reader.episode_rows(path, episode) is actual
    assert reclaimed == [True], "a cached window should not purge the allocator again"


@pytest.mark.parametrize("field", ["episode_index", "timestamp", "action_valid"])
def test_episode_rows_rejects_null_required_fields(source, field):
    root, _, _ = source
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    values = table[field].to_pylist()
    values[1] = None
    table = table.set_column(table.schema.get_field_index(field), field, pa.array(values, type=table[field].type))
    pq.write_table(table, path, row_group_size=3)
    episode = pq.read_table(root / "meta/episodes/chunk-000/episodes.parquet").to_pylist()[0]

    with pytest.raises(CatalogSchemaError, match=f"nulls in Action128 frame rows:.*{field}"):
        BoundedRows().episode_rows(path, episode)


def test_episode_metadata_projection_preserves_training_samples(source, media_stub, monkeypatch):
    root, _, _ = source
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    table = pq.read_table(path)
    for name in ("stats/action/mean", "stats/observation.state/std", "stats/action_dim_mask/count"):
        table = table.append_column(name, pa.array([[999.0] * 128] * table.num_rows))
    table = table.append_column("unused_provenance", pa.array(["unused"] * table.num_rows))
    pq.write_table(table, path)
    read_table = pq.read_table

    # Reproduce the previous full-column read against the exact same source files.
    def read_all(path, **kwargs):
        kwargs.pop("columns", None)
        return read_table(path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pq, "read_table", read_all)
        legacy = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)

    reads = []

    def read_projected(path, **kwargs):
        columns = kwargs.get("columns")
        assert columns is not None  # Selecting after a full read would still cause the peak.
        assert not any(name.startswith("stats/") or name == "unused_provenance" for name in columns)
        reads.append(path)
        return read_table(path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pq, "read_table", read_projected)
        projected = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)

    assert reads == [path]
    assert projected.episodes.equals(legacy.episodes.select(projected.episodes.column_names))
    assert projected.episodes.nbytes < table.nbytes
    assert len(projected) == len(legacy) == 2
    assert list(projected.get_shuffle_blocks()) == list(legacy.get_shuffle_blocks())
    assert projected.fingerprint == legacy.fingerprint
    assert projected.transforms == legacy.transforms
    for index in range(len(projected)):
        assert projected.media_sample_identity(index) == legacy.media_sample_identity(index)
        actual, expected = projected[index], legacy[index]
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            elif key == "sequence_plan":
                assert actual[key].as_dict() == expected[key].as_dict()
            else:
                assert actual[key] == expected[key], key


def test_native_windows_values_and_masks(source, media_stub):
    root, _, expected = source
    ds = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
    assert len(ds) == 2  # Not the failure archive; caption outcome never filters.
    first, tail = ds[0], ds[1]
    normalized = torch.from_numpy(expected[:4]).clone()
    normalized[:, 0] = normalized[:, 0].mul(0.5).clamp(-1, 1)
    torch.testing.assert_close(first["action"], normalized)
    assert not first["action_loss_mask"][2].any()
    assert first["action_loss_mask"][1, 0]  # Observation targets retained by default.
    assert not first["action_control_mask"][1].any()
    assert tail["temporal_valid_mask"].tolist() == [True, True, True, False]
    assert not tail["action_loss_mask"][-1].any()
    assert first["video"].shape[1] % 4 == 1
    assert first["sequence_plan"].as_dict()["action_start_frame_offset"] == 0
    assert first["domain_id"].item() == 8  # Fixed Franka joint head; source local ID 99 is ignored.
    assert first["normalization_domain_id"] == 0
    ds.control_only = True
    assert not ds[0]["action_loss_mask"][1].any()


def test_published_transform_and_process_handles(source):
    root, _, _ = source
    spec = build_action128_spec([str(root)])
    assert spec["transforms"][0]["scale"][0] == 0.5
    ds = UnifiedAction128Dataset(root, action128_spec=spec)
    assert len(ds) == 2
    reader = BoundedRows()
    reader.parquet(root / "data/frames.parquet")
    assert not pickle.loads(pickle.dumps(reader)).handles
    assert build_registry([str(root), str(root)]) == spec["registry"]


def test_catalog_row_group_limit_and_worker_serialization(source):
    from cosmos_framework.configs.toml_config.sft_config import DataloaderMediaConfig
    from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog

    root, catalog, _ = source
    path = root / "data/frames.parquet"
    size = pq.ParquetFile(path).metadata.row_group(0).total_byte_size
    default = build_action_dataset_from_catalog(catalog)
    assert default.reader.max_row_group_bytes == 256 * 1024**2
    for limit, allowed in ((size - 1, False), (size, True)):
        media = DataloaderMediaConfig(action_max_row_group_bytes=limit).model_dump()
        dataset = build_action_dataset_from_catalog(catalog, media_config=media)
        reader = pickle.loads(pickle.dumps(dataset.reader))
        assert reader.max_row_group_bytes == limit
        if allowed:
            assert reader.row_group(path, 0).equals(pq.ParquetFile(path).read_row_group(0))
        else:
            with pytest.raises(CatalogSchemaError, match=f"uncompressed_bytes={size}, limit_bytes={limit}"):
                reader.row_group(path, 0)
            assert reader.group is None


def test_row_group_replacement_releases_old_cache_before_read_and_recovers(source, monkeypatch):
    root, _, _ = source
    path = root / "data/frames.parquet"
    reader = BoundedRows()
    reader.row_group(path, 0, columns=["index"])
    old = weakref.ref(reader.group)
    handle = reader.parquet(path)

    def read(index, columns, use_threads):
        assert old() is None, "previous cached Arrow table remains alive during replacement"
        assert reader.group is None and reader.group_key is None
        assert use_threads is False
        return handle.read_row_group(index, columns=columns, use_threads=use_threads)

    proxy = SimpleNamespace(metadata=handle.metadata, read_row_group=read)
    monkeypatch.setattr(reader, "parquet", lambda _: proxy)
    reader.row_group(path, 0, columns=["timestamp"])
    assert reader.group.equals(handle.read_row_group(0, columns=["timestamp"]))

    def fail(*args, **kwargs):
        raise OSError("read failed")

    proxy.read_row_group = fail
    with pytest.raises(OSError, match="read failed"):
        reader.row_group(path, 0, columns=["index"])
    assert reader.group is None and reader.group_key is None
    proxy.read_row_group = read
    assert reader.row_group(path, 0, columns=["timestamp"]) is not None


def test_frame_rows_after_fork_preserve_values_with_parent_arrow_pool(source):
    root, _, _ = source
    path = root / "data/frames.parquet"
    pq.read_table(path, use_threads=True)
    episode = pq.read_table(root / "meta/episodes/chunk-000/episodes.parquet").to_pylist()[0]
    reader = BoundedRows()
    expected = reader.episode_rows(path, episode).to_pydict()
    context = mp.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def read_in_child():
        try:
            sender.send(reader.episode_rows(path, episode).to_pydict())
        finally:
            sender.close()

    process = context.Process(target=read_in_child, daemon=True)
    process.start()
    sender.close()
    try:
        assert receiver.poll(10), "frame read blocked after forking an Arrow-initialized parent"
        assert receiver.recv() == expected
        process.join(timeout=2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        receiver.close()


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_row_group_limit_rejected(limit):
    from pydantic import ValidationError

    from cosmos_framework.configs.toml_config.sft_config import DataloaderMediaConfig

    with pytest.raises(ValidationError):
        DataloaderMediaConfig(action_max_row_group_bytes=limit)
    with pytest.raises(ValueError, match="positive integer"):
        BoundedRows(max_row_group_bytes=limit)


def test_append_registry_preserves_legacy_source_fingerprint(source):
    import copy

    from cosmos_framework.data.generator.action.unified_action128 import stable_hash

    root, _, _ = source
    old = UnifiedAction128Dataset(root)
    spec = copy.deepcopy(old.action128_spec)
    spec["registry"]["domains"].append("new/joint128/test")
    spec["registry"]["embodiments"].append("new")
    spec["registry"]["domain_embodiments"].append(1)
    spec["registry"]["domain_projection_ids"].append(22)
    spec["transforms"].append(copy.deepcopy(spec["transforms"][0]))
    spec["source_digests"].append("new")
    new = UnifiedAction128Dataset(root, action128_spec=spec)
    assert old.fingerprint != new.fingerprint
    assert new.fingerprint_for_spec(old.action128_spec) == old.fingerprint
    # Preserve the pre-extension implementation's exact hash format.
    inputs = {
        "root": str(old.root),
        "registry": old.registry,
        "metadata": [
            (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.glob("meta/episodes/**/*.parquet"))
        ],
        "duration": old.duration,
        "stride": old.stride,
        "control_only": False,
        "video_fps": old.video_fps,
        "video_fps_max": old.video_fps_max,
        "frequency_contract": "view-video-fps-cap-v3",
    }
    assert old.fingerprint == stable_hash([stable_hash(inputs), stable_hash(old.action128_spec)])
    changed_window = UnifiedAction128Dataset(root, action128_spec=spec, window_duration_seconds=1)
    assert changed_window.fingerprint_for_spec(old.action128_spec) != old.fingerprint
    changed_fps_cap = UnifiedAction128Dataset(root, action128_spec=spec, video_fps_max=30)
    assert changed_fps_cap.fingerprint_for_spec(old.action128_spec) != old.fingerprint


@pytest.mark.parametrize("missing", ["meta/action_normalization.json", "meta/a1_structural_gate.json", "manifest.json"])
def test_published_contract_has_no_missing_sidecar_fallback(source, missing):
    root, _, _ = source
    (root / missing).unlink()
    with pytest.raises(CatalogSchemaError):
        build_action128_spec([str(root)])


def test_published_contract_accepts_parallel_publisher(source):
    root, _, _ = source
    path = root / "meta/action_normalization.json"
    document = json.loads(path.read_text())
    del document["fit_scope"]["control_mask_used_for_fit"]
    document["application"]["field"] = "action"
    path.write_text(json.dumps(document))
    assert build_action128_spec([str(root)])["registry"]["domains"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("application", "parquet_values_pre_scaled", True),
        ("application", "field", "observation.state"),
        ("fit_scope", "control_mask_used_for_fit", True),
        ("fit_scope", "row_mask", "control_mask"),
    ],
)
def test_published_contract_rejects_incompatible_normalization(source, section, key, value):
    root, _, _ = source
    normalization_path = root / "meta/action_normalization.json"
    normalization = json.loads(normalization_path.read_text())
    normalization[section][key] = value
    normalization_path.write_text(json.dumps(normalization))
    with pytest.raises(CatalogSchemaError, match="fit/application"):
        build_action128_spec([str(root)])


def test_collator_action128_topology(source, media_stub):
    root, _, _ = source
    ds = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
    samples = [ds[0], ds[1]]
    collator = VFMListCollator(
        static_packing_config=StaticPackingConfig(action_dim=128, enable_fps_modulation=True),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
    )
    batch = collator.collate(samples)
    packed = batch["packed_sequence"]
    assert len(packed.action.tokens) == 2
    assert packed.action.token_shapes == [(4,), (4,)]
    assert batch["action"][0][0].shape[-1] == 128
    torch.testing.assert_close(packed.action.loss_valid_mask[1], samples[1]["action_loss_mask"])
    torch.testing.assert_close(packed.vision.loss_valid_mask[1], samples[1]["vision_loss_mask"])
    restored = pickle.loads(pickle.dumps(packed))
    torch.testing.assert_close(restored.action.loss_valid_mask[1], samples[1]["action_loss_mask"])


@pytest.mark.parametrize("through_catalog", [False, True])
@pytest.mark.parametrize("video_override", ["missing", None, 3.0, 30.0])
def test_independent_video_action_frequency(source, media_stub, video_override, through_catalog):
    root, catalog, _ = source
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    episodes = pq.read_table(path).to_pylist()
    for episode in episodes:
        episode["video_views"][0]["fps"] = 4.0  # Video 4 Hz, action 2 Hz.
    pq.write_table(pa.Table.from_pylist(episodes), path)
    rows = pq.read_table(catalog).to_pylist()
    data = json.loads(rows[0]["data"])
    if video_override == "missing":
        data.pop("fps", None)
    else:
        data["fps"] = video_override
    rows[0]["data"] = json.dumps(data)
    pq.write_table(pa.Table.from_pylist(rows), catalog)
    if through_catalog:
        from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog

        ds = build_action_dataset_from_catalog(catalog, tokenizer_config=object())
    else:
        kwargs = {} if video_override == "missing" else {"fps": video_override}
        ds = UnifiedAction128Dataset(root, tokenizer_config=object(), **kwargs)
    ds.decoder = media_stub
    sample = ds[0]
    video_fps = min(video_override, 4.0) if isinstance(video_override, float) else 4.0
    caption = json.loads(sample["ai_caption"])
    assert caption["fps"] == sample["conditioning_fps"] == video_fps
    assert caption["action_frequency_hz"] == sample["conditioning_fps_action"] == 2.0
    expected_frames = {3.0: 9, 4.0: 9}[video_fps]
    assert sample["video"].shape[1] == expected_frames
    assert sample["action"].shape == (4, 128)
    torch.testing.assert_close(sample["action_timestamps"], torch.arange(4, dtype=torch.float64) / 2)
    collator = VFMListCollator(
        static_packing_config=StaticPackingConfig(action_dim=128, enable_fps_modulation=True),
        special_tokens={"eos_token_id": 2, "start_of_generation": 3, "end_of_generation": 4},
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collator.collate)))
    packed = batch["packed_sequence"]
    action_positions = packed.position_ids[:, packed.action.sequence_indexes]
    vision_positions = packed.position_ids[:, packed.vision.sequence_indexes]
    offset = vision_positions[0, 0]
    torch.testing.assert_close(action_positions[0] - offset, torch.arange(4).float() * 3)
    assert not action_positions[1:].any()
    torch.testing.assert_close(
        vision_positions[0].unique() - offset,
        torch.arange((expected_frames - 1) // 4 + 1).float() * (24 / video_fps),
    )
    assert batch["conditioning_fps_action"][0].item() == 2.0
    assert batch["conditioning_fps"][0].item() == video_fps


@pytest.mark.parametrize("bad_fps", [0, -1, float("nan"), float("inf"), True, "30"])
def test_invalid_video_frequency_override(source, bad_fps):
    root, _, _ = source
    with pytest.raises(CatalogSchemaError, match="video fps override"):
        UnifiedAction128Dataset(root, fps=bad_fps)


@pytest.mark.parametrize("catalog_fps", ["missing", None, 3.0, 100.0])
@pytest.mark.parametrize("view_fps", [15.0, 20.0, 30.0, 187.0])
def test_catalog_video_fps_cap_preserves_lower_rates_action_and_source(source, media_stub, catalog_fps, view_fps):
    from cosmos_framework.data.generator.catalog_processors import build_action_dataset_from_catalog

    root, catalog, _ = source
    metadata = root / "meta/episodes/chunk-000/episodes.parquet"
    episodes = pq.read_table(metadata).to_pylist()
    for episode in episodes:
        episode["video_views"][0]["fps"] = view_fps
    pq.write_table(pa.Table.from_pylist(episodes), metadata)
    metadata_before = metadata.read_bytes()
    expected = min(view_fps, 30.0, catalog_fps if isinstance(catalog_fps, float) else float("inf"))
    rows = pq.read_table(catalog).to_pylist()
    data = json.loads(rows[0]["data"])
    if catalog_fps != "missing":
        data["fps"] = catalog_fps
    rows[0]["data"] = json.dumps(data)
    pq.write_table(pa.Table.from_pylist(rows), catalog)
    before = catalog.read_bytes()
    ds = build_action_dataset_from_catalog(catalog, tokenizer_config=object(), action_video_fps_default=30.0)
    ds.decoder = media_stub
    sample = ds[0]
    assert sample["conditioning_fps"] == expected
    assert sample["conditioning_fps_action"] == 2.0
    assert sample["action"].shape == (4, 128)
    assert catalog.read_bytes() == before
    assert metadata.read_bytes() == metadata_before


def test_missing_camera_frequency_is_not_replaced_with_action_or_default(source, media_stub):
    root, _, _ = source
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    episodes = pq.read_table(path).to_pylist()
    for episode in episodes:
        episode["video_views"][0].pop("fps")
    pq.write_table(pa.Table.from_pylist(episodes), path)
    ds = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
    with pytest.raises(CatalogSchemaError, match="missing video_views fps"):
        ds[0]
    override = UnifiedAction128Dataset(
        root, fps=4, video_fps_max=30, tokenizer_config=object(), video_decoder=media_stub
    )
    with pytest.raises(CatalogSchemaError, match="missing video_views fps"):
        override[0]
    assert ds.fingerprint != override.fingerprint


def test_caption_keeps_complete_single_subtask():
    caption = dict(
        task_title="Put cup",
        command="Put cup on coaster",
        source_task_description="The robot places the cup on the coaster and opens the gripper.",
        subtasks=[dict(command="Place cup", start_time=0, end_time=1.0667, is_idle=False)],
    )
    modes = set()
    for seed in range(100):
        result = json.loads(
            render_caption(
                caption,
                start=0.5,
                duration=2,
                fps=30,
                action_frequency_hz=30,
                width=640,
                height=640,
                embodiment="franka_panda__parallel_gripper",
                seed=seed,
            )
        )
        assert result["embodiment"] == "franka_panda__parallel_gripper"
        assert result["command"] == caption["source_task_description"]
        if "task_plan" in result:
            assert result["task_plan"] == ["Place cup"]
        modes.add(("task_plan" in result, "current_subtask" in result))
    assert modes == {(False, False), (True, False), (True, True)}


def test_auxiliary_view_is_explicit_primary_fallback():
    views = [
        dict(role="auxiliary_0", source_key="front", width=640, height=480),
        dict(role="left_wrist", source_key="wrist", width=320, height=240),
    ]
    selected, boxes, width, height = view_layout(
        {"video_views": views, "view_mapping": {"main": None, "auxiliary_0": "front", "left_wrist": "wrist"}}
    )
    assert [view["role"] for view in selected] == ["auxiliary_0", "left_wrist"]
    assert len(boxes) == 2 and (width, height) == (640, 960)


def test_caption_probability_boundaries_and_camera_context(monkeypatch):
    caption = dict(
        task_title="Task",
        command="Full task",
        source_task_description="The robot performs the complete task described by the VLM.",
        subtasks=[
            dict(start_time=0, end_time=1, command="First"),
            dict(start_time=1, end_time=3, command="Second"),
        ],
    )
    layout = [["main", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    for draw, expected in (
        (0.0, (True, False)),
        (0.599, (True, False)),
        (0.6, (True, True)),
        (0.899, (True, True)),
        (0.9, (False, False)),
        (0.999, (False, False)),
    ):
        monkeypatch.setattr(
            "cosmos_framework.data.generator.action.unified_action128.stable_hash",
            lambda _, d=draw: format(int(d * 2**128), "032x"),
        )
        result = json.loads(
            render_caption(
                caption,
                start=1.5,
                duration=2,
                fps=30,
                action_frequency_hz=30,
                width=640,
                height=960,
                embodiment="franka_panda__parallel_gripper",
                camera_layout=layout,
                seed=0,
            )
        )
        assert ("task_plan" in result, "current_subtask" in result) == expected
        assert result["command"] == caption["source_task_description"] and result["camera_layout"] == layout
        assert result["embodiment"] == "franka_panda__parallel_gripper"
        if expected[0]:
            assert result["task_plan"] == ["First", "Second"]
        if expected[1]:
            assert result["current_subtask"] == "Second"


@pytest.mark.parametrize("description", [None, "", "   "])
def test_caption_requires_source_description(description):
    with pytest.raises(CatalogSchemaError, match="source_task_description"):
        render_caption(
            {"task_title": "Task", "command": "Concatenated steps", "source_task_description": description},
            start=0,
            duration=2,
            fps=30,
            action_frequency_hz=30,
            width=640,
            height=480,
            embodiment="franka_panda__parallel_gripper",
            seed=42,
        )


@pytest.mark.parametrize("views_count", [1, 2, 3])
def test_video_cover_crop_and_four_pixel_separators(source, views_count):
    class Decoder:
        def decode(self, path, *, timestamps_sec, camera_output_size=None):
            # Tall source: cover resize must crop red top/bottom bands away.
            pixels = np.full((192, 64, 3), 200, dtype=np.uint8)
            pixels[:40] = [255, 0, 0]
            pixels[-40:] = [255, 0, 0]
            return [pixels for _ in timestamps_sec], {}

    ds = UnifiedAction128Dataset(source[0], video_decoder=Decoder(), window_duration_seconds=0.1)
    roles = ["main", "left_wrist", "right_wrist"][:views_count]
    episode = {
        "view_mapping": {role: role for role in roles},
        "video_views": [
            dict(
                role=role,
                source_key=role,
                width=64,
                height=192,
                path="unused.mp4",
                from_timestamp=0,
                to_timestamp=1,
                fps=10,
            )
            for role in roles
        ],
    }
    video, valid, _, meta, width, height = ds._video(episode, 0, 0.1, 10)
    frame = video[:, 0]
    expected = torch.full_like(frame, 200)
    if views_count > 1:
        boundary = height // 2 if views_count == 2 else round(height * 2 / 3)
        expected[:, boundary - 2 : boundary + 2] = 0
    if views_count == 3:
        expected[:, boundary:, width // 2 - 2 : width // 2 + 2] = 0
    assert torch.equal(frame, expected)
    assert valid[0].all()  # Separators remain valid pixels.
    assert (meta["camera_layout"] is not None) == (views_count > 1)
    if views_count > 1:
        assert isinstance(meta["camera_layout"], list)
        assert len(meta["camera_layout"]) == views_count


@pytest.mark.parametrize("count", [1, 4, 5, 9])
@pytest.mark.parametrize(
    "source_hw,target_hw",
    [((48, 64), (37, 51)), ((73, 41), (53, 87)), ((32, 48), (32, 48)), ((1080, 1920), (480, 640))],
)
def test_chunked_camera_resize_is_pixel_exact_and_preserves_canvas(count, source_hw, target_hw):
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (*source_hw, 3), dtype=np.uint8) for _ in range(count)]
    height, width = target_hw
    canvas = torch.full((3, count, height + 4, width + 6), 17, dtype=torch.uint8)
    expected = canvas.clone()
    pixels = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    scale = max(width / source_hw[1], height / source_hw[0])
    h, w = math.ceil(source_hw[0] * scale), math.ceil(source_hw[1] * scale)
    resized = torch.nn.functional.interpolate(pixels, (h, w), mode="bilinear", align_corners=False)
    resized = resized.round().clamp(0, 255).byte()
    x, y = (w - width) // 2, (h - height) // 2
    expected[:, :, 2 : 2 + height, 3 : 3 + width] = resized[:, :, y : y + height, x : x + width].permute(1, 0, 2, 3)
    _copy_resized_camera(frames, canvas[:, :, 2 : 2 + height, 3 : 3 + width])
    assert torch.equal(canvas, expected)


def test_projection_routing_shares_eef_but_keeps_joint_embodiments():
    from cosmos_framework.data.generator.action.projection_routing import PROJECTION_HEAD_IDS, projection_head_id

    assert len(PROJECTION_HEAD_IDS) == 39
    assert set(range(32)).issubset(PROJECTION_HEAD_IDS.values())
    assert len(set(PROJECTION_HEAD_IDS.values())) == len(PROJECTION_HEAD_IDS)
    assert projection_head_id("eef20", "franka_panda__parallel_gripper") == 0
    assert projection_head_id("eef20", "agibot_g1__mixed_end_effectors") == 0
    assert projection_head_id("joint128", "franka_panda__parallel_gripper") == 8
    assert projection_head_id("joint128", "agibot_g1__mixed_end_effectors") == 1
    with pytest.raises(ValueError, match="Unregistered"):
        projection_head_id("joint128", "unknown")


def test_original_projection_ids_are_unchanged():
    from cosmos_framework.data.generator.action.projection_routing import NUM_PROJECTION_SLOTS, PROJECTION_HEAD_IDS

    original_joint_embodiments = [
        "agibot_g1__mixed_end_effectors",
        "agilex_aloha__parallel_gripper",
        "agilex_aloha_mobile__parallel_gripper",
        "agilex_piper__parallel_gripper",
        "ark_bimanual__parallel_gripper",
        "ark_mobile_bimanual__parallel_gripper",
        "arx_x5__parallel_gripper",
        "franka_panda__parallel_gripper",
        "galbot_g1_mobile_bimanual__parallel_grippers",
        "robomind_agilex_bimanual__native_end_effectors",
        "tianyi_humanoid__parallel_grippers",
        "tianyi_mobile_humanoid__parallel_grippers",
        "tienkung__gello_parallel_grippers",
        "tienkung__native_hand6",
        "tienkung__parallel_grippers",
        "tienkung__prod1_parallel_grippers",
        "tienkung__xsens_hands",
        "tienkung_sim__native_hand12",
        "universal_robots_ur5__native_hand12",
        "universal_robots_ur5__parallel_gripper",
        "universal_robots_ur5e__parallel_gripper",
    ]
    assert PROJECTION_HEAD_IDS["eef20/shared"] == 0
    for head, embodiment in enumerate(original_joint_embodiments, start=1):
        assert PROJECTION_HEAD_IDS[f"joint128/{embodiment}"] == head
    assert NUM_PROJECTION_SLOTS == 64


def _replace_source_embodiment(root, embodiment):
    path = root / "meta/action_normalization.json"
    document = json.loads(path.read_text())
    old_domain, record = next(iter(document["domains"].items()))
    domain = embodiment + "/" + old_domain.split("/", 1)[1]
    document["domains"] = {domain: record}
    path.write_text(json.dumps(document))
    (root / "meta/domains.json").write_text(json.dumps({"domains": {domain: 99}}))
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    episodes = pq.read_table(path).to_pylist()
    for episode in episodes:
        episode.update(embodiment_name=embodiment, action_domain=domain)
    pq.write_table(pa.Table.from_pylist(episodes), path)


@pytest.mark.parametrize(
    ("head", "embodiment"),
    [
        (22, "agibot_g2__parallel_grippers"),
        (23, "galaxea_r1lite_mobile_bimanual__parallel_grippers"),
        (24, "galaxea_r1pro_mobile_bimanual__parallel_grippers"),
        (25, "agilex_split_aloha__parallel_grippers"),
        (26, "arx_lift2__parallel_grippers"),
        (27, "genie1__parallel_grippers"),
        (28, "panda_omron_mobile__parallel_gripper"),
        (29, "paxini_df2__faas_bimanual_hands"),
        (30, "sharpa_north__wave_bimanual_hands"),
        (31, "agilex_piper_x__parallel_gripper"),
    ],
)
def test_appended_projection_routes_published_samples(source, media_stub, head, embodiment):
    from cosmos_framework.data.generator.action.projection_routing import projection_head_id

    root = source[0]
    _replace_source_embodiment(root, embodiment)
    ds = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
    sample = ds[0]
    assert sample["domain_id"].item() == head
    assert sample["normalization_domain_id"] == 0  # Source/statistics ID is independent.
    assert sample["action"].shape[-1] == 128
    assert json.loads(sample["ai_caption"])["embodiment"] == embodiment
    assert projection_head_id("eef20", embodiment) == 0


@pytest.mark.parametrize(
    ("space", "embodiment"),
    [("action128", "xspark_humantouch_bimanual"), ("joint128", "unregistered_robot")],
)
def test_unpublished_or_unconfirmed_projection_is_not_registered(space, embodiment):
    from cosmos_framework.data.generator.action.projection_routing import projection_head_id

    with pytest.raises(ValueError, match="Unregistered"):
        projection_head_id(space, embodiment)


@pytest.mark.parametrize(
    ("head", "embodiment"),
    [(14, "tienkung__native_hand6"), (17, "tienkung__xsens_hands"), (18, "tienkung_sim__native_hand12")],
)
def test_hand_contract_reload_invalidates_fingerprint(source, media_stub, head, embodiment):
    root = source[0]
    _replace_source_embodiment(root, embodiment)
    old = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
    before = old[0]  # Materialize the old per-worker row-group cache.
    path = root / "meta/action_normalization.json"
    document = json.loads(path.read_text())
    record = next(iter(document["domains"].values()))
    record["transform"]["scale"][0] = 0.25
    path.write_text(json.dumps(document))
    path = root / "data/frames.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        row["action"][14] = 0.0
        row["action_dim_mask"][14] = False
        row["action_control_mask"][14] = False
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, row_group_size=3)
    fresh = UnifiedAction128Dataset(root, tokenizer_config=object(), video_decoder=media_stub)
    after = fresh[0]
    assert old.fingerprint != fresh.fingerprint
    assert before["action128_contract_fingerprint"] != after["action128_contract_fingerprint"]
    assert before["domain_id"].item() == after["domain_id"].item() == head
    assert before["action"][0, 0] == -1.0 and after["action"][0, 0] == -0.75
    assert before["action_dim_mask"][:, 14].all() and not after["action_dim_mask"][:, 14].any()
    assert not after["action_loss_mask"][:, 14].any()


def test_shared_projection_does_not_merge_normalization(source):
    root = source[0]
    path = root / "meta/action_normalization.json"
    document = json.loads(path.read_text())
    record = next(iter(document["domains"].values()))
    record["action_space"] = "eef20"
    first = "franka_panda__parallel_gripper/eef20/observed_local_delta/source_a"
    second = "franka_panda__parallel_gripper/eef20/observed_local_delta/source_b"
    other = json.loads(json.dumps(record))
    other["transform"]["scale"][0] = 0.25
    document["domains"] = {first: record, second: other}
    path.write_text(json.dumps(document))
    (root / "meta/domains.json").write_text(json.dumps({"domains": {first: 9, second: 3}}))
    spec = build_action128_spec([str(root)])
    assert spec["registry"]["domain_projection_ids"] == [0, 0]
    assert [item["scale"][0] for item in spec["transforms"]] == [0.5, 0.25]
    inverse = Action128Normalization(spec)
    assert inverse.denormalize(torch.ones(1, 128), torch.tensor([0]))[0, 0] == 2
    assert inverse.denormalize(torch.ones(1, 128), torch.tensor([1]))[0, 0] == 4


@pytest.mark.parametrize("field", ["action_projection_key", "state_projection_key"])
@pytest.mark.parametrize("first_published", [False, True])
def test_shared_domain_rejects_conflicting_projection_routes(source, tmp_path, field, first_published):
    root = source[0]
    other = tmp_path / "other_robot"
    shutil.copytree(root, other)
    domain = next(iter(json.loads((root / "meta/domains.json").read_text())["domains"]))
    family = "joint::franka_panda__parallel_gripper/joint128/observed_absolute_state_t_plus_1/univtac-franka9-two-finger-mean04-next-state-v1"
    route = {"action_projection_key": family, "state_projection_key": family}
    if first_published:
        (root / "meta/state_action_projection_routing.json").write_text(json.dumps({"domains": {domain: route}}))
    # Equivalent legacy/published declarations must continue to merge.
    (other / "meta/state_action_projection_routing.json").write_text(json.dumps({"domains": {domain: route}}))
    assert build_action128_spec([str(root), str(other)]) == build_action128_spec([str(root)])
    route[field] = "joint::bench2dex_all_pretrain_v1"
    (other / "meta/state_action_projection_routing.json").write_text(json.dumps({"domains": {domain: route}}))
    for roots in ([str(root), str(other)], [str(other), str(root)]):
        with pytest.raises(CatalogSchemaError, match="conflicting.*projection"):
            build_action128_spec(roots)


def test_toml_wires_fresh_head(source, tmp_path, monkeypatch, media_stub):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator.catalog_processors import build_catalog_training_dataloader
    from cosmos_framework.data.test.catalog_test import _FakeQwenProcessor

    _, catalog, _ = source
    recipe = tmp_path / "train.toml"
    recipe.write_text(
        '''[job]
task = "vfm"
experiment = "action_policy_droid_nano"
[model]
action_gen = true
[dataloader_train]
num_workers = 0
[[dataloader_train.datasets]]
path = "'''
        + str(catalog)
        + """"
ratio = 1
"""
    )
    config = load_experiment_from_toml(recipe)
    assert config.model.config.max_action_dim == 128
    assert config.model.config.num_embodiment_domains == 64
    assert config.model.config.action128_spec["registry"]["domain_projection_ids"] == [8]
    assert config.model.config.action128_spec["transforms"][0]["scale"][0] == 0.5
    assert "action2llm" in config.optimizer.keys_to_select
    assert config.checkpoint.keys_to_skip_loading == []
    assert config.checkpoint.action128_load_mode == "auto"
    monkeypatch.setattr(
        "cosmos_framework.data.generator.catalog_processors.build_video_decoder_from_config", lambda _: media_stub
    )
    joint = build_catalog_training_dataloader(
        dataset_specs=[dict(spec) for spec in config.dataloader_train.dataset_specs],
        action128_spec=config.dataloader_train.action128_spec,
        processor=_FakeQwenProcessor(),
        num_workers=0,
        max_samples_per_batch=2,
        static_packing_config=dict(config.dataloader_train.static_packing_config),
    )
    batch = next(iter(joint))
    assert len(batch["packed_sequence"].action.loss_valid_mask) == 2


@pytest.mark.parametrize("spatial_mode", ["action128_seam", "canvas"])
def test_production_three_tower_recipe_uses_action128_contract_and_complete_schedule(
    source, tmp_path, monkeypatch, spatial_mode
):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.functional.lr_scheduler import LambdaLinearScheduler

    _, catalog, _ = source
    monkeypatch.setenv("HF_MODEL_PATH", "/tmp/tokenizer")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/converted-three-tower")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/vae.pth")
    recipe_text = Path("examples/toml/sft_config/action128_wam_nano.toml").read_text()
    for dataset in tomllib.loads(recipe_text)["dataloader_train"]["datasets"]:
        recipe_text = recipe_text.replace(dataset["path"], str(catalog))
    recipe = tmp_path / "action128_three_tower.toml"
    recipe.write_text(recipe_text)
    overrides = (
        []
        if spatial_mode == "action128_seam"
        else ["model.config.diffusion_expert_config.vision_spatial_position_mode=canvas"]
    )
    config = load_experiment_from_toml(recipe, extra_overrides=overrides)
    model = config.model.config
    assert model.diffusion_expert_config.vision_spatial_position_mode == spatial_mode
    assert config.dataloader_train.static_packing_config.vision_spatial_position_mode == spatial_mode
    assert model.tower_structure == "three_tower"
    assert model.action_tower_preset == "width_only_1b"
    assert model.action_tower_init == "checkpoint"
    assert model.action_tower_execution == "native"
    assert model.max_action_dim == 128
    assert model.num_embodiment_domains == 64
    assert model.action128_spec["registry"]["domain_projection_ids"] == [8]
    assert config.checkpoint.strict_resume is True
    assert config.checkpoint.keys_to_skip_loading == []
    assert config.checkpoint.action128_load_mode == "auto"
    from types import SimpleNamespace

    from cosmos_framework.callbacks.training_stats import TrainingStatsCallback
    from cosmos_framework.utils.callback import CallBackGroup

    callback_config = config.trainer.callbacks.training_stats
    group = CallBackGroup(SimpleNamespace(trainer=SimpleNamespace(callbacks={"training_stats": callback_config})), None)
    assert isinstance(group._callbacks[0], TrainingStatsCallback)
    assert model.vision_vae_mode == "online"
    assert model.compile.enabled and not model.compile.compile_dynamic
    assert model.compile.compiled_region == "language"
    assert model.fixed_sequence_length.enabled
    assert (model.fixed_sequence_length.und_tokens, model.fixed_sequence_length.gen_tokens) == (11776, 62592)
    assert model.fixed_sequence_length.action_tokens == 2944
    assert model.action_tower_und_short_tokens == 5120
    assert model.parallelism.data_parallel_shard_degree == 16
    assert model.parallelism.data_parallel_replicate_degree == -1
    assert config.trainer.compile_cache.enabled
    assert config.dataloader_train.fixed_sequence_length_enabled
    assert config.dataloader_train.fixed_gen_tokens == 62592
    assert config.dataloader_train.fixed_action_tokens == 2944
    assert config.dataloader_train.max_samples_per_batch == 92
    assert config.dataloader_train.media_config.decode_timeout_sec == 300
    # The documented local CLI overrides must reach both model and packer.
    local = load_experiment_from_toml(
        recipe,
        extra_overrides=[
            "dataloader_train.max_samples_per_batch=4",
            "dataloader_train.max_sequence_length=40960",
            "model.config.max_num_tokens_after_packing=40960",
            "model.config.fixed_sequence_length.und_tokens=8192",
            "model.config.fixed_sequence_length.gen_tokens=32768",
            "model.config.fixed_sequence_length.action_tokens=1024",
            "model.config.action_tower_und_short_tokens=2048",
            "model.config.parallelism.data_parallel_shard_degree=8",
        ],
    )
    assert local.model.config.fixed_sequence_length.action_tokens == local.dataloader_train.fixed_action_tokens == 1024
    assert local.model.config.fixed_sequence_length.gen_tokens == local.dataloader_train.fixed_gen_tokens == 32768
    assert local.model.config.max_num_tokens_after_packing == local.dataloader_train.max_sequence_length == 40960
    assert local.model.config.parallelism.data_parallel_shard_degree == 8
    assert local.model.config.action_tower_und_short_tokens == 2048
    assert local.dataloader_train.max_samples_per_batch == 4
    scheduler = LambdaLinearScheduler(
        **{key: config.scheduler[key] for key in ("warm_up_steps", "f_min", "f_max", "f_start", "cycle_lengths")}
    )
    for step in (0, 100, 101, config.trainer.max_iter):
        assert np.isfinite(scheduler(step)) and scheduler(step) >= 0


def test_fresh_heads_meta_and_masked_loss(source):
    transform = dict(
        normalization_mask=[True] + [False] * 127,
        scale=[2.0] + [1.0] * 127,
        bias=[1.0] + [0.0] * 127,
        clip_min=-1.0,
        clip_max=1.0,
    )
    spec = build_action128_spec([str(source[0])])
    spec["transforms"] = [transform]
    with torch.device("meta"):
        meta = Action128DomainLinear(128, 8, 64)
    assert meta.fc.weight.is_meta and meta.fc.weight.shape == (64, 128 * 8)
    input_head = Action128DomainLinear(128, 8, 64)
    output_head = Action128DomainLinear(8, 128, 64)
    head = Action128Normalization(spec)
    values = torch.randn(4, 128)
    ids = torch.zeros(4, dtype=torch.long)
    result = output_head(input_head(values, ids), ids)
    assert result.shape == (4, 128)
    assert not any("action2llm" in name for name in head.state_dict())
    wrong = dict(spec, registry=dict(spec["registry"], domains=["other_robot/joint"]))
    incompatible = Action128Normalization(wrong)
    with pytest.raises(ValueError, match="checkpoint registry/transforms"):
        incompatible.load_state_dict(head.state_dict())
    denormalized = head.denormalize(torch.ones(1, 128), torch.tensor([0]))
    assert denormalized[0, 0] == 0 and denormalized[0, 1] == 1

    class Flow:
        def train_time_weight(self, ts, kwargs):
            return torch.ones_like(ts)

    prediction = torch.ones(4, 128, requires_grad=True)
    mask = torch.zeros(4, 128, dtype=torch.bool)
    mask[0, 0] = True
    loss, _ = compute_flow_matching_loss(
        [prediction],
        [torch.zeros_like(prediction)],
        [torch.zeros(4, 1)],
        torch.zeros(1, 1),
        True,
        Flow(),
        {},
        loss_valid_mask=[mask],
    )
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad.count_nonzero() == 1
    mask.zero_()
    empty, _ = compute_flow_matching_loss(
        [prediction],
        [torch.zeros_like(prediction)],
        [torch.zeros(4, 1)],
        torch.zeros(1, 1),
        True,
        Flow(),
        {},
        loss_valid_mask=[mask],
    )
    assert empty.item() == 0

    video = torch.ones(2, 3, 2, 2, requires_grad=True)
    spatial_valid = torch.zeros(3, 2, 2, dtype=torch.bool)
    spatial_valid[:, 0, 0] = True
    condition = torch.zeros(3, 1, 1)
    condition[0] = 1
    vision_loss, _ = compute_flow_matching_loss(
        [video],
        [torch.zeros_like(video)],
        [condition],
        torch.zeros(1, 1),
        True,
        Flow(),
        {},
        loss_valid_mask=[spatial_valid],
    )
    vision_loss.backward()
    assert vision_loss.item() == 1 and video.grad.count_nonzero() == 4


def test_action128_registration_preserves_cli_overrides(source, tmp_path):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    _, catalog, _ = source
    recipe = tmp_path / "overrides.toml"
    recipe.write_text(
        '[job]\ntask="vfm"\nexperiment="action_policy_droid_nano"\n'
        '[model]\naction_gen=true\n[[dataloader_train.datasets]]\npath="' + str(catalog) + '"\nratio=1\n'
    )
    config = load_experiment_from_toml(
        recipe,
        extra_overrides=[
            "trainer.callbacks.training_stats.log_freq=7",
            "checkpoint.keys_to_skip_loading=[net_ema.]",
            "checkpoint.action128_load_mode=reset_heads",
        ],
    )
    assert config.trainer.callbacks.training_stats.log_freq == 7
    assert config.trainer.callbacks.training_stats._target_.endswith(".TrainingStatsCallback")
    assert list(config.checkpoint.keys_to_skip_loading) == ["net_ema."]
    assert config.checkpoint.action128_load_mode == "reset_heads"


def test_media_failure_carries_camera_context_and_resets_decoder(source, media_stub):
    from cosmos_framework.data.generator.media_decode import MediaDecodeError

    class BrokenDecoder:
        closed = False

        def decode(self, path, *, timestamps_sec, camera_output_size=None):
            raise MediaDecodeError("invalid AV1 packet")

        def close(self):
            self.closed = True

    decoder = BrokenDecoder()
    dataset = UnifiedAction128Dataset(source[0], tokenizer_config=object(), video_decoder=decoder)
    identity = dataset.media_sample_identity(0)
    assert identity["episode_index"] == identity["window_index"] == 0
    with pytest.raises(MediaDecodeError) as caught:
        dataset[0]
    assert decoder.closed
    assert caught.value.context["camera"] == "main"
    assert caught.value.context["request_start_sec"] == 0
    assert caught.value.context["video_path"].startswith(str(source[0]))


def test_camera_coverage_failure_uses_media_policy_with_context(source, media_stub, tmp_path, monkeypatch):
    from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor
    from cosmos_framework.data.generator.media_decode import MediaDecodeError

    monkeypatch.setattr(media_stub, "close", lambda: None, raising=False)
    dataset = UnifiedAction128Dataset(source[0], tokenizer_config=object(), video_decoder=media_stub)
    episode = dataset.episodes.slice(0, 1).to_pylist()[0]
    short = {**episode, "video_views": [{**episode["video_views"][0], "to_timestamp": 0.25}]}
    with pytest.raises(MediaDecodeError, match="camera reference does not cover") as caught:
        dataset._video(short, 0, 1, 2)
    assert caught.value.context["camera"] == "main"
    assert caught.value.context["request_end_sec"] == 0.5
    assert caught.value.context["video_path"] == str(source[0] / "video.mp4")

    class Candidates:
        def get_shuffle_blocks(self):
            return [(0, 2)]

        def media_sample_identity(self, index):
            return {"root": str(source[0]), "episode_index": 0, "window_index": index}

        def get_item(self, index, *, epoch):
            dataset._video(short if index == 0 else episode, 0, 1, 2)
            return {"value": index}

    config = {"action_skip_media_errors": True, "action_failure_log_dir": str(tmp_path / "failures")}
    stream = EpisodeBlockDistributor(Candidates(), media_config=config).stream(0, 1, 0, 1)
    item = next(stream)
    assert item["value"] == item["_dp_stream_pos"] == 1
    records = [json.loads(line) for p in (tmp_path / "failures").glob("*.jsonl") for line in p.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["window_index"] == 0
    assert records[0]["camera"] == "main"
    assert records[0]["request_end_sec"] == 0.5
    with pytest.raises(MediaDecodeError, match="camera reference does not cover"):
        next(EpisodeBlockDistributor(Candidates()).stream(0, 1, 0, 1))


def test_native_camera_coverage_uses_source_clock_and_endpoint_tolerance(source, media_stub, monkeypatch):
    monkeypatch.setattr(media_stub, "close", lambda: None, raising=False)
    dataset = UnifiedAction128Dataset(source[0], tokenizer_config=object(), video_decoder=media_stub)
    episode = dataset.episodes.slice(0, 1).to_pylist()[0]
    # Decoder timestamps are relative to a different origin than coverage PTS.
    # The final measured PTS may exceed the nominal endpoint by half a frame.
    view = {
        **episode["video_views"][0],
        "from_timestamp": 10.0,
        "to_timestamp": 10.49,
        "source_clock_origin_s": -100.0,
        "pts_column": "camera_pts",
        "source_fps": 30.0,
    }
    native = {**episode, "video_views": [view]}
    rows = pa.table({"timestamp": [0.0, 0.5], "camera_pts": [10.0, 10.5]})
    result = dataset._video(native, 0, 1, 2, rows=rows)
    mapping = result[3]["views"][0]["time_mapping"]
    assert mapping["decoder_timestamps_sec"][:2] == [110.0, 110.5]


def test_mixed_camera_rates_use_slowest_shared_grid(source):
    dataset = UnifiedAction128Dataset(source[0], video_fps_max=30)
    episode = {
        "view_mapping": {"main": "front", "left_wrist": "wrist"},
        "video_views": [
            dict(role="main", source_key="front", width=640, height=480, fps=200),
            dict(role="left_wrist", source_key="wrist", width=640, height=480, fps=15),
        ],
    }
    assert dataset.sampling_video_fps(episode) == 15


def test_shared_shard_probe_uses_action128_episode_count_without_materializing_blocks(source, monkeypatch):
    from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor

    dataset = UnifiedAction128Dataset(source[0])
    expected = len(tuple(dataset.get_shuffle_blocks()))
    assert dataset.num_shuffle_blocks == expected
    monkeypatch.setattr(
        dataset, "get_shuffle_blocks", lambda: (_ for _ in ()).throw(AssertionError("materialized blocks"))
    )
    distributor = EpisodeBlockDistributor(dataset)
    assert distributor.active_worker_ids(0, 1, expected + 2) == list(range(expected))


def test_production_23_source_action_recipe_routes_global_prefetch_with_post_oom_topology(
    source, tmp_path, monkeypatch
):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    _, catalog, _ = source
    monkeypatch.setenv("HF_MODEL_PATH", "/tmp/action-tokenizer")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/action128-i8800")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/action-vae.pth")
    text = Path("examples/toml/sft_config/action128_wam_nano_160gpu.toml").read_text()
    original = tomllib.loads(text)
    assert original["job"]["experiment"] == "action_policy_droid_nano"
    assert len(original["dataloader_train"]["datasets"]) == 23
    assert sum(row["ratio"] for row in original["dataloader_train"]["datasets"]) == 10000
    for row in original["dataloader_train"]["datasets"]:
        text = text.replace(row["path"], str(catalog))
    recipe = tmp_path / "production_action128.toml"
    recipe.write_text(text)
    config = load_experiment_from_toml(recipe, extra_overrides=["trainer.compile_cache.path=/tmp/nano-action-v1"])
    assert config.model.config.action_gen
    assert config.model.config.tower_structure == "three_tower"
    assert config.model.config.max_action_dim == 128
    assert config.model.config.action_tower_preset == "width_only_1b"
    assert config.model.config.parallelism.data_parallel_shard_degree == 16
    assert config.checkpoint.action128_load_mode == "warm_start"
    assert config.dataloader_train.execution_backend == "shared_prefetch"
    assert config.dataloader_train.num_workers == 4
    assert config.dataloader_train.prefetch_factor == original["dataloader_train"]["prefetch_factor"]
    assert list(config.dataloader_train.logical_workers) == [0] * 23
    assert config.dataloader_train.fixed_action_tokens == 2944
    assert config.dataloader_train.fixed_und_tokens == 11776
    assert config.dataloader_train.fixed_gen_tokens == 62592
    assert config.dataloader_train.max_samples_per_batch == 92
    assert config.trainer.compile_cache.path == "/tmp/nano-action-v1"


@pytest.mark.parametrize("corruption", [None, "null", "index", "episode"])
def test_episode_rows_streamed_batches_preserve_contract(tmp_path, monkeypatch, corruption):
    n = 70000
    indexes = np.arange(n)
    episodes = np.ones(n, dtype=np.int64)
    episodes[8000:18000] = 0
    valid = np.ones(n, dtype=bool)
    null_mask = np.zeros(n, dtype=bool)
    if corruption == "null":
        null_mask[9001] = True
    if corruption == "index":
        indexes[9001] = 9000
    if corruption == "episode":
        episodes[9001] = 2
    values = pa.FixedSizeListArray.from_arrays(pa.array(np.arange(n * 128, dtype=np.float32)), 128)
    masks = pa.FixedSizeListArray.from_arrays(pa.array(np.ones(n * 128, dtype=bool)), 128)
    table = pa.table(
        {
            "index": indexes,
            "episode_index": episodes,
            "timestamp": np.arange(n) / 30.0,
            "action": values,
            "action_dim_mask": masks,
            "action_control_mask": masks,
            "action_valid": pa.array(valid, mask=null_mask),
            "observation.state": values,
            "observation.state_mask": masks,
        }
    )
    path = tmp_path / "frames.parquet"
    pq.write_table(table, path, row_group_size=n)
    episode = {
        "episode_index": 0,
        "dataset_from_index": 8000,
        "dataset_to_index": 18000,
        "length": 10000,
    }
    reader = BoundedRows(max_row_group_bytes=2 * 1024**3)

    def forbid_full(*args, **kwargs):
        raise AssertionError("large episode used full read_row_group")

    monkeypatch.setattr(reader, "row_group", forbid_full)
    if corruption:
        with pytest.raises(CatalogSchemaError):
            reader.episode_rows(path, episode, use_state=True)
        assert reader.episode is None
    else:
        actual = reader.episode_rows(path, episode, use_state=True)
        assert actual.equals(table.slice(8000, 10000))
        assert reader.group is None
        assert reader.episode_rows(path, episode, use_state=True) is actual
        size = reader.parquet(path).metadata.row_group(0).total_byte_size
        with pytest.raises(CatalogSchemaError, match="oversized Action128 row group"):
            BoundedRows(max_row_group_bytes=size - 1).episode_rows(path, episode, use_state=True)


def test_episode_cache_resume_and_fingerprint_parity(source, tmp_path, monkeypatch):
    """Existing uncached cursors resume identically after enabling shared metadata."""
    from itertools import islice

    from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor

    root, _, _ = source
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    episodes = pq.read_table(path)
    # Several admitted blocks exercise block shuffle, rank partition and epochs.
    rows = episodes.to_pylist()
    pq.write_table(pa.Table.from_pylist([dict(rows[0], episode_index=i) for i in range(13)]), path)
    raw = UnifiedAction128Dataset(root, episode_cache_dir="")
    built = UnifiedAction128Dataset(root, episode_cache_dir=tmp_path / "cache")
    hit = UnifiedAction128Dataset(root, episode_cache_dir=tmp_path / "cache")
    assert not built.episode_cache_hit and hit.episode_cache_hit
    assert raw.fingerprint == built.fingerprint == hit.fingerprint
    assert raw.episodes.equals(hit.episodes)
    np.testing.assert_array_equal(raw.counts, hit.counts)
    np.testing.assert_array_equal(raw.ends, hit.ends)
    assert list(raw.get_shuffle_blocks()) == list(hit.get_shuffle_blocks())
    # Exercise the production distributor's actual resume cursor protocol without
    # media decoding: identities are exactly the episode/window mapping in use.
    for dataset in (raw, built, hit):
        dataset.get_item = lambda index, epoch, ds=dataset: ds.media_sample_identity(index)
    for rank in range(2):
        for worker in range(2):
            original = EpisodeBlockDistributor(raw, seed=42, name="cache").stream(rank, 2, worker, 2)
            consumed = list(islice(original, 11))
            expected = list(islice(original, 17))
            for dataset in (built, hit):
                monkeypatch.setenv("COSMOS_DL_STATE_cache_WORKER_%d_EPOCH" % worker, str(consumed[-1]["_dp_epoch"]))
                monkeypatch.setenv(
                    "COSMOS_DL_STATE_cache_WORKER_%d_INDEX" % worker, str(consumed[-1]["_dp_stream_pos"])
                )
                resumed = EpisodeBlockDistributor(dataset, seed=42, name="cache").stream(rank, 2, worker, 2)
                assert list(islice(resumed, 17)) == expected

    from cosmos_framework.callbacks.cosmos_dataloader_state import JointCosmosDataLoaderStateCallback
    from cosmos_framework.data.generator.dataflow import CosmosDataLoader, IdentityProcessor
    from cosmos_framework.data.generator.dataflow.loader import JointCosmosDataLoader

    def build_loader(dataset):
        dataset.get_item = lambda index, epoch: {"id": torch.tensor(index)}
        inner = CosmosDataLoader(
            EpisodeBlockDistributor(dataset, seed=42, name="cache_resume", dataset_fingerprint=dataset.fingerprint),
            IdentityProcessor(),
            batch_size=1,
            num_workers=0,
        )
        outer = JointCosmosDataLoader({"cache_resume": {"dataloader": inner, "ratio": 1}}, seed=42)
        return outer, JointCosmosDataLoaderStateCallback(outer)

    for before, after in ((raw, built), (built, hit), (hit, hit)):
        original, callback = build_loader(before)
        iterator = iter(original)
        for _ in range(7):
            callback._update_state_from_batch(next(iterator))
        saved = callback.state_dict()
        saved_rng = torch.get_rng_state()
        expected = [next(iterator)["id"].item() for _ in range(15)]
        expected_rng = torch.get_rng_state()
        restored, restored_callback = build_loader(after)
        restored_callback.load_state_dict(saved)
        torch.set_rng_state(saved_rng)
        restored.preserve_rng_on_resume()
        resumed = iter(restored)
        assert [next(resumed)["id"].item() for _ in range(15)] == expected
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


def test_episode_cache_hits_skip_parquet_and_rebuild_corruption(source, tmp_path, monkeypatch):
    from cosmos_framework.data.generator.action.episode_metadata_cache import load_episode_metadata

    root, _, _ = source
    cache = tmp_path / "cache"
    first = load_episode_metadata(root, cache_dir=cache)
    with monkeypatch.context() as patch:
        patch.setattr(pq, "read_schema", lambda *a, **kw: pytest.fail("cache hit opened source schema"))
        patch.setattr(pq, "read_table", lambda *a, **kw: pytest.fail("cache hit opened source table"))
        second = load_episode_metadata(root, cache_dir=cache)
    assert second.cache_hit and first.episodes.equals(second.episodes)
    corrupt = first.cache_path.with_suffix(".corrupt")
    corrupt.write_bytes(b"truncated")
    corrupt.replace(first.cache_path)
    repaired = load_episode_metadata(root, cache_dir=cache)
    assert not repaired.cache_hit and first.episodes.equals(repaired.episodes)
    assert not list(cache.glob("*.tmp"))


def test_episode_cache_invalidation_and_registry_validation(source, tmp_path):
    from cosmos_framework.data.generator.action.episode_metadata_cache import load_episode_metadata

    root, _, _ = source
    cache = tmp_path / "cache"
    first = UnifiedAction128Dataset(root, episode_cache_dir=cache)
    other_stride = load_episode_metadata(root, stride=1.0, cache_dir=cache)
    assert other_stride.cache_path != first.episode_cache_path
    assert other_stride.counts.tolist() == [4]
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["duration_seconds"] = 5.0
    pq.write_table(pa.Table.from_pylist(rows), path)
    changed = UnifiedAction128Dataset(root, episode_cache_dir=cache)
    assert not changed.episode_cache_hit
    assert changed.fingerprint != first.fingerprint
    assert changed.counts.tolist() == [3]
    extra = path.with_name("extra.parquet")
    pq.write_table(pa.Table.from_pylist(rows), extra)
    added = load_episode_metadata(root, cache_dir=cache)
    assert added.episodes.num_rows == 2
    extra.unlink()
    assert load_episode_metadata(root, cache_dir=cache).cache_hit
    spec = pickle.loads(pickle.dumps(changed.action128_spec))
    spec["registry"]["embodiments"][0] = "wrong"
    with pytest.raises(CatalogSchemaError, match="domain/embodiment mismatch"):
        UnifiedAction128Dataset(root, action128_spec=spec, episode_cache_dir=cache)


def test_episode_cache_concurrent_builders(source, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock

    from cosmos_framework.data.generator.action import episode_metadata_cache as cache_module

    root, _, _ = source
    build = Mock(wraps=cache_module._build)
    monkeypatch.setattr(cache_module, "_build", build)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: cache_module.load_episode_metadata(root, cache_dir=tmp_path / "cache"), range(4))
        )
    assert build.call_count == 1
    assert sum(not result.cache_hit for result in results) == 1
    assert all(result.episodes.equals(results[0].episodes) for result in results)


def test_episode_metadata_cache_write_failure_falls_back(source, tmp_path, monkeypatch):
    from cosmos_framework.data.generator.action import episode_metadata_cache as cache_module

    root, _, _ = source
    expected = UnifiedAction128Dataset(root, episode_cache_dir="")
    monkeypatch.setenv("ACTION128_EPISODE_CACHE_DIR", str(tmp_path / "cache"))

    def fail_publish(*args):
        raise OSError("simulated cache publication failure")

    monkeypatch.setattr(cache_module.os, "replace", fail_publish)
    actual = UnifiedAction128Dataset(root)
    assert actual.episode_cache_path is None
    assert actual.fingerprint == expected.fingerprint
    assert actual.episodes.equals(expected.episodes)
    assert not list((tmp_path / "cache").glob("*.tmp"))
    assert not list((tmp_path / "cache").glob("*.arrow"))


def test_episode_metadata_cache_rejects_source_change_during_build(source, tmp_path, monkeypatch):
    from cosmos_framework.data.generator.action import episode_metadata_cache as cache_module

    root, _, _ = source
    original_build = cache_module._build

    def changing_build(metadata, stride):
        result = original_build(metadata, stride)
        pq.write_table(result.episodes, root / "meta/episodes/chunk-000/extra.parquet")
        return result

    monkeypatch.setattr(cache_module, "_build", changing_build)
    with pytest.raises(CatalogSchemaError, match="changed while building"):
        cache_module.load_episode_metadata(root, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.arrow"))


def test_episode_metadata_cache_writes_bounded_batches(source, tmp_path, monkeypatch):
    from cosmos_framework.data.generator.action import episode_metadata_cache as cache_module

    root, _, _ = source
    path = root / "meta/episodes/chunk-000/episodes.parquet"
    row = pq.read_table(path).to_pylist()[0]
    pq.write_table(pa.Table.from_pylist([dict(row, episode_index=i) for i in range(7)]), path)
    monkeypatch.setattr(cache_module, "_BATCH_ROWS", 2)
    result = cache_module.load_episode_metadata(root, cache_dir=tmp_path / "cache")
    with pa.memory_map(str(result.cache_path), "r") as mapping:
        reader = pa.ipc.open_file(mapping)
        assert reader.num_record_batches == 4
        assert [len(reader.get_batch(i)) for i in range(4)] == [2, 2, 2, 1]
    hit = cache_module.load_episode_metadata(root, cache_dir=tmp_path / "cache")
    assert hit.cache_hit and hit.episodes.equals(result.episodes)
    assert hit.episodes["episode_index"].to_pylist() == list(range(7))
    np.testing.assert_array_equal(hit.ends, np.arange(1, 8) * 2)


def test_toml_preserves_source_normalization_in_model_and_loader(source, tmp_path, monkeypatch):
    import copy

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    root, catalog, _ = source
    saved_spec = build_action128_spec([str(root)])
    saved_spec["transforms"][0]["scale"][0] = 2.0
    saved_spec["source_digests"][0]["normalization"] = "checkpoint-normalization"
    saved = {"version": 1, "spec": saved_spec, "vision_spatial_position_mode": "canvas"}
    monkeypatch.setattr(
        "cosmos_framework.checkpoint.action128.read_local_source_contract", lambda _: copy.deepcopy(saved)
    )
    recipe = tmp_path / "resume.toml"
    recipe.write_text(
        '[job]\ntask="vfm"\nexperiment="action_policy_droid_nano"\n'
        "[model]\naction_gen=true\n"
        '[checkpoint]\naction128_load_mode="extend"\nload_training_state=true\n'
        'action128_preserve_source_normalization=true\nload_path="/source/checkpoint"\n'
        '[[dataloader_train.datasets]]\npath="' + str(catalog) + '"\nratio=1\n'
    )
    config = load_experiment_from_toml(recipe)
    assert config.model.config.action128_spec == saved_spec
    assert config.dataloader_train.action128_spec == config.model.config.action128_spec
    assert build_action128_spec([str(root)])["transforms"][0]["scale"][0] == 0.5
    with pytest.raises(ValueError, match="requires full-state"):
        load_experiment_from_toml(recipe, extra_overrides=["checkpoint.load_training_state=false"])
