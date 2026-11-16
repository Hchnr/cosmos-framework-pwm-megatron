"""Regression coverage for temporary Action media quarantine and stream cursors."""

import json

import pytest

from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor
from cosmos_framework.data.generator.dataflow.batchers import SimpleBatcher
from cosmos_framework.data.generator.dataflow.loader import CosmosDataLoader
from cosmos_framework.data.generator.dataflow.processors import IdentityProcessor
from cosmos_framework.data.generator.drop_sample_contract import (
    DROP_SAMPLE_CONTEXT_KEY,
    DROP_SAMPLE_KEY,
    DROP_SAMPLE_REASON_KEY,
)
from cosmos_framework.data.generator.media_decode import MediaDecodeError


class Windows:
    def __init__(self, failures=(), error_type=MediaDecodeError):
        self.failures = set(failures)
        self.error_type = error_type
        self.calls = []

    def get_shuffle_blocks(self):
        return [(0, 5)]

    def media_sample_identity(self, index):
        return {"root": "/robot", "episode_index": 7, "window_index": index}

    def get_item(self, index, *, epoch):
        self.calls.append((epoch, index))
        if index in self.failures:
            error = self.error_type("broken video")
            if isinstance(error, MediaDecodeError):
                error.context = {"video_path": "/bad.mp4", "camera": "wrist", "request_start_sec": 501.4}
            raise error
        return {"value": index}


def distributor(tmp_path, dataset, **overrides):
    return EpisodeBlockDistributor(
        dataset,
        name="robot",
        media_config={
            "action_skip_media_errors": True,
            "action_failure_log_dir": str(tmp_path),
            "action_max_consecutive_skips": 3,
            **overrides,
        },
    )


def test_skip_refills_batch_and_keeps_original_cursor(tmp_path):
    dataset = Windows([1, 3])
    stream = distributor(tmp_path, dataset).stream(0, 1, 0, 1)
    batch = next(SimpleBatcher(3).batches(stream))
    assert [item["value"] for item in batch] == [0, 2, 4]
    assert [item["_dp_stream_pos"] for item in batch] == [0, 2, 4]
    assert {item["_dp_epoch"] for item in batch} == {0}
    records = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert [r["window_index"] for r in records] == [1, 3]
    assert records[-1]["skipped"] == 2
    assert records[-1]["accepted"] == 2
    assert records[-1]["video_path"] == "/bad.mp4"
    assert records[-1]["camera"] == "wrist"


def test_resume_after_success_does_not_replay_failed_window(tmp_path, monkeypatch):
    dataset = Windows([1])
    stream = distributor(tmp_path, dataset).stream(0, 1, 0, 1)
    next(stream)
    delivered = next(stream)
    assert delivered["value"] == 2
    monkeypatch.setenv("COSMOS_DL_STATE_robot_WORKER_0_EPOCH", str(delivered["_dp_epoch"]))
    monkeypatch.setenv("COSMOS_DL_STATE_robot_WORKER_0_INDEX", str(delivered["_dp_stream_pos"]))
    resumed = Windows([1])
    assert next(distributor(tmp_path, resumed).stream(0, 1, 0, 1))["value"] == 3
    assert resumed.calls == [(0, 3)]


def test_consecutive_limit_crosses_epochs(tmp_path):
    dataset = Windows(range(5))
    stream = distributor(tmp_path, dataset, action_max_consecutive_skips=7).stream(0, 1, 0, 1)
    with pytest.raises(RuntimeError, match="7 consecutive skipped"):
        next(stream)
    assert len(dataset.calls) == 7
    assert dataset.calls[-1] == (1, 1)


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_structural_and_programming_errors_are_not_swallowed(tmp_path, error_type):
    with pytest.raises(error_type, match="broken video"):
        next(distributor(tmp_path, Windows([0], error_type)).stream(0, 1, 0, 1))


def test_default_remains_fail_fast():
    with pytest.raises(MediaDecodeError):
        next(EpisodeBlockDistributor(Windows([0])).stream(0, 1, 0, 1))


def test_exclusion_avoids_decode_and_advances_cursor(tmp_path):
    exclusion = tmp_path / "exclusions.jsonl"
    exclusion.write_text(json.dumps({"root": "/robot", "episode_index": 7, "window_index": 0}) + "\n")
    dataset = Windows([0])
    item = next(distributor(tmp_path, dataset, action_exclusion_jsonl=str(exclusion)).stream(0, 1, 0, 1))
    assert item["value"] == item["_dp_stream_pos"] == 1
    assert dataset.calls == [(0, 1)]


def test_empty_worker_returns_instead_of_spinning(tmp_path):
    assert list(distributor(tmp_path, Windows()).stream(1, 2, 0, 1)) == []


def test_loader_delivers_full_batches_with_failed_windows(tmp_path):
    loader = CosmosDataLoader(
        distributor=distributor(tmp_path, Windows([1])),
        processor=IdentityProcessor(),
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["value"].tolist() == [0, 2]


def test_candidate_audit_keeps_drop_identity_and_counts_without_changing_order(tmp_path, monkeypatch):
    from cosmos_framework.data.generator import catalog
    from cosmos_framework.data.generator.dataflow import loader as loader_module

    class DroppingWindows(Windows):
        def get_item(self, index, *, epoch):
            if index == 0:
                return {
                    DROP_SAMPLE_KEY: True,
                    DROP_SAMPLE_REASON_KEY: "Action128 window has no eligible target",
                    DROP_SAMPLE_CONTEXT_KEY: {"root": "/robot", "episode_index": 7, "window_index": 0},
                }
            return super().get_item(index, epoch=epoch)

    warnings = []
    reports = []
    monkeypatch.setattr(loader_module.log, "warning", lambda message, **_: warnings.append(message))
    monkeypatch.setattr(catalog.log, "info", lambda message, **_: reports.append(message))
    original_record = catalog._ActionCandidateAudit.record

    def record_and_report(audit, outcome):
        original_record(audit, outcome)
        audit.report()

    monkeypatch.setattr(catalog._ActionCandidateAudit, "record", record_and_report)
    dataset = DroppingWindows([1])
    dist = distributor(tmp_path, dataset)
    loader = CosmosDataLoader(distributor=dist, processor=IdentityProcessor(), batch_size=1, num_workers=0)
    iterator = iter(loader)

    batch = next(iterator)

    assert batch["value"].tolist() == [2]
    assert batch["sample_index"].tolist() == [2]
    context = json.loads(warnings[0].split("context=", 1)[1])
    assert (context["dataset"], context["episode_index"], context["window_index"]) == ("robot", 7, 0)
    assert context["candidate_counts"]["no_target"] == 1
    failures = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert failures[0]["candidate_counts"]["media_error"] == 1
    assert failures[0]["candidate_counts"]["candidates"] == 2
    report = json.loads([line for line in reports if "Action128 candidate audit:" in line][-1].split(": ", 1)[1])
    assert {key: report[key] for key in ("candidates", "prepared", "no_target", "media_error")} == {
        "candidates": 3,
        "prepared": 1,
        "no_target": 1,
        "media_error": 1,
    }


def test_candidate_audit_reports_cumulative_counts_and_does_not_double_flush(monkeypatch):
    from cosmos_framework.data.generator import catalog

    reports = []
    monkeypatch.setattr(catalog.log, "info", lambda message, **_: reports.append(message))
    audit = catalog._ActionCandidateAudit("robot:1", dp_rank=3, worker_id=2, report_every=2)
    audit.record("prepared")
    assert reports == []
    audit.record("no_target")
    audit.record("prepared")
    audit.report()
    audit.report()

    assert len(reports) == 2
    first, last = [json.loads(line.split(": ", 1)[1]) for line in reports]
    assert (first["candidates"], last["candidates"], last["prepared"], last["no_target"]) == (2, 3, 2, 1)
    assert first["instance"] == last["instance"]
    assert (last["dataset"], last["dp_rank"], last["worker_id"]) == ("robot:1", 3, 2)


def test_recipe_enables_bounded_temporary_policy():
    from pathlib import Path

    import tomllib

    from cosmos_framework.configs.toml_config.sft_config import DataloaderMediaConfig

    config = tomllib.loads(Path("examples/toml/sft_config/action128_wam_nano_160gpu.toml").read_text())
    media = DataloaderMediaConfig.model_validate(config["dataloader_train"]["media"])
    assert media.action_skip_media_errors
    assert media.action_max_consecutive_skips == 100
    assert not DataloaderMediaConfig().action_skip_media_errors
    with pytest.raises(ValueError):
        DataloaderMediaConfig(action_max_consecutive_skips=0)


def test_success_resets_consecutive_failure_limit(tmp_path):
    stream = distributor(tmp_path, Windows([0, 2, 4]), action_max_consecutive_skips=2).stream(0, 1, 0, 1)
    assert next(stream)["value"] == 1
    assert next(stream)["value"] == 3
    with pytest.raises(RuntimeError, match="2 consecutive skipped"):
        next(stream)  # End of epoch and next epoch's first window both fail.


def test_invalid_exclusion_file_is_not_silently_ignored(tmp_path):
    exclusion = tmp_path / "bad.jsonl"
    exclusion.write_text("not json\n")
    with pytest.raises(ValueError):
        next(distributor(tmp_path, Windows(), action_exclusion_jsonl=str(exclusion)).stream(0, 1, 0, 1))
