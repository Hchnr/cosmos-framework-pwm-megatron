"""Retention excludes incomplete, foreign, future and linked checkpoints."""

import json

import pytest

from cosmos_framework.training_backends.megatron_tp_pp.checkpoint import prune_completed


def completed(root, step, *, format_name="cosmos_megatron_tp_pp"):
    path = root / f"iter_{step:09d}"
    path.mkdir()
    (path / ".save_complete").touch()
    (path / "manifest.json").write_text(json.dumps({"format": format_name}))
    return path


def test_retention_keeps_newest_and_milestones_and_stays_inside_job(tmp_path):
    for step in (1000, 2000, 3000, 4000, 5000, 6000):
        completed(tmp_path, step)
    foreign = completed(tmp_path, 100, format_name="legacy")
    future = completed(tmp_path, 7000)
    incomplete = tmp_path / "iter_000000200"
    incomplete.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "iter_000000300"
    link.symlink_to(outside, target_is_directory=True)
    (tmp_path / "latest_checkpoint.txt").write_text("iter_000006000\n")
    prune_completed(tmp_path / "iter_000006000", keep_last_n=2, keep_every_n_steps=2000)
    assert not (tmp_path / "iter_000001000").exists()
    assert not (tmp_path / "iter_000003000").exists()
    assert all((tmp_path / f"iter_{step:09d}").is_dir() for step in (2000, 4000, 5000, 6000))
    assert all(path.exists() for path in (foreign, future, incomplete, outside, link))


def test_retention_requires_latest_publication(tmp_path):
    current = completed(tmp_path, 4000)
    old = completed(tmp_path, 2000)
    (tmp_path / "latest_checkpoint.txt").write_text(old.name + "\n")
    with pytest.raises(ValueError, match="published latest"):
        prune_completed(current, keep_last_n=1)
    assert old.exists()
    prune_completed(current, keep_last_n=None)
    assert old.exists()
