"""Check storage setup without changing real mounts or published datasets."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("existing_directory", [False, True])
def test_published_storage_aliases_support_existing_directories(tmp_path, existing_directory):
    target = tmp_path / "mixed"
    for mount in ("project", "projset"):
        (target / mount / "eai_pwm").mkdir(parents=True)
    legacy = tmp_path / "legacy"
    if existing_directory:
        (legacy / "projset").mkdir(parents=True)
    command = ["bash", str(Path(__file__).with_name("_pretrain_state_paths.sh")), str(legacy), str(target)]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)
    for mount in ("project", "projset"):
        assert (legacy / mount / "eai_pwm").resolve() == target / mount / "eai_pwm"
    if existing_directory:
        assert not (legacy / "projset").is_symlink()


def test_wrong_existing_storage_alias_is_rejected(tmp_path):
    target = tmp_path / "mixed"
    for mount in ("project", "projset"):
        (target / mount / "eai_pwm").mkdir(parents=True)
    legacy = tmp_path / "legacy"
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (legacy / "project").mkdir(parents=True)
    (legacy / "project" / "eai_pwm").symlink_to(wrong, target_is_directory=True)
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("_pretrain_state_paths.sh")), str(legacy), str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does not resolve" in result.stderr
    assert (legacy / "project" / "eai_pwm").resolve() == wrong


@pytest.mark.parametrize("existing_dataset", [False, True])
def test_egostandard_published_mcap_prefix(tmp_path, existing_dataset):
    target = tmp_path / "mixed"
    for mount in ("project", "projset"):
        (target / mount / "eai_pwm" / "datasets").mkdir(parents=True)
    actual = target / "project/eai_pwm/datasets/EgoStandard"
    sample = actual / "EgoStandard/mcap/task/sample.mcap"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"sample")
    published = target / "projset/eai_pwm/datasets/EgoStandard"
    if existing_dataset:
        published.mkdir()
        (published / "existing-data").write_text("keep")
    command = ["bash", str(Path(__file__).with_name("_pretrain_state_paths.sh")), str(tmp_path / "legacy"), str(target)]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)
    if existing_dataset:
        assert not published.is_symlink()
        assert (published / "existing-data").read_text() == "keep"
    else:
        assert (published / "EgoStandard/mcap/task/sample.mcap").read_bytes() == b"sample"
        assert published.resolve() == actual
