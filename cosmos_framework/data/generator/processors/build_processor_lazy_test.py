# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

from cosmos_framework.data.generator import processors


def test_build_processor_lazy_uses_local_repository_without_hub_download(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_build_processor(path: str, **kwargs):
        calls.append((path, kwargs))
        return "processor"

    class DownloadMustNotRun:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise AssertionError("local repository must not construct a Hub checkpoint")

    monkeypatch.setattr(processors, "build_processor", fake_build_processor)
    monkeypatch.setattr("cosmos_framework.utils.checkpoint_db.CheckpointDirHf", DownloadMustNotRun)

    assert processors.build_processor_lazy(repository=str(tmp_path), revision="main", cache_dir="cache") == "processor"
    assert calls == [(str(tmp_path), {"cache_dir": "cache"})]


def test_build_processor_lazy_applies_subdir_to_local_repository(monkeypatch, tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "text_tokenizer"
    tokenizer_dir.mkdir()
    seen: list[str] = []
    monkeypatch.setattr(processors, "build_processor", lambda path, **kwargs: seen.append(path))

    processors.build_processor_lazy(repository=str(tmp_path), revision=None, subdir="text_tokenizer")

    assert seen == [str(tokenizer_dir)]
