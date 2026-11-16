# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.utils.compile_cache import (
    DEFAULT_COMPILE_CACHE_PATH,
    CompileCacheError,
    configure_compile_cache,
    resolve_compile_cache_path,
    sha256_file,
)


def _config(*, enabled: bool = True, path: str = DEFAULT_COMPILE_CACHE_PATH, compile_enabled: bool = True):
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(compile=SimpleNamespace(enabled=compile_enabled))),
        trainer=SimpleNamespace(compile_cache=SimpleNamespace(enabled=enabled, path=path)),
    )


@pytest.fixture(autouse=True)
def _clean_cache_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)


def test_disabled_configuration_does_not_touch_existing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "external"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(existing))

    assert configure_compile_cache(_config(enabled=False), repo_root=tmp_path) is None
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(existing)
    assert not existing.exists()


def test_default_relative_path_resolves_from_repo_root(tmp_path: Path) -> None:
    resolved = configure_compile_cache(_config(), repo_root=tmp_path)

    assert resolved == tmp_path / ".compile_cache"
    assert resolved.is_dir()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(resolved)


def test_custom_relative_path_is_independent_of_launch_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    launch_cwd = tmp_path / "elsewhere"
    repo_root.mkdir()
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)

    resolved = configure_compile_cache(_config(path="var/compiler"), repo_root=repo_root)

    assert resolved == repo_root / "var" / "compiler"
    assert not (launch_cwd / "var" / "compiler").exists()


def test_absolute_path_is_used_directly(tmp_path: Path) -> None:
    cache_path = tmp_path / "absolute-cache"

    assert configure_compile_cache(_config(path=str(cache_path)), repo_root=tmp_path / "unused") == cache_path


def test_configured_path_expands_environment_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "expanded"
    monkeypatch.setenv("COSMOS_TEST_COMPILE_CACHE", str(base))

    resolved = resolve_compile_cache_path("$COSMOS_TEST_COMPILE_CACHE/nested", repo_root=tmp_path)

    assert resolved == base / "nested"


def test_configured_path_expands_user_home() -> None:
    assert resolve_compile_cache_path("~/cosmos-test-cache") == Path.home() / "cosmos-test-cache"


def test_enabled_configuration_overrides_external_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "external"))

    configured = configure_compile_cache(_config(path="configured"), repo_root=tmp_path)

    assert configured == tmp_path / "configured"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(configured)


@pytest.mark.parametrize(
    ("config", "deterministic", "message"),
    [
        (_config(compile_enabled=False), False, "requires model.config.compile.enabled=true"),
        (_config(), True, "--deterministic"),
        (_config(path=""), False, "must not be empty"),
        (_config(path="/"), False, "refusing broad"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path,
    config: SimpleNamespace,
    deterministic: bool,
    message: str,
) -> None:
    with pytest.raises(CompileCacheError, match=message):
        configure_compile_cache(config, deterministic=deterministic, repo_root=tmp_path)


def test_existing_file_cannot_be_used_as_cache_directory(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache-file"
    cache_path.write_text("not a directory")

    with pytest.raises(CompileCacheError, match="cannot create"):
        configure_compile_cache(_config(path=str(cache_path)), repo_root=tmp_path)


def test_sha256_file_streams_tokenizer_package_digest(tmp_path: Path) -> None:
    package = tmp_path / "tokenizer.pt2"
    package.write_bytes(b"tokenizer package")

    assert sha256_file(package) == hashlib.sha256(b"tokenizer package").hexdigest()
