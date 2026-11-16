"""Online-required jobs must fail instead of silently changing logging mode."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cosmos_framework.utils import wandb_util


@pytest.fixture
def setup_wandb(monkeypatch, tmp_path):
    job = SimpleNamespace(wandb_mode="online", project="test", group="test", name="test", path_local=str(tmp_path))
    config = SimpleNamespace(job=job, checkpoint=SimpleNamespace())
    monkeypatch.setattr(wandb_util, "_read_wandb_id", lambda *_: "existing-run")
    monkeypatch.setattr(wandb_util.LazyConfig, "save_yaml", lambda *_: str(tmp_path / "absent.yaml"))
    monkeypatch.setattr(wandb_util.attrs, "asdict", lambda _: {})
    monkeypatch.setattr(wandb_util, "get_wandb_entity", lambda: None)
    monkeypatch.setattr(wandb_util, "get_wandb_project", lambda value: value)
    monkeypatch.setattr(
        wandb_util.wandb, "run", SimpleNamespace(settings=SimpleNamespace(mode="online"), config=MagicMock())
    )
    init = MagicMock()
    monkeypatch.setattr(wandb_util.wandb, "init", init)
    return config, init


def test_required_online_rejects_auth_fallback(monkeypatch, setup_wandb):
    config, init = setup_wandb
    monkeypatch.setenv("COSMOS_REQUIRE_WANDB_ONLINE", "1")
    init.side_effect = RuntimeError("returned error 401")
    with pytest.raises(RuntimeError, match="refusing fallback"):
        wandb_util._init_wandb_rank0(config, None)
    assert init.call_count == 1
    assert init.call_args.kwargs["mode"] == "online"


def test_required_online_rejects_offline_config(monkeypatch, setup_wandb):
    config, init = setup_wandb
    monkeypatch.setenv("COSMOS_REQUIRE_WANDB_ONLINE", "1")
    config.job.wandb_mode = "offline"
    with pytest.raises(RuntimeError, match="requires job.wandb_mode=online"):
        wandb_util._init_wandb_rank0(config, None)
    init.assert_not_called()


def test_required_online_checks_actual_mode(monkeypatch, setup_wandb):
    config, _ = setup_wandb
    monkeypatch.setenv("COSMOS_REQUIRE_WANDB_ONLINE", "1")
    wandb_util.wandb.run.settings.mode = "offline"
    with pytest.raises(RuntimeError, match="did not create"):
        wandb_util._init_wandb_rank0(config, None)


def test_legacy_auth_fallback_unchanged(monkeypatch, setup_wandb):
    config, init = setup_wandb
    monkeypatch.delenv("COSMOS_REQUIRE_WANDB_ONLINE", raising=False)
    init.side_effect = [RuntimeError("returned error 401"), None]
    wandb_util._init_wandb_rank0(config, None)
    assert [call.kwargs["mode"] for call in init.call_args_list] == ["online", "offline"]
