"""End-of-training validation must be allowed to log before W&B is finished."""

from unittest.mock import Mock

from cosmos_framework.utils import callback


def test_wandb_finishes_after_train_end_callbacks(monkeypatch):
    finish = Mock()
    monkeypatch.setattr(callback.wandb, "finish", finish)
    logger = callback.WandBCallback()
    logger.on_train_end(None, iteration=16)
    finish.assert_not_called()
    logger.on_app_end()
    finish.assert_called_once_with()
