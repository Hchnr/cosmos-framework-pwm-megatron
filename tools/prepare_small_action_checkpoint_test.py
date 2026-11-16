from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.prepare_small_action_checkpoint import load_source_state


def _save(path: Path, state):
    dcp.save(state, storage_writer=dcp.FileSystemWriter(path / "model"))


def test_training_weights_preserved_and_only_missing_sound_filled(tmp_path):
    source, fallback = tmp_path / "source", tmp_path / "fallback"
    _save(source, {"net.weight": torch.tensor([2.0]), "net_ema.weight": torch.tensor([9.0])})
    _save(fallback, {"net.weight": torch.tensor([1.0]), "net.sound_modality_embed": torch.tensor([3.0])})
    state, keys = load_source_state(source, fallback)
    assert set(state) == {"net.weight", "net.sound_modality_embed"}
    assert state["net.weight"].item() == 2.0
    assert state["net.sound_modality_embed"].item() == 3.0
    assert keys == ["net.sound_modality_embed"]


def test_refuses_to_fill_missing_backbone(tmp_path):
    source, fallback = tmp_path / "source", tmp_path / "fallback"
    _save(source, {"net.weight": torch.tensor([2.0])})
    _save(fallback, {"net.missing_backbone": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="non-sound"):
        load_source_state(source, fallback)


def test_rejects_ema_only_source(tmp_path):
    _save(tmp_path, {"net_ema.weight": torch.tensor([2.0])})
    with pytest.raises(ValueError, match="No net"):
        load_source_state(tmp_path)
