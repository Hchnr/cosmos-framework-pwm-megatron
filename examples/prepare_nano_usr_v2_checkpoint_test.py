# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.scripts._merge_understanding_dcp import _allocate, _tensor_metadata
from examples.prepare_nano_usr_v2_checkpoint import prepare


def test_restart_preserves_training_state_and_resets_catalog(tmp_path: Path) -> None:
    generation, understanding, output = (tmp_path / name for name in ("generation", "understanding", "output"))
    embedding = "net.language_model.model.embed_tokens.weight"
    attention = "net.language_model.model.layers.0.self_attn.q_proj.weight"
    video = "net.time_embedder.weight"
    visual = "net.language_model.visual.weight"
    original = {embedding: torch.ones(2), attention: torch.ones(2), video: torch.full((2,), 3.0)}
    replacement = {embedding: torch.full((2,), 7.0), attention: torch.full((2,), 8.0), visual: torch.ones(4)}
    dcp.save(original, checkpoint_id=generation / "model")
    dcp.save(replacement, checkpoint_id=understanding / "model")
    for component, state in (
        ("trainer", {"iteration": 12600}),
        ("optim", {"moment": torch.tensor([2.0])}),
        ("scheduler", {"last_epoch": 12600}),
    ):
        dcp.save(state, checkpoint_id=generation / component)
    (generation / "dataloader").mkdir()
    (generation / "dataloader" / "rank_0.pkl").write_bytes(b"old catalog cursor")

    report = prepare(generation, understanding, output)
    actual = _allocate(_tensor_metadata(output / "model"))
    dcp.load(actual, checkpoint_id=output / "model")
    assert torch.equal(actual[embedding], replacement[embedding])
    assert torch.equal(actual[attention], replacement[attention])
    assert torch.equal(actual[video], original[video])
    assert visual not in actual
    assert report["iteration"] == 12600
    assert report["full_tensors_verified"] == 3
    assert not (output / "dataloader").exists()
    for component in ("optim", "scheduler", "trainer"):
        for source in (generation / component).iterdir():
            assert (output / component / source.name).read_bytes() == source.read_bytes()
    assert (generation / "dataloader" / "rank_0.pkl").read_bytes() == b"old catalog cursor"
    source_model = _allocate(_tensor_metadata(generation / "model"))
    dcp.load(source_model, checkpoint_id=generation / "model")
    assert torch.equal(source_model[embedding], original[embedding])
    with pytest.raises(FileExistsError):
        prepare(generation, understanding, output)


def test_restart_rejects_unexpected_nonvisual_weights(tmp_path: Path) -> None:
    generation, understanding = tmp_path / "generation", tmp_path / "understanding"
    dcp.save({"net.language_model.model.embed_tokens.weight": torch.ones(1)}, checkpoint_id=generation / "model")
    dcp.save({"net.unexpected.weight": torch.ones(1)}, checkpoint_id=understanding / "model")
    with pytest.raises(ValueError, match="Unexpected non-visual"):
        prepare(generation, understanding, tmp_path / "output")
    assert not (tmp_path / "output").exists()
