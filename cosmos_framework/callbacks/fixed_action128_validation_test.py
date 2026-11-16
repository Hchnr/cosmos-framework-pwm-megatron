"""Validation must retain the same supervision and geometry as training."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.callbacks.fixed_action_validation import FixedActionValidation
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.test.action128_rope_test import packed_canvas  # noqa: F401
from cosmos_framework.data.test.action128_test import media_stub, source  # noqa: F401

SPECIAL = dict(eos_token_id=2, start_of_generation=3, end_of_generation=4)


@pytest.mark.parametrize("mode", ["canvas", "action128_seam"])
@pytest.mark.parametrize("after_forward", [False, True])
def test_validation_rebuild_preserves_positions_and_masks(packed_canvas, tmp_path, mode, after_forward):
    sample, geometry = packed_canvas(3, (16, 9), parts=True)
    geometry = replace(geometry, vision_spatial_position_mode=mode)
    batch = VFMListCollator(static_packing_config=geometry, special_tokens=SPECIAL).collate([sample])
    if after_forward:
        batch["video"] = [items[0].unsqueeze(0).float() / 127.5 - 1.0 for items in batch["video"]]
        batch["is_preprocessed"] = True
        batch["num_vision_items_per_sample"] = None
    fixture = tmp_path / "fixture.pt"
    torch.save({"reference_batches": [batch]}, fixture)
    cb = FixedActionValidation(str(fixture))
    cb.config = SimpleNamespace(
        dataloader_train=SimpleNamespace(static_packing_config=asdict(geometry)),
        job=SimpleNamespace(path_local=str(tmp_path)),
    )
    parameters = [
        (
            "language_model.model.layers.0." + name,
            torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=trainable),
        )
        for name, trainable in [("und", False), ("mlp_moe_gen", True), ("mlp_moe_gen_action", True)]
    ]
    model = SimpleNamespace(
        llm_special_tokens=SPECIAL,
        config=SimpleNamespace(action128_spec={"present": True}, tower_structure="three_tower"),
        net=SimpleNamespace(named_parameters=lambda: parameters),
    )
    cb.evaluate = Mock()
    cb.on_train_start(model)
    rebuilt = cb.batches[0]["packed_sequence"]
    torch.testing.assert_close(rebuilt.position_ids, batch["packed_sequence"].position_ids)
    for modality in ("action", "vision"):
        actual = getattr(rebuilt, modality).loss_valid_mask[0]
        expected = getattr(batch["packed_sequence"], modality).loss_valid_mask[0]
        torch.testing.assert_close(actual, expected)
        assert actual.dtype == torch.bool
    assert not rebuilt.action.loss_valid_mask[0].all()
    if after_forward:
        assert cb.batches[0]["video"][0].ndim == 5
        assert cb.batches[0]["is_preprocessed"] is True
    cb.evaluate.assert_called_once_with(model, 0)


def test_validation_multisample_metrics_and_rng_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Model(torch.nn.Module):
        def training_step(self, batch, seed):
            n = len(batch["sequence_plan"])
            values = torch.tensor([1.0, 3.0]) if n == 2 else torch.tensor([8.0])
            torch.rand(5)
            return {"flow_matching_loss_vision_per_instance": values}, values.mean()

    cb = FixedActionValidation("unused")
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    cb.batches = [{"sequence_plan": [None, None]}, {"sequence_plan": [None]}]
    cb.fixture_sha256 = "synthetic"
    reports = []
    cb._write = lambda name, report: reports.append(report)
    model = Model()
    rng = torch.random.get_rng_state().clone()
    cb.evaluate(model, 1)
    assert reports[0]["mean"]["loss"] == 4.0
    assert reports[0]["mean"]["flow_matching_loss_vision_per_instance"] == 4.0
    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
