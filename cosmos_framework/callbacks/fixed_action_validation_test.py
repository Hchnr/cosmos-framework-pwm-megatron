import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.callbacks.fixed_action_validation import FixedActionValidation


class FakeModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.child = torch.nn.Linear(1, 1).eval()
        self.fail = fail

    def training_step(self, batch, iteration):
        assert not self.training and not torch.is_grad_enabled()
        batch["value"].add_(1)
        value = torch.rand(()) + random.random() + np.random.random() + iteration * 0.001
        if self.fail:
            raise RuntimeError("evaluation failed")
        return {"flow_matching_loss_action": value}, value


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_preserves_rng_modes_and_input(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = FakeModel(fail)
    callback = FixedActionValidation("unused")
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    callback.batches = [{"value": torch.tensor(0.0)}]
    callback.fixture_sha256 = "test"
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            callback.evaluate(model, 0)
    else:
        callback.evaluate(model, 0)
        path = tmp_path / "fixed_action_validation/iter_000000000.json"
        first = path.read_text()
        callback.evaluate(model, 0)
        assert path.read_text() == first
        callback.evaluate(model, 100)
        later = tmp_path / "fixed_action_validation/iter_000000100.json"
        assert json.loads(later.read_text())["mean"] == json.loads(first)["mean"]
    assert model.training and not model.child.training
    assert callback.batches[0]["value"].item() == 0
    assert random.getstate() == state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1][1])
    assert np.random.get_state()[2:] == state[1][2:]
    assert torch.equal(torch.random.get_rng_state(), state[2])
