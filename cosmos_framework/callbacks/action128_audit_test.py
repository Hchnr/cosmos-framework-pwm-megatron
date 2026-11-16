import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.action128_audit import Action128Audit, _scalars


def test_scalar_metadata_keeps_independent_projection_and_source_ids():
    assert _scalars([[torch.tensor([0])], [torch.tensor([8])]]) == [0, 8]
    assert _scalars([torch.tensor([19]), 25]) == [19, 25]
    assert _scalars([["a/1/0"], ["b/2/1"]]) == ["a/1/0", "b/2/1"]


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_audit_checks_gradients_without_mutation(bad_gradient):
    net = torch.nn.Module()
    net.register_parameter("und", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    net.register_parameter("mlp_moe_gen", torch.nn.Parameter(torch.ones(2)))
    net.register_parameter("mlp_moe_gen_action", torch.nn.Parameter(torch.ones(2)))
    net.mlp_moe_gen.grad = torch.tensor([3.0, 4.0])
    net.mlp_moe_gen_action.grad = torch.tensor([float("nan") if bad_gradient else 5.0, 12.0])
    before = net.mlp_moe_gen.grad.clone()
    audit = Action128Audit()
    if bad_gradient:
        with pytest.raises(RuntimeError, match="Non-finite"):
            audit.on_after_backward(SimpleNamespace(net=net))
    else:
        audit.on_after_backward(SimpleNamespace(net=net))
        assert audit.gradient_norms == {"UND": 0.0, "VIDEO": 5.0, "ACTION": 13.0}
    torch.testing.assert_close(net.mlp_moe_gen.grad, before)


def test_audit_head_snapshot_detects_updates_and_preserves_contract(tmp_path):
    net = torch.nn.Module()
    net.action2llm = torch.nn.Linear(2, 3, dtype=torch.bfloat16)
    net.llm2action = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    net.action_normalization = SimpleNamespace(
        contract_digest=torch.arange(16, dtype=torch.uint8), vision_spatial_position_mode="canvas"
    )
    audit = Action128Audit()
    audit.output, audit.rank = tmp_path, 0
    model = SimpleNamespace(net=net)
    audit._write_head_state(model, 0, "start")
    with torch.no_grad():
        net.action2llm.weight.add_(1)
    audit._write_head_state(model, 1, "end")
    before = json.loads((tmp_path / "start_iter_000000000_rank_0.json").read_text())
    after = json.loads((tmp_path / "end_iter_000000001_rank_0.json").read_text())
    assert before["contract_digest"] == after["contract_digest"]
    assert before["heads"]["action2llm.weight"] != after["heads"]["action2llm.weight"]
    assert before["heads"]["llm2action.weight"] == after["heads"]["llm2action.weight"]


@pytest.mark.parametrize("bad", [False, True])
def test_audit_inspects_actual_network_spatial_inputs(bad):
    module = SimpleNamespace(config=SimpleNamespace(vision_spatial_position_mode="action128_seam"))
    packed = SimpleNamespace(
        vision_spatial_position_mode="action128_seam",
        spatial_seams_vision=[1.25],
        vision=SimpleNamespace(token_shapes=[(1, 3, 2)], sequence_indexes=torch.arange(6)),
        position_ids=torch.tensor([[0.0] * 6, [1.25, 1.25, 0.25, 0.25, -0.75, -0.75], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    audit = Action128Audit()
    if bad:
        packed.position_ids[1].abs_()
        with pytest.raises(AssertionError):
            audit._audit_spatial_inputs(module, (packed,), {})
    else:
        audit._audit_spatial_inputs(module, (packed,), {})
        assert audit.spatial_evidence["items"][0]["negative_height_tokens"] == 2
