"""Post-load fusion preserves the checkpoint and the three-tower equations."""

import copy
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.action_optimization import fuse_small_action_qkv
from cosmos_framework.model.generator.mot.inference_small_action_memory import SmallActionMemoryState
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_layer_forward


@pytest.mark.parametrize("compile_layer", [False, True])
def test_qkv_fusion_is_nonpersistent_and_matches_reference_with_und_reuse(compile_layer):
    torch.manual_seed(79)
    layer, shape = make_layer()
    layer.eval()
    reference = copy.deepcopy(layer)
    # Cover mixed bias/no-bias projections as well as the checkpoint's bias-free case.
    layer.self_attn.q_proj.bias = torch.nn.Parameter(torch.randn(layer.self_attn.q_proj.out_features))
    reference.self_attn.q_proj.bias = copy.deepcopy(layer.self_attn.q_proj.bias)
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    model = SimpleNamespace(
        training=False,
        parallel_dims=None,
        net=SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(small_action_shape=shape, layers=[layer]))
        ),
    )
    assert fuse_small_action_qkv(model) > 0
    assert layer.state_dict().keys() == before.keys()
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    run = torch.compile(small_layer_forward, backend="eager", fullgraph=True) if compile_layer else small_layer_forward
    memory = SmallActionMemoryState(1)
    und = torch.randn(3, 64)
    with torch.no_grad():
        for _ in range(2):
            pack = make_pack([(und, torch.randn(4, 64), torch.randn(2, 32))], shape)
            expected = small_layer_forward(reference, pack)[0]
            actual, _, stored = run(layer, pack, memory_value=memory.read_for_layer(0))
            if stored is not None:
                memory.write_for_layer(0, stored)
            for key in ("causal_seq", "full_only_seq", "action_seq"):
                torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
