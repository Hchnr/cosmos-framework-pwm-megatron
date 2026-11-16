"""Exact reuse must follow changing noisy streams and isolate requests/CFG."""

import copy

import pytest
import torch

from cosmos_framework.model.generator.mot.inference_small_action_memory import SmallActionMemoryState
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_decoder_inputs, small_layer_forward


@pytest.mark.parametrize("compile_layer", [False, True])
def test_cached_und_matches_fresh_for_changing_noise_and_independent_branches(compile_layer):
    torch.manual_seed(71)
    layer, shape = make_layer()
    layer.eval()
    reference = copy.deepcopy(layer)
    run = torch.compile(small_layer_forward, backend="eager", fullgraph=True) if compile_layer else small_layer_forward
    caches = [SmallActionMemoryState(1), SmallActionMemoryState(1)]
    unds = [torch.randn(3, 64), torch.randn(2, 64)]
    with torch.no_grad():
        for _ in range(3):
            for und, memory in zip(unds, caches, strict=True):
                samples = [(und, torch.randn(4, 64), torch.randn(2, 32))]
                pack = small_decoder_inputs(make_pack(samples, shape))
                actual, _, stored = run(layer, pack, memory_value=memory.read_for_layer(0))
                if stored is not None:
                    memory.write_for_layer(0, stored)
                expected = small_layer_forward(reference, pack)[0]
                for key in ("causal_seq", "full_only_seq", "action_seq"):
                    torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5)
    assert caches[0].read_for_layer(0).und_key.shape[0] == 3
    assert caches[1].read_for_layer(0).und_key.shape[0] == 2
    assert SmallActionMemoryState(1).read_for_layer(0).und_key is None


def test_cache_rejects_training_or_grad_enabled():
    layer, shape = make_layer()
    pack = make_pack([(torch.randn(2, 64), torch.randn(3, 64), torch.randn(1, 32))], shape)
    with pytest.raises(ValueError, match="eval and no_grad"):
        small_layer_forward(layer, pack, memory_value=SmallActionMemoryState(1).read_for_layer(0))


@pytest.mark.parametrize("reuse_und", [False, True])
def test_whole_decoder_reuses_metadata_and_keeps_all_layers_exact(reuse_und):
    from cosmos_framework.model.generator.mot.action_tower_test import tiny_net
    from cosmos_framework.model.generator.mot.small_action_tower import ActionBlock, ActionShape

    torch.manual_seed(73)
    model = tiny_net("two_tower", num_hidden_layers=2).language_model.model
    shape = ActionShape(32, 64, 4, 2, 16, (0, 1))
    model.small_action_shape = shape
    model.small_action_fixed_lengths = None
    model.norm_moe_gen_action = type(model.norm)(32, eps=1e-6)
    for layer in model.layers:
        layer.small_action_enabled = True
        layer.action_block_moe_gen_action = ActionBlock(
            shape, type(layer.input_layernorm), 1e-6, base_heads=2, base_dim=16
        )
    model.eval()
    reference = copy.deepcopy(model)
    und = torch.randn(3, 64)
    memory = SmallActionMemoryState(2, reuse_und=reuse_und)
    with torch.no_grad():
        for _ in range(3):
            samples = [(und, torch.randn(4, 64), torch.randn(2, 32))]
            pack, ids = make_pack(samples, shape, raw=True)
            expected = reference(pack, None, ids)[0]
            actual = model(pack, None, ids, memory=memory)[0]
            for key in ("causal_seq", "full_only_seq"):
                torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5)
    assert memory.compact_metadata is not None
    assert all((value.und_key is not None) == reuse_und for value in memory.layers)
