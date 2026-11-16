"""Small-expert shape, numerical attention, routing and gradient contracts."""

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.action_tower import prepare_joint_metadata
from cosmos_framework.model.generator.mot.action_tower_test import tiny_net
from cosmos_framework.model.generator.mot.small_action_tower import (
    ActionBlock,
    ActionShape,
    action_shape,
    attend,
    compact_pack,
    small_layer_forward,
)


@pytest.mark.parametrize("heads,kv,dim", [(32, 8, 128), (12, 4, 128), (32, 8, 48)])
def test_attention_matches_explicit_gqa(heads, kv, dim):
    torch.manual_seed(42)
    q = torch.randn(3, heads, dim, requires_grad=True)
    k = torch.randn(7, kv, dim, requires_grad=True)
    v = torch.randn(7, kv, dim, requires_grad=True)
    mask = torch.arange(7)[None, :] <= torch.tensor([2, 4, 6])[:, None]
    actual = attend(q, k, v, mask)
    kr = k.repeat_interleave(heads // kv, dim=1)
    vr = v.repeat_interleave(heads // kv, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, kr) * dim**-0.5
    scores = scores.masked_fill(~mask[None], float("-inf"))
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vr).flatten(1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("action_head_dim", [8, 16])
def test_separate_streams_bidirectional_gradients_and_sample_isolation(active, action_head_dim):
    torch.manual_seed(42)
    net = tiny_net("two_tower")
    layer = net.language_model.model.layers[0]
    # Frozen UND; GEN remains trainable.
    for name, p in layer.named_parameters():
        p.requires_grad_("_moe_gen" in name)
    shape = ActionShape(32, 64, 4, 2, action_head_dim, (0,))
    if active:
        layer.action_block_moe_gen_action = ActionBlock(
            shape, type(layer.input_layernorm), 1e-6, base_heads=2, base_dim=16
        )
        if action_head_dim == 16:
            assert len(layer.action_block_moe_gen_action.adapters) == 0
            assert layer.action_block_moe_gen_action.q_proj.out_features == 64
            assert layer.action_block_moe_gen_action.o_proj.out_features == 32
    x = torch.randn(11, 64)
    p = sequence_pack_from_packed_sequence(
        x,
        ["causal", "full", "causal", "full"],
        [2, 5, 1, 3],
        [7, 4],
        torch.tensor([0, 1, 7]),
        torch.tensor([2, 3, 4, 5, 6, 8, 9, 10]),
    )
    prepare_joint_metadata(p, torch.tensor([5, 6, 10]), "native")
    u, g = p["causal_seq"], p["full_only_seq"]
    pos = (
        from_und_gen_splits(torch.ones(len(u), 16), torch.ones(len(g), 16), p),
        from_und_gen_splits(torch.zeros(len(u), 16), torch.zeros(len(g), 16), p),
    )
    p = compact_pack(p, pos, torch.zeros(3, 11, dtype=torch.long), shape)
    for key in ("full_only_seq", "action_seq"):
        p[key] = p[key].detach().requires_grad_()
    out, _, _ = small_layer_forward(layer, p)
    assert out["action_seq"].shape == (3, 32)
    assert out["full_only_seq"].shape == (5, 64)
    assert not out["causal_seq"].requires_grad
    if active:
        ga = torch.autograd.grad(out["action_seq"].square().sum(), p["full_only_seq"], retain_graph=True)[0]
        gv = torch.autograd.grad(out["full_only_seq"].square().sum(), p["action_seq"], retain_graph=True)[0]
        assert ga.abs().sum() > 0 and gv.abs().sum() > 0
        loss = out["action_seq"].square().sum() + out["full_only_seq"].square().sum()
        loss.backward()
        for name, param in layer.action_block_moe_gen_action.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    else:
        assert out["action_seq"] is p["action_seq"]
    perturbed = dict(p)
    perturbed["action_seq"] = p["action_seq"].detach().clone()
    perturbed["action_seq"][-1] += 100
    other, _, _ = small_layer_forward(layer, perturbed)
    torch.testing.assert_close(other["full_only_seq"][:3], out["full_only_seq"][:3], atol=0, rtol=0)


def test_small_mrope_matches_qwen_interleaving():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    from cosmos_framework.model.generator.mot.small_action_tower import action_rope

    config = Qwen3VLTextConfig(
        hidden_size=1536,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=48,
        rope_theta=5000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [9, 8, 7], "mrope_interleaved": True},
    )
    rope = Qwen3VLTextRotaryEmbedding(config)
    positions = torch.tensor([[0, 1, 17, 15001], [0, 2, 5, 7], [0, 4, 9, 11]])
    expected = rope(torch.zeros(4, 1536), positions.unsqueeze(1))
    actual = action_rope(positions, torch.float32, 48)
    for x, y in zip(actual, expected, strict=True):
        # Reciprocal-power versus negative-power FP32 evaluation differs by
        # ~3e-5 at position 15001; well below BF16 rounding. Axis layout must match.
        torch.testing.assert_close(x, y.squeeze(0), atol=5e-5, rtol=5e-4)


def test_checkpoint_shape_mismatch_rejected():
    from types import SimpleNamespace

    from cosmos_framework.model.generator.mot.action_tower import validate_checkpoint_initialization

    key = "net.language_model.model.layers.0.action_block_moe_gen_action.q_proj.weight"
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_checkpoint_initialization({key: torch.empty(3, 4)}, {key: SimpleNamespace(size=(4, 4))})


def test_adapter_coordinate_pairs_and_domain_projection():
    from tools.prepare_small_action_checkpoint import adapter_matrix, indices

    a = adapter_matrix(128, 48, paired=True)
    ix = a.argmax(1)
    torch.testing.assert_close(ix[:24] + 64, ix[24:])
    assert torch.count_nonzero(a) == 48
    r, _, qi, ki, co, heads = indices("width_heads_1b")
    assert len(r) == 1536 and len(qi) == 1536 and len(ki) == 512
    # Each target GQA group takes three query heads from its matching source KV group.
    qheads = (qi.reshape(12, 128)[:, 0] // 128).reshape(4, 3)
    torch.testing.assert_close(qheads // 4, heads[:, None].expand(4, 3))


def test_width_only_keeps_full_attention_interface_without_adapters():
    from tools.prepare_small_action_checkpoint import indices

    shape = action_shape("width_only_1b")
    assert (shape.hidden, shape.intermediate, len(shape.layers)) == (1280, 3840, 36)
    with torch.device("meta"):
        block = ActionBlock(shape, torch.nn.RMSNorm, 1e-6)
    assert block.q_proj.weight.shape == (4096, 1280)
    assert block.k_proj.weight.shape == block.v_proj.weight.shape == (1024, 1280)
    assert block.o_proj.weight.shape == (1280, 4096)
    assert len(block.adapters) == 0
    residual, ffn, qi, ki, coordinates, kv_heads = indices("width_only_1b")
    assert len(residual) == 1280 and len(ffn) == 3840
    torch.testing.assert_close(qi, torch.arange(4096))
    torch.testing.assert_close(ki, torch.arange(1024))
    torch.testing.assert_close(coordinates, torch.arange(128))
    torch.testing.assert_close(kv_heads, torch.arange(8))
