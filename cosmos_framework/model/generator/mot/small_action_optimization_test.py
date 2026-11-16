"""Numerical contracts for packed and fixed-shape small action decoding."""

import copy

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, sequence_pack_from_packed_sequence
from cosmos_framework.model.generator.mot.action_tower import prepare_joint_metadata
from cosmos_framework.model.generator.mot.action_tower_test import tiny_net
from cosmos_framework.model.generator.mot.small_action_tower import (
    ActionBlock,
    ActionShape,
    action_rope,
    compact_pack,
    small_decoder_inputs,
    small_layer_forward,
)


def make_layer():
    layer = tiny_net("two_tower").language_model.model.layers[0]
    for name, parameter in layer.named_parameters():
        parameter.requires_grad_("_moe_gen" in name)
    shape = ActionShape(32, 64, 4, 2, 16, (0,))
    layer.action_block_moe_gen_action = ActionBlock(shape, type(layer.input_layernorm), 1e-6, base_heads=2, base_dim=16)
    return layer, shape


def make_pack(samples, shape, *, raw=False, outer_lengths=(None, None), **kwargs):
    und, gen, action, splits, lens, positions = [], [], [], [], [], []
    offset = 0
    for u, v, a in samples:
        total = len(u) + len(v) + len(a)
        und.extend(range(offset, offset + len(u)))
        gen.extend(range(offset + len(u), offset + total))
        action.extend(range(offset + len(u) + len(v), offset + total))
        splits.extend((len(u), len(v) + len(a)))
        lens.append(total)
        positions.append(torch.arange(total).expand(3, -1))
        offset += total
    x = torch.cat([torch.cat((u, v, torch.nn.functional.pad(a, (0, 32)))) for u, v, a in samples])
    pack = sequence_pack_from_packed_sequence(
        x,
        ["causal", "full"] * len(samples),
        splits,
        lens,
        torch.tensor(und),
        torch.tensor(gen),
        fixed_causal_len=outer_lengths[0],
        fixed_full_len=outer_lengths[1],
    )
    prepare_joint_metadata(pack, torch.tensor(action), "native")
    ids = torch.cat(positions, dim=1)
    if raw:
        return pack, ids
    rope = tuple(from_all_seq(t, pack) for t in action_rope(ids, x.dtype, 16))
    return compact_pack(pack, rope, ids, shape, **kwargs)


def sample_loss(out, samples):
    losses = []
    vo = ao = 0
    for _, v, a in samples:
        losses.append(
            out["full_only_seq"][vo : vo + len(v)].square().mean() + out["action_seq"][ao : ao + len(a)].square().mean()
        )
        vo += len(v)
        ao += len(a)
    return torch.stack(losses).mean()


def test_packing_matches_individual_sample_outputs_and_mean_gradients():
    torch.manual_seed(12)
    layer, shape = make_layer()
    separate = copy.deepcopy(layer)
    samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in ((2, 3, 2), (1, 2, 1))]
    packed, _, _ = small_layer_forward(layer, make_pack(samples, shape))
    outputs = [small_layer_forward(separate, make_pack([s], shape))[0] for s in samples]
    for key in ("causal_seq", "full_only_seq", "action_seq"):
        torch.testing.assert_close(packed[key], torch.cat([out[key] for out in outputs]), atol=2e-6, rtol=2e-5)
    sample_loss(packed, samples).backward()
    torch.stack([sample_loss(out, [s]) for out, s in zip(outputs, samples, strict=True)]).mean().backward()
    for (name, p), (_, ref) in zip(layer.named_parameters(), separate.named_parameters(), strict=True):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, ref.grad, atol=2e-6, rtol=2e-5, msg=name)
        else:
            assert p.grad is None and ref.grad is None


def test_fixed_padding_matches_eager_outputs_gradients_and_is_inert():
    torch.manual_seed(19)
    layer, shape = make_layer()
    ref = copy.deepcopy(layer)
    samples = [(torch.randn(2, 64), torch.randn(3, 64), torch.randn(2, 32))]
    p = make_pack(samples, shape, fixed_lengths=(8, 12, 6))
    out = small_layer_forward(layer, small_decoder_inputs(p))[0]
    expected = small_layer_forward(ref, make_pack(samples, shape))[0]
    for key in ("causal_seq", "full_only_seq", "action_seq"):
        assert torch.isfinite(out[key]).all()
        torch.testing.assert_close(out[key][: len(expected[key])], expected[key], atol=2e-6, rtol=2e-5)
    sample_loss(out, samples).backward()
    sample_loss(expected, samples).backward()
    for (name, param), (_, other) in zip(layer.named_parameters(), ref.named_parameters(), strict=True):
        if param.requires_grad:
            torch.testing.assert_close(param.grad, other.grad, atol=2e-6, rtol=2e-5, msg=name)
    changed = small_decoder_inputs(p)
    for key, count in zip(("causal_seq", "full_only_seq", "action_seq"), (2, 3, 2), strict=True):
        changed[key] = changed[key].clone()
        changed[key][count:] = 100
    perturbed = small_layer_forward(layer, changed)[0]
    for key in ("causal_seq", "full_only_seq", "action_seq"):
        torch.testing.assert_close(perturbed[key][: len(expected[key])], out[key][: len(expected[key])], atol=0, rtol=0)


@pytest.mark.parametrize("attention_backend", ["sdpa", "flex"])
def test_static_decoder_reuses_graph_across_real_lengths_and_sample_counts(attention_backend):
    torch.manual_seed(21)
    layer, shape = make_layer()
    layer.small_action_enabled = True
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    torch.compiler.reset()
    compiled = torch.compile(layer, backend=backend, fullgraph=True, dynamic=False)
    for layout in (((2, 3, 2),), ((1, 2, 1), (3, 4, 2)), ((3, 1, 1),)):
        samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in layout]
        pack = make_pack(samples, shape, fixed_lengths=(8, 12, 6), attention_backend=attention_backend)
        inputs = small_decoder_inputs(pack)
        out = compiled(inputs, None, None)[0]
        expected = layer(inputs, None, None)[0]
        for key in ("causal_seq", "full_only_seq", "action_seq"):
            torch.testing.assert_close(out[key], expected[key])
        sample_loss(out, samples).backward()
        layer.zero_grad(set_to_none=True)
    assert len(graphs) == 1
    torch.compiler.reset()


def test_fixed_capacity_overflow_fails_without_truncation():
    _, shape = make_layer()
    samples = [(torch.randn(2, 64), torch.randn(3, 64), torch.randn(2, 32))]
    with pytest.raises(ValueError, match="exceeding capacity"):
        make_pack(samples, shape, fixed_lengths=(8, 12, 1))


def test_fixed_whole_decoder_restores_outer_pack_and_gradients():
    torch.manual_seed(27)
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
    fixed = copy.deepcopy(model)
    fixed.small_action_fixed_lengths = (8, 12, 6)
    samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in ((2, 3, 2), (1, 2, 1))]
    pack, ids = make_pack(samples, shape, raw=True)
    padded, padded_ids = make_pack(samples, shape, raw=True, outer_lengths=(8, 18))
    expected, _ = model(pack, None, ids)
    actual, _ = fixed(padded, None, padded_ids)
    assert actual["causal_seq"].shape == (8, 64)
    assert actual["full_only_seq"].shape == (18, 64)
    for key in ("causal_seq", "full_only_seq"):
        real = len(expected[key])
        torch.testing.assert_close(actual[key][:real], expected[key], atol=2e-6, rtol=2e-5)
        assert torch.count_nonzero(actual[key][real:]) == 0
    actual["full_only_seq"][:8].square().sum().backward()
    expected["full_only_seq"].square().sum().backward()
    for (name, param), (_, ref) in zip(fixed.named_parameters(), model.named_parameters(), strict=True):
        if param.grad is not None or ref.grad is not None:
            torch.testing.assert_close(param.grad, ref.grad, atol=2e-5, rtol=2e-4, msg=name)


@pytest.mark.parametrize("warm_full_und", [False, True])
def test_checkpointed_whole_decoder_reuses_graph_across_pack_layouts(warm_full_und):
    """Exercise the real caller: unused outer RoPE packs must not reach AC."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    torch.manual_seed(27)
    model = tiny_net("two_tower").language_model.model
    shape = ActionShape(32, 64, 4, 2, 16, (0,))
    model.small_action_shape = shape
    model.small_action_fixed_lengths = (8, 12, 6)
    model.small_action_und_short_tokens = 4 if warm_full_und else 0
    model.norm_moe_gen_action = type(model.norm)(32, eps=1e-6)
    layer = model.layers[0]
    layer.small_action_enabled = True
    layer.action_block_moe_gen_action = ActionBlock(shape, type(layer.input_layernorm), 1e-6, base_heads=2, base_dim=16)
    reference = copy.deepcopy(model)
    model._small_action_warm_full_und = warm_full_und
    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    torch.compiler.reset()
    model.layers[0] = torch.compile(checkpoint_wrapper(layer), fullgraph=True, dynamic=False, backend=backend)
    try:
        layouts = (((2, 3, 2), (1, 2, 1)), ((1, 4, 1),), ((2, 2, 1), (2, 1, 2), (1, 3, 1)), ((1, 2, 1),))
        executed_buckets = []
        for layout in layouts:
            samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in layout]
            pack, ids = make_pack(samples, shape, raw=True, outer_lengths=(8, 18))
            ref_pack, ref_ids = make_pack(samples, shape, raw=True, outer_lengths=(8, 18))
            actual, _ = model(pack, None, ids)
            expected, _ = reference(ref_pack, None, ref_ids)
            executed_buckets.append(model._small_action_last_und_tokens)
            und_length = sum(u for u, _, _ in layout)
            gen_length = sum(v + a for _, v, a in layout)
            for key, length in (("causal_seq", und_length), ("full_only_seq", gen_length)):
                torch.testing.assert_close(actual[key][:length], expected[key][:length])
            actual["full_only_seq"][:gen_length].square().sum().backward()
            expected["full_only_seq"][:gen_length].square().sum().backward()
            for param, ref in zip(model.parameters(), reference.parameters(), strict=True):
                if param.grad is not None or ref.grad is not None:
                    torch.testing.assert_close(param.grad, ref.grad)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
        assert executed_buckets == ([8, 4, 8, 4] if warm_full_und else [8, 8, 8, 8])
        assert len(graphs) == (2 if warm_full_und else 1)
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("fixed_lengths", [(8, 12, 6), (128, 256, 128)])
def test_sparse_small_tower_matches_dense_outputs_and_gradients(fixed_lengths):
    torch.manual_seed(42)
    layer, shape = make_layer()
    reference = copy.deepcopy(layer)
    samples = [(torch.randn(u, 64), torch.randn(v, 64), torch.randn(a, 32)) for u, v, a in ((2, 3, 2), (1, 2, 1))]
    sparse = make_pack(samples, shape, fixed_lengths=fixed_lengths, attention_backend="flex")
    dense = make_pack(samples, shape, fixed_lengths=fixed_lengths)
    actual = small_layer_forward(layer, small_decoder_inputs(sparse))[0]
    expected = small_layer_forward(reference, small_decoder_inputs(dense))[0]
    for key in ("causal_seq", "full_only_seq", "action_seq"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
    sample_loss(actual, samples).backward()
    sample_loss(expected, samples).backward()
    for (name, param), (_, ref) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        if param.requires_grad:
            torch.testing.assert_close(param.grad, ref.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_sparse_head48_gqa_scaling_and_skipped_cross_sample_blocks():
    from cosmos_framework.model.generator.mot.small_action_attention import sample_block_mask, sparse_attend
    from cosmos_framework.model.generator.mot.small_action_tower import attend

    torch.manual_seed(43)
    ids = torch.arange(3).repeat_interleave(128)
    mask = sample_block_mask(ids, ids)
    assert mask.sparsity() > 50  # Cross-sample work is actually omitted.
    q = torch.randn(384, 4, 48, requires_grad=True)
    k = torch.randn(384, 2, 48, requires_grad=True)
    v = torch.randn(384, 2, 48, requires_grad=True)
    refs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    out = sparse_attend(q, k, v, mask)
    expected = attend(*refs, ids[:, None] == ids[None, :])
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-5)
    out.square().mean().backward()
    expected.square().mean().backward()
    for x, ref in zip((q, k, v), refs, strict=True):
        torch.testing.assert_close(x.grad, ref.grad, atol=2e-7, rtol=2e-4)


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(1)
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("attention_backend", ["sdpa", "flex", "flash3_video"])
@pytest.mark.parametrize("und_short_tokens", [0, 2, 4])
def test_cuda_bfloat16_inductor_forward_and_backward_match_eager(checkpointed, attention_backend, und_short_tokens):
    # Run only between distributed benchmarks: this is a numerical unit test,
    # not a one-GPU performance measurement.
    torch.manual_seed(33)
    layer, shape = make_layer()
    layer.small_action_enabled = True
    layer = layer.to(device="cuda", dtype=torch.bfloat16)
    reference = copy.deepcopy(layer)
    samples = [
        (
            torch.randn(u, 64, dtype=torch.bfloat16),
            torch.randn(v, 64, dtype=torch.bfloat16),
            torch.randn(a, 32, dtype=torch.bfloat16),
        )
        for u, v, a in ((2, 3, 2), (1, 2, 1))
    ]
    # Build BlockMasks on the actual device; their captured sample IDs must
    # share the Q/K/V device. Both paths use identical samples and noise.
    # make_pack's sequence metadata starts on CPU, then compact on CUDA below.
    raw, ids = make_pack(samples, shape, raw=True, outer_lengths=(8, 18))
    raw = torch.utils._pytree.tree_map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, raw)
    ids = ids.cuda()
    rope = tuple(from_all_seq(t, raw) for t in action_rope(ids, torch.bfloat16, 16))
    inputs = small_decoder_inputs(
        compact_pack(
            raw,
            rope,
            ids,
            shape,
            fixed_lengths=(8, 12, 6),
            attention_backend=attention_backend,
            und_short_tokens=und_short_tokens,
        )
    )
    reference_inputs = small_decoder_inputs(compact_pack(raw, rope, ids, shape, fixed_lengths=(8, 12, 6)))
    torch.compiler.reset()
    target = layer
    if checkpointed:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        target = checkpoint_wrapper(layer, preserve_rng_state=True)
    actual = torch.compile(
        target, fullgraph=True, dynamic=False, options={"triton.mix_order_reduction_split_size": 32}
    )(inputs, None, None)[0]
    expected = reference(reference_inputs, None, None)[0]
    for key, length in zip(("causal_seq", "full_only_seq", "action_seq"), (3, 5, 3), strict=True):
        assert torch.isfinite(actual[key]).all()
        torch.testing.assert_close(actual[key][:length], expected[key][:length], atol=0.02, rtol=0.03)
    sample_loss(actual, samples).backward()
    sample_loss(expected, samples).backward()
    for (name, param), (_, ref) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad, ref.grad, atol=0.003, rtol=0.04, msg=name)
        else:
            assert param.grad is None and ref.grad is None
    torch.compiler.reset()


@pytest.mark.parametrize("real_und,expected_capacity", [(2, 4), (5, 8)])
def test_short_und_bucket_preserves_outputs_gradients_and_long_caption_fallback(real_und, expected_capacity):
    torch.manual_seed(42)
    layer, shape = make_layer()
    reference = copy.deepcopy(layer)
    samples = [(torch.randn(real_und, 64), torch.randn(3, 64), torch.randn(2, 32))]
    packed = make_pack(samples, shape, fixed_lengths=(8, 12, 6), und_short_tokens=4)
    assert packed["causal_seq"].shape[0] == expected_capacity
    result = small_layer_forward(layer, small_decoder_inputs(packed))[0]
    expected = small_layer_forward(reference, make_pack(samples, shape, fixed_lengths=(8, 12, 6)))[0]
    for key, length in zip(("causal_seq", "full_only_seq", "action_seq"), (real_und, 3, 2), strict=True):
        torch.testing.assert_close(result[key][:length], expected[key][:length], rtol=2e-5, atol=2e-6)
    sample_loss(result, samples).backward()
    sample_loss(expected, samples).backward()
    for (name, param), (_, other) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        if param.requires_grad:
            torch.testing.assert_close(param.grad, other.grad, rtol=2e-5, atol=2e-6, msg=name)


@pytest.mark.parametrize("fixed_capacity", [False, True])
def test_video_varlen_metadata_preserves_real_sample_visibility(fixed_capacity):
    from cosmos_framework.model.generator.mot.small_action_attention import video_varlen_mask

    query_ids = torch.tensor([0, 0, 1, 1, -1])
    key_ids = torch.tensor([0, 1, -1, 0, 0, 1, -1])
    mask = video_varlen_mask(query_ids, key_ids, 2, fixed_capacity=fixed_capacity)
    assert mask.cumulative_q.tolist() == [0, 2, 4, 5]
    assert mask.cumulative_kv.tolist() == [0, 3, 5, 7]
    assert sorted(mask.kv_order.tolist()) == list(range(len(key_ids)))
    for sample in range(2):
        indexes = mask.kv_order[mask.cumulative_kv[sample] : mask.cumulative_kv[sample + 1]]
        assert indexes.tolist() == torch.where(key_ids == sample)[0].tolist()
    assert mask.max_q >= max(torch.diff(mask.cumulative_q)).item()
    assert mask.max_kv >= max(torch.diff(mask.cumulative_kv)).item()


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(1)
def test_fixed_flash3_metadata_reuses_graph_and_preserves_gradients():
    from cosmos_framework.model.generator.mot.small_action_attention import video_varlen_attend, video_varlen_mask
    from cosmos_framework.model.generator.mot.small_action_tower import attend

    torch.manual_seed(42)
    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    torch.compiler.reset()
    compiled = torch.compile(video_varlen_attend, fullgraph=True, dynamic=False, backend=backend)
    try:
        for lengths in ((128, 192), (200,), (64, 128, 96)):
            qids = torch.cat([torch.full((n,), i, device="cuda") for i, n in enumerate(lengths)])
            qids = torch.nn.functional.pad(qids, (0, 384 - len(qids)), value=-1)
            kids = torch.cat((torch.arange(len(lengths), device="cuda").repeat_interleave(16), qids))
            kids = torch.nn.functional.pad(kids, (0, 448 - len(kids)), value=-1)
            inputs = [
                torch.randn(n, heads, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for n, heads in ((384, 4), (448, 2), (448, 2))
            ]
            refs = [x.detach().clone().requires_grad_() for x in inputs]
            mask = video_varlen_mask(qids, kids, len(lengths), fixed_capacity=True)
            actual = compiled(*inputs, mask)
            visibility = (qids[:, None] >= 0) & (qids[:, None] == kids[None, :])
            visibility |= (qids[:, None] < 0) & (torch.arange(448, device="cuda")[None, :] == 0)
            expected = attend(*refs, visibility)
            real = sum(lengths)
            torch.testing.assert_close(actual[:real], expected[:real], atol=0.02, rtol=0.03)
            actual[:real].float().square().mean().backward()
            expected[:real].float().square().mean().backward()
            for value, reference in zip(inputs, refs, strict=True):
                torch.testing.assert_close(value.grad, reference.grad, atol=2e-5, rtol=0.04)
        assert len(graphs) == 1
    finally:
        torch.compiler.reset()
