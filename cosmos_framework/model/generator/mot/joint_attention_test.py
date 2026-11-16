# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small independent math oracle; safe under torchrun with pytest-env disabled."""

import os

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.joint_attention import (
    build_joint_block_mask,
    joint_attention,
    prepare_joint_attention,
)


@pytest.fixture(autouse=True)
def rank_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def make_pack(device, *, multiple_und=False, fixed=True, single=False):
    lens = [17, 63, 29, 71]
    modes = ["causal", "full", "causal", "full"]
    und_ids = list(range(17)) + list(range(80, 109))
    gen_ids = list(range(17, 80)) + list(range(109, 180))
    if single:
        lens, modes, und_ids, gen_ids = lens[:2], modes[:2], list(range(17)), list(range(17, 80))
    pack = sequence_pack_from_packed_sequence(
        torch.zeros(sum(lens), 8, device=device),
        modes,
        lens,
        [80] if single else [80, 100],
        torch.tensor(und_ids, device=device),
        torch.tensor(gen_ids, device=device),
        fixed_causal_len=128 if fixed else None,
        fixed_full_len=256 if fixed else None,
    )
    if multiple_und:
        # Helper-only edge: the production fixed packer requires paired splits.
        pack["_causal_seq_offsets"] = torch.tensor([0, 7, 17, 46], dtype=torch.int32, device=device)
    return pack


def oracle_mask(pack):
    # Independent explicit construction from source sample lengths/segment lengths.
    nu = get_und_seq(pack).shape[0]
    n = nu + get_gen_seq(pack).shape[0]
    m = torch.zeros(n, n, dtype=torch.bool, device=get_und_seq(pack).device)
    offsets = pack["_causal_seq_offsets"].tolist()
    for a, b in zip(offsets, offsets[1:]):
        m[a:b, a:b] = torch.ones(b - a, b - a, dtype=torch.bool, device=m.device).tril()
    m[nu : nu + 63, :17] = True
    m[nu : nu + 63, nu : nu + 63] = True
    ureal, greal = 17, 63
    if pack["sample_offsets"].numel() > 2:
        m[nu + 63 : nu + 134, 17:46] = True
        m[nu + 63 : nu + 134, nu + 63 : nu + 134] = True
        ureal, greal = 46, 134
    pad = torch.tensor(list(range(ureal, nu)) + list(range(nu + greal, n)), device=m.device, dtype=torch.long)
    m[pad, pad] = True
    return m


@pytest.mark.parametrize("multiple_und", [False, True])
@pytest.mark.parametrize("stream", ["all", "und", "gen"])
def test_block_mask_matches_independent_oracle(multiple_und, stream):
    pack = make_pack("cpu", multiple_und=multiple_und)
    mask = build_joint_block_mask(pack, query_stream=stream)
    expected = oracle_mask(pack)
    expected = expected[:128, :128] if stream == "und" else expected[128:] if stream == "gen" else expected
    q = torch.arange(expected.shape[0])[:, None]
    k = torch.arange(expected.shape[1])[None, :]
    actual = mask.mask_mod(0, 0, q, k)
    assert torch.equal(actual, expected)
    # Every nonempty oracle block must be scheduled; a full block must be fully visible.
    for i in range(mask.kv_num_blocks.shape[-1]):
        partial = set(mask.kv_indices[0, 0, i, : mask.kv_num_blocks[0, 0, i]].tolist())
        full = set(mask.full_kv_indices[0, 0, i, : mask.full_kv_num_blocks[0, 0, i]].tolist())
        for j in range(mask.kv_indices.shape[-1]):
            block = expected[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            assert bool(block.any()) == (j in partial or j in full)
            if j in full:
                assert bool(block.all())


BACKENDS = ["split_flex_triton", "unified_flex_triton"]
if os.environ.get("COSMOS_TEST_JOINT_FLASH") == "1":
    BACKENDS += ["split_fa3_fork", "unified_fa3_grouped", "split_fa4", "unified_fa4_mask"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("frozen_und", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_cuda_output_and_source_gradients(backend, frozen_und, fixed, single):
    torch.manual_seed(420 + int(os.environ.get("LOCAL_RANK", "0")))
    pack = make_pack("cuda", fixed=fixed, single=single)
    nu, ng = get_und_seq(pack).shape[0], get_gen_seq(pack).shape[0]
    metadata = prepare_joint_attention(pack, backend)
    sources = [
        torch.randn(n, h, 128, device="cuda", dtype=torch.bfloat16).requires_grad_(not frozen_und or stream == 1)
        for h in (8, 2, 2)
        for stream, n in enumerate((nu, ng))
    ]
    refs = [t.detach().float().requires_grad_(t.requires_grad) for t in sources]
    qs, ks, vs = [sources[i : i + 2] for i in (0, 2, 4)]
    rq, rk, rv = [torch.cat(refs[i : i + 2]).transpose(0, 1).unsqueeze(0) for i in (0, 2, 4)]
    expected = F.scaled_dot_product_attention(rq, rk, rv, attn_mask=oracle_mask(pack), enable_gqa=True)
    result = joint_attention(*[from_und_gen_splits(*parts, pack) for parts in (qs, ks, vs)], metadata)
    actual = torch.cat((get_und_seq(result), get_gen_seq(result))).reshape(nu + ng, 8, 128)
    expected = expected.squeeze(0).transpose(0, 1)
    valid = torch.zeros(nu + ng, device="cuda", dtype=torch.bool)
    valid[: pack["_causal_indices"].numel()] = True
    valid[nu : nu + pack["_full_indices"].numel()] = True
    torch.testing.assert_close(actual[valid].float(), expected[valid], atol=2e-2, rtol=2e-2)
    do = torch.randn_like(expected) * valid[:, None, None]
    (actual.float() * do).sum().backward()
    (expected * do).sum().backward()
    for source, ref in zip(sources, refs):
        if not source.requires_grad:
            assert source.grad is None
            continue
        assert source.grad is not None and torch.isfinite(source.grad).all()
        error = (source.grad.float() - ref.grad).norm() / ref.grad.norm().clamp_min(1e-10)
        assert error < 0.03, (backend, frozen_und, error.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("backend", BACKENDS)
def test_two_tower_decoder_dispatch_and_backward(backend):
    from cosmos_framework.model.generator.mot.action_tower_test import tiny_net
    from cosmos_framework.model.generator.mot.attention import SplitInfo

    torch.manual_seed(42)
    base = tiny_net("two_tower", head_dim=128).cuda().to(torch.bfloat16)
    candidate = tiny_net("two_tower", head_dim=128).cuda().to(torch.bfloat16)
    candidate.load_state_dict(base.state_dict())
    pack = make_pack("cuda")
    pack = from_und_gen_splits(
        torch.randn(128, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(256, 64, device="cuda", dtype=torch.bfloat16),
        pack,
    )
    cos = from_und_gen_splits(
        torch.ones(128, 128, device="cuda", dtype=torch.bfloat16),
        torch.ones(256, 128, device="cuda", dtype=torch.bfloat16),
        pack,
    )
    sin = from_und_gen_splits(
        torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16),
        torch.zeros(256, 128, device="cuda", dtype=torch.bfloat16),
        pack,
    )
    reference_mask = SplitInfo([17, 63, 29, 71], ["causal", "full", "causal", "full"], [80, 100], 180)
    candidate_mask = SplitInfo([17, 63, 29, 71], ["causal", "full", "causal", "full"], [80, 100], 180)
    candidate_mask.joint_attention_metadata = prepare_joint_attention(pack, backend)
    ref, _, _ = base.language_model.model.layers[0](pack, reference_mask, (cos, sin))
    layer = candidate.language_model.model.layers[0]
    if os.environ.get("COSMOS_TEST_COMPILED") == "1":
        layer = torch.compile(layer, dynamic=False, fullgraph=True)
    got, _, _ = layer(pack, candidate_mask, (cos, sin))
    target = torch.randn(134, 64, device="cuda")
    a, b = get_gen_seq(got)[:134], get_gen_seq(ref)[:134]
    torch.testing.assert_close(a, b, atol=0.03, rtol=0.02)
    (a.float() * target).sum().backward()
    (b.float() * target).sum().backward()
    for (name, p), (_, r) in zip(candidate.named_parameters(), base.named_parameters()):
        if r.grad is not None:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            err = (p.grad.float() - r.grad.float()).norm() / r.grad.float().norm().clamp_min(1e-10)
            assert err < 0.04, (name, err.item())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("empty", ["_causal_indices", "_full_indices"])
def test_two_tower_empty_real_stream_is_explicitly_rejected(backend, empty):
    pack = make_pack("cpu")
    pack[empty] = torch.empty(0, dtype=torch.long)
    with pytest.raises(ValueError, match="nonempty real UND and GEN"):
        prepare_joint_attention(pack, backend)
