"""FA3 numerical, sample isolation and compilation contracts for offline action inference."""

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.mot.small_action_attention import video_varlen_attend, video_varlen_mask


def test_causal_metadata_preserves_sample_order():
    ids = torch.tensor([0, 0, 0, 1, 1])
    mask = video_varlen_mask(ids, ids, 2, causal=True)
    assert mask.causal
    assert mask.cumulative_q.tolist() == mask.cumulative_kv.tolist() == [0, 3, 5, 5]
    assert mask.kv_order.tolist() == list(range(5))


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(1)
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_flash3_matches_masked_sdpa_and_compiles(causal):
    torch.manual_seed(42)
    qi = torch.tensor([0, 0, 0, 1, 1], device="cuda")
    # Noncausal KV is deliberately interleaved between modalities and samples.
    ki = qi if causal else torch.tensor([0, 0, 1, 0, 1, 1, 0], device="cuda")
    q = torch.randn(len(qi), 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(len(ki), 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = video_varlen_mask(qi, ki, 2, causal=causal)
    visible = qi[:, None] == ki[None, :]
    if causal:
        visible &= torch.arange(len(qi), device="cuda")[:, None] >= torch.arange(len(ki), device="cuda")[None]
    expected = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            attn_mask=visible[None, None],
            enable_gqa=True,
        )[0]
        .transpose(0, 1)
        .flatten(1)
    )
    with torch.inference_mode():
        actual = video_varlen_attend(q, k, v, mask)
        compiled = torch.compile(video_varlen_attend, fullgraph=True, dynamic=False)
        optimized = compiled(q, k, v, mask)
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
        torch.testing.assert_close(optimized, actual, atol=0, rtol=0)
        # Changing another sample's values must have no effect on sample zero.
        changed = v.clone()
        changed[ki == 1] = 100
        isolated = compiled(q, k, changed, mask)
        torch.testing.assert_close(isolated[qi == 0], actual[qi == 0], atol=0, rtol=0)
