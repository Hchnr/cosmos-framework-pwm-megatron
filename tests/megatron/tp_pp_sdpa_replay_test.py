import hashlib

import pytest
import torch

from tools.replay_tp_pp_sdpa_capture import attend, run_case


def save(directory, tag, value, rank, size):
    directory.mkdir(parents=True, exist_ok=True)
    stored = value.detach().contiguous().cpu()
    digest = hashlib.sha256(stored.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    record = {
        "layer": 0,
        "tag": "attention_call1",
        "kind": tag,
        "rank": rank,
        "tp_rank": rank,
        "tp_size": size,
        "scope": {"kind": "full"},
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "stored_sha256": digest,
        "forward": 0,
        "value": stored,
    }
    torch.save(record, directory / f"layer00.attention_call1.{tag}.rank{rank}.pt")


def make_capture(tmp_path):
    torch.manual_seed(9)
    native, tp = tmp_path / "native", tmp_path / "tp"
    q, k, v = torch.randn(3, 8, 4), torch.randn(7, 4, 4), torch.randn(7, 4, 4)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[0, -1] = False
    out = attend(q, k, v, mask)
    dy = torch.randn_like(out)
    for kind, value in zip(
        ("query", "key", "value", "mask", "output", "output_dgrad"),
        (q, k, v, mask, out, dy),
        strict=True,
    ):
        save(native, kind, value, 0, 1)
    for rank in range(4):
        lq, lk, lv = (
            q[:, rank * 2 : (rank + 1) * 2].contiguous(),
            k[:, rank : rank + 1].contiguous(),
            v[:, rank : rank + 1].contiguous(),
        )
        local = attend(lq, lk, lv, mask)
        for kind, value in zip(
            ("query", "key", "value", "mask", "output", "output_dgrad"),
            (lq, lk, lv, mask, local, dy[:, rank * 8 : (rank + 1) * 8].contiguous()),
            strict=True,
        ):
            save(tp, kind, value, rank, 4)
    return native, tp


def test_common_inputs_and_native_cotangent_are_compared(tmp_path):
    native, tp = make_capture(tmp_path)
    report = run_case(native, tp, layer=0, call=1, device="cpu")
    assert report["native_forward_matches_capture"]["exact"]
    assert len(report["rank_results"]) == 4
    for rank in report["rank_results"]:
        assert all(rank["qkv_input_exact"]) and rank["mask_exact"]
        assert rank["local_forward_matches_capture"]["exact"]
        assert set(rank["full_vs_shard_vjp"]) == {"dq", "dk", "dv"}
        assert rank["captured_tp_dy_vs_native_dy"]["exact"]


@pytest.mark.parametrize("field", ["query", "mask"])
def test_different_common_input_or_mask_cannot_be_qualified(tmp_path, field):
    native, tp = make_capture(tmp_path)
    path = tp / f"layer00.attention_call1.{field}.rank0.pt"
    item = torch.load(path, weights_only=False)
    if field == "query":
        item["value"][0, 0, 0] += 1
    else:
        item["value"][0, 0] = False
    item["stored_sha256"] = hashlib.sha256(item["value"].reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    torch.save(item, path)
    with pytest.raises(ValueError, match="input or mask"):
        run_case(native, tp, layer=0, call=1, device="cpu")
