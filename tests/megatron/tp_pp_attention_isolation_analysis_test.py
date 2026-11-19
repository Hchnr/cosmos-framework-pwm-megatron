import hashlib

import pytest
import torch

from tools.analyze_tp_pp_attention_isolation import analyze, compare
from tools.trace_tp_pp_attention_isolation import Recorder


def record(value, *, rank=0, tp_rank=0, tp_size=1, shape=None, scope=None):
    return {
        "forward": 1,
        "layer": 30,
        "tag": "attention_call1",
        "kind": "query",
        "rank": rank,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "dtype": str(value.dtype),
        "stored_sha256": hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        "shape": list(value.shape) if shape is None else shape,
        "scope": {"kind": "full"} if scope is None else scope,
        "value": value,
    }


def write_record(directory, value):
    directory.mkdir(exist_ok=True)
    torch.save(value, directory / f"layer30.attention_call1.query.rank{value['rank']}.pt")


def test_compares_actual_head_shards_and_validates_all_ranks(tmp_path):
    value = torch.arange(40, dtype=torch.float32).reshape(5, 4, 2)
    native, tp = tmp_path / "native", tmp_path / "tp"
    write_record(native, record(value))
    write_record(tp, record(value[:, :2], rank=2, tp_rank=0, tp_size=2))
    write_record(tp, record(value[:, 2:], rank=3, tp_rank=1, tp_size=2))
    rows = analyze(native, tp)["rows"]
    assert [row["shard_axis"] for row in rows] == [1, 1]
    assert [row["maximum_absolute_error"] for row in rows] == [0, 0]
    (tp / "layer30.attention_call1.query.rank3.pt").unlink()
    with pytest.raises(ValueError, match="Missing or duplicated TP shards"):
        analyze(native, tp)


def test_aligns_sampled_and_full_tensors_with_different_row_limits():
    value = torch.arange(240, dtype=torch.float32).reshape(20, 6, 2)
    sample = torch.cat((value[:5], value[-5:]))
    shard = value[:, 3:]
    shard_sample = torch.cat((shard[:3], shard[-3:]))
    reference = record(sample, shape=list(value.shape), scope={"kind": "first_last_dim0", "rows_each_end": 5})
    actual = record(
        shard_sample,
        rank=1,
        tp_rank=1,
        tp_size=2,
        shape=list(shard.shape),
        scope={"kind": "first_last_dim0", "rows_each_end": 3},
    )
    result = compare(reference, actual)
    assert result["scope"] == {"first_last_dim0": 3}
    assert result["maximum_absolute_error"] == 0
    assert compare(record(value), actual)["maximum_absolute_error"] == 0


def test_replicated_and_changed_values_are_not_conflated():
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    changed = value.clone()
    changed[1, 1] += 1
    result = compare(record(value), record(changed, rank=1, tp_rank=1, tp_size=2))
    assert result["shard_axis"] is None
    assert result["maximum_absolute_error"] == 1
    assert result["relative_l2"] == pytest.approx(1 / (30**0.5))


def test_rejects_nonfinite_and_unmappable_values():
    value = torch.ones(5, 4, 2)
    with pytest.raises(ValueError, match="Cannot uniquely map"):
        compare(record(value), record(torch.ones(5, 3, 2), tp_size=2))
    bad = value.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare(record(value), record(bad))


def test_rejects_wrong_forward_and_malformed_sample():
    value = torch.ones(10, 4, 2)
    bad = record(value)
    bad["forward"] = 2
    with pytest.raises(ValueError, match="forward index"):
        compare(record(value), bad)
    bad = record(value[:3], shape=list(value.shape), scope={"kind": "first_last_dim0", "rows_each_end": 2})
    with pytest.raises(ValueError, match="Invalid sampled trace shape"):
        compare(record(value), bad)


def test_reads_actual_recorder_files_when_only_native_requires_sampling(tmp_path):
    value = torch.arange(160, dtype=torch.float32).reshape(20, 4, 2)
    native, tp = tmp_path / "native", tmp_path / "tp"
    Recorder(native, rank=0, tp_rank=0, tp_size=1, max_elements=100, rows=3).save(value, 30, "attention_call1", "query")
    for rank in range(2):
        Recorder(tp, rank=rank, tp_rank=rank, tp_size=2, max_elements=100, rows=3).save(
            value[:, rank * 2 : (rank + 1) * 2], 30, "attention_call1", "query"
        )
    rows = analyze(native, tp)["rows"]
    assert len(rows) == 2
    assert all(row["scope"] == {"first_last_dim0": 3} for row in rows)
    assert all(row["maximum_absolute_error"] == 0 for row in rows)
