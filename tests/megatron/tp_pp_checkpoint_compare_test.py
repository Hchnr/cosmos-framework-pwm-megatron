"""Exact archive checks must still reject changed moments and nonfinite values."""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.compare_tp_pp_runtime_checkpoints import compare, tensor_difference


def save_pair(tmp_path, left, right):
    paths = [tmp_path / label for label in ("left", "right")]
    for path, value in zip(paths, (left, right), strict=True):
        dcp.save({"net.weight": torch.tensor([1.0, 2.0]), "optimizer.test.exp_avg": value}, checkpoint_id=path)
        (path / ".save_complete").touch()
        (path / "manifest.json").write_text(json.dumps({"iteration": 1}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"parameters": {"net.weight": {"optimizer_fields": ["state"]}}}))
    return *paths, bundle


def test_compare_rejects_modified_moments_with_identical_model(tmp_path):
    args = save_pair(tmp_path, torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.3]))
    report = compare(*args)
    assert not report["passed"]
    rows = {row["name"]: row for row in report["tensors"]}
    assert rows["net.weight"]["passed"]
    assert not rows["optimizer.test.exp_avg"]["passed"]


@pytest.mark.parametrize("dtype,value", [(torch.float32, float("nan")), (torch.bfloat16, float("inf"))])
def test_identical_archives_cannot_hide_nonfinite_optimizer_state(tmp_path, dtype, value):
    tensor = torch.tensor([value, 0.0], dtype=dtype)
    report = compare(*save_pair(tmp_path, tensor, tensor))
    assert not report["passed"]
    failed = next(row for row in report["tensors"] if not row["passed"])
    assert failed["nonfinite"]


def test_flat_bucket_windows_prevent_a_large_matrix_hiding_a_changed_small_tensor():
    reference = torch.cat((torch.ones(1 << 20), torch.full((1 << 20,), 1e-6)))
    changed = reference.clone()
    changed[1 << 20 :] *= 2
    arguments = {"relative_tolerance": 1e-4, "absolute_tolerance": 2e-6, "exact": False}
    assert not tensor_difference(reference, changed, **arguments)["passed"]
    assert tensor_difference(reference, changed, **arguments, local_windows=False)["passed"]
