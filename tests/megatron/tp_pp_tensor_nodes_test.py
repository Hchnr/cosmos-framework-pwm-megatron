# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json

import pytest
import torch

from tools.compare_tp_pp_tensor_nodes import FORMAT, compare_manifests, main


def manifest(root, tensors):
    root.mkdir(exist_ok=True)
    records = []
    for name, value in tensors.items():
        path = root / f"{name}.raw"
        data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        path.write_bytes(data)
        records.append(
            {
                "name": name,
                "path": path.name,
                "shape": list(value.shape),
                "dtype": str(value.dtype)[6:],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    path = root / "manifest.json"
    path.write_text(json.dumps({"format": FORMAT, "version": 1, "byte_order": "little", "tensors": records}))
    return path


def compare(a, b, **kwargs):
    return compare_manifests(a, b, relative_l2=1e-4, zero_max_abs=2e-6, min_cosine=0.999999, **kwargs)


def test_equal_and_threshold_failure(tmp_path):
    a = manifest(tmp_path / "ref", {"gate": torch.tensor([3.0, 4.0]), "up": torch.zeros(3)})
    b = manifest(tmp_path / "cand", {"gate": torch.tensor([3.0, 4.0]), "up": torch.full((3,), 1e-7)})
    report = compare(a, b)
    assert report["passed"] and len(report["nodes"]) == 2
    assert report["nodes"][1]["metrics"]["reference_norm_zero"] is True
    strict = compare(a, b, require_bitwise=True)
    assert not strict["passed"] and not strict["nodes"][1]["passed"]
    c = manifest(tmp_path / "bad", {"gate": torch.tensor([0.0, 4.0]), "up": torch.zeros(3)})
    assert not compare(a, c)["passed"]


def test_missing_extra_and_shape_mismatch_fail(tmp_path):
    a = manifest(tmp_path / "ref", {"gate": torch.ones(2), "up": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"gate": torch.ones(3), "down": torch.ones(2)})
    report = compare(a, b)
    assert not report["passed"] and report["missing"] == ["up"] and report["extra"] == ["down"]
    assert all(item["passed"] for item in report["unpaired_integrity"])
    assert "shape or dtype mismatch" in report["nodes"][0]["error"]
    (a.parent / "up.raw").write_bytes(b"corrupt")
    report = compare(a, b)
    assert any(not item["passed"] and item["name"] == "up" for item in report["unpaired_integrity"])


def test_hash_nonfinite_and_empty_manifest_fail_closed(tmp_path):
    a = manifest(tmp_path / "ref", {"x": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"x": torch.ones(2)})
    (b.parent / "x.raw").write_bytes(b"corrupt")
    assert "SHA-256 mismatch" in compare(a, b)["nodes"][0]["error"]
    c = manifest(tmp_path / "nan", {"x": torch.tensor([float("nan"), 1.0])})
    result = compare(a, c)
    assert not result["passed"] and result["nodes"][0]["metrics"]["status"] == "nonfinite"
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"format": FORMAT, "version": 1, "byte_order": "little", "tensors": []}))
    assert not compare(empty, b)["passed"]


def test_bad_paths_duplicates_and_symlinks_are_rejected(tmp_path):
    a = manifest(tmp_path / "ref", {"x": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"x": torch.ones(2)})
    data = json.loads(b.read_text())
    data["tensors"][0]["path"] = "../ref/x.raw"
    b.write_text(json.dumps(data))
    assert not compare(a, b)["passed"]
    data["tensors"][0]["path"] = "link.raw"
    (b.parent / "link.raw").symlink_to(a.parent / "x.raw")
    b.write_text(json.dumps(data))
    assert not compare(a, b)["passed"]
    data["tensors"].append(data["tensors"][0])
    b.write_text(json.dumps(data))
    assert not compare(a, b)["passed"]


def test_explicit_policy_validation_and_tensor_size_limit(tmp_path):
    a = manifest(tmp_path / "ref", {"x": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"x": torch.ones(2)})
    assert not compare(a, b, max_tensor_bytes=4)["passed"]
    with pytest.raises(ValueError, match="limits"):
        compare_manifests(a, b, relative_l2=float("nan"), zero_max_abs=0, min_cosine=1)


def test_cli_outputs_machine_readable_failure(tmp_path):
    a = manifest(tmp_path / "ref", {"x": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"y": torch.ones(2)})
    output = tmp_path / "result.json"
    result = main(
        [
            "--reference",
            str(a),
            "--candidate",
            str(b),
            "--output",
            str(output),
            "--relative-l2",
            "1e-4",
            "--zero-max-abs",
            "2e-6",
            "--min-cosine",
            "0.999999",
        ]
    )
    assert result == 1 and json.loads(output.read_text())["missing"] == ["x"]
    invalid_output = tmp_path / "invalid.json"
    result = main(
        [
            "--reference",
            str(a),
            "--candidate",
            str(b),
            "--output",
            str(invalid_output),
            "--relative-l2",
            "nan",
            "--zero-max-abs",
            "2e-6",
            "--min-cosine",
            "0.9",
        ]
    )
    assert result == 1 and json.loads(invalid_output.read_text())["passed"] is False
