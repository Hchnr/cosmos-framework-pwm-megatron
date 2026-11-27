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


def expected_contract(path, names):
    path.write_text(json.dumps({"format": "cosmos_tp_pp_expected_node_names", "version": 1, "names": names}))
    return path


def test_expected_names_detect_both_sides_missing_and_extras(tmp_path):
    a = manifest(tmp_path / "ref", {"gate": torch.ones(2), "unexpected": torch.ones(1)})
    b = manifest(tmp_path / "cand", {"gate": torch.ones(2), "unexpected": torch.ones(1)})
    contract = expected_contract(tmp_path / "expected.json", ["gate", "up"])
    # Existing behavior remains a successful comparison with explicitly unverified coverage.
    old = compare(a, b)
    assert old["passed"] is True and old["coverage"]["status"] == "unverified"
    result = compare(a, b, expected_names=contract)
    assert result["passed"] is False and result["coverage"]["status"] == "failed"
    assert result["coverage"]["reference_missing"] == result["coverage"]["candidate_missing"] == ["up"]
    assert result["coverage"]["reference_extra"] == result["coverage"]["candidate_extra"] == ["unexpected"]
    assert result["coverage"]["expected_manifest_sha256"] == hashlib.sha256(contract.read_bytes()).hexdigest()


def test_expected_names_exact_coverage_and_one_sided_discrepancy(tmp_path):
    a = manifest(tmp_path / "ref", {"gate": torch.ones(2), "up": torch.ones(1)})
    b = manifest(tmp_path / "cand", {"gate": torch.ones(2), "up": torch.ones(1)})
    contract = expected_contract(tmp_path / "expected.json", ["gate", "up"])
    result = compare(a, b, expected_names=contract)
    assert result["passed"] is True and result["coverage"]["status"] == "passed"
    b = manifest(tmp_path / "other", {"gate": torch.ones(2), "other": torch.ones(1)})
    result = compare(a, b, expected_names=contract)
    assert not result["passed"]
    assert result["coverage"]["reference_missing"] == result["coverage"]["reference_extra"] == []
    assert result["coverage"]["candidate_missing"] == ["up"] and result["coverage"]["candidate_extra"] == ["other"]


@pytest.mark.parametrize(
    "body",
    [
        '{"format":"cosmos_tp_pp_expected_node_names","version":1,"names":["x","x"]}',
        '{"format":"cosmos_tp_pp_expected_node_names","version":1,"names":[]}',
        '{"format":"cosmos_tp_pp_expected_node_names","version":true,"names":["x"]}',
        '{"format":"cosmos_tp_pp_expected_node_names","version":1,"names":[1]}',
        '{"format":"cosmos_tp_pp_expected_node_names","version":1,"names":["x"],"unknown":1}',
        '{"format":"cosmos_tp_pp_expected_node_names","version":1,"names":["x"],"names":["x"]}',
        "not json",
    ],
)
def test_invalid_expected_contract_fails_closed(tmp_path, body):
    a = manifest(tmp_path / "ref", {"x": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"x": torch.ones(2)})
    contract = tmp_path / "expected.json"
    contract.write_text(body)
    result = compare(a, b, expected_names=contract)
    assert not result["passed"] and result["coverage"]["status"] == "invalid" and result["errors"]


def test_expected_contract_absent_file_and_symlink_fail_closed(tmp_path):
    a = manifest(tmp_path / "ref", {"x": torch.ones(2)})
    b = manifest(tmp_path / "cand", {"x": torch.ones(2)})
    contract = expected_contract(tmp_path / "expected.json", ["x"])
    link = tmp_path / "link.json"
    link.symlink_to(contract)
    for path in (link, tmp_path / "missing.json"):
        result = compare(a, b, expected_names=path)
        assert not result["passed"] and result["coverage"]["status"] == "invalid" and result["errors"]


def test_cli_expected_names_detects_missing_node(tmp_path):
    a = manifest(tmp_path / "ref", {"x": torch.ones(1)})
    b = manifest(tmp_path / "cand", {"x": torch.ones(1)})
    contract = expected_contract(tmp_path / "expected.json", ["x", "y"])
    out = tmp_path / "out.json"
    code = main(
        [
            "--reference",
            str(a),
            "--candidate",
            str(b),
            "--output",
            str(out),
            "--relative-l2",
            "0",
            "--zero-max-abs",
            "0",
            "--min-cosine",
            "1",
            "--expected-names",
            str(contract),
        ]
    )
    assert code == 1 and json.loads(out.read_text())["coverage"]["reference_missing"] == ["y"]
