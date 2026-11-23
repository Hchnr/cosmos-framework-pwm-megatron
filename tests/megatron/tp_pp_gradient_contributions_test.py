# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import hashlib
import json
import math
import random
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tools.tp_pp_gradient_contributions import analyze, verify_completed_status


def _manifest_sha(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _analyze(manifest, reports):
    return analyze(manifest, reports, manifest_sha256=_manifest_sha(manifest))


def _metadata(rank, manifest):
    return {
        "rank": rank,
        "run_id": "30b152b0-2520-4e4d-b678-b13e9cda7d4e",
        "world_size": 2,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "gradient_scope": "pre_clip",
        "source_manifest_sha256": "a" * 64,
        "native_status_sha256": "b" * 64,
        "baseline_status_sha256": None,
        "bundle_manifest_sha256": _manifest_sha(manifest),
    }


def _row(name, reference, actual):
    difference = [
        actual_value - reference_value for reference_value, actual_value in zip(reference, actual, strict=True)
    ]
    reference_squared = sum(value * value for value in reference)
    difference_norm = math.sqrt(sum(value * value for value in difference))
    return {
        "update": 0,
        "name": name,
        "reference_squared_l2": reference_squared,
        "relative_l2": difference_norm / max(math.sqrt(reference_squared), 1e-20),
        "maximum_absolute_error": max(abs(value) for value in difference),
        "gradient_dtype": "torch.float32",
        "reference_dtype": "torch.float32",
        "model_parameter_dtype": "torch.float32",
    }


def _fixture():
    manifest = {
        "tp": 2,
        "pp": 1,
        "parameters": {
            "replicated": {"axis": None, "optimizer_fields": ["master"]},
            "sharded": {"axis": 0, "optimizer_fields": ["master"]},
        },
    }
    summary = [{"relative_l2": math.sqrt(0.5625 / 30.0)}]
    reports = [
        {
            **_metadata(0, manifest),
            "updates": 1,
            "gradient_summary": summary,
            "parameter_gradient_errors": [
                _row("replicated", [3, 4], [3, 4.5]),
                _row("sharded", [2, 0], [2.25, 0]),
            ],
        },
        {
            **_metadata(1, manifest),
            "updates": 1,
            "gradient_summary": summary,
            "parameter_gradient_errors": [
                _row("replicated", [3, 4], [3, 4.5]),
                _row("sharded", [0, 1], [0, 1.5]),
            ],
        },
    ]
    return manifest, reports


def test_weights_replicated_gradients_and_ranks_parameter_contributions():
    manifest, reports = _fixture()
    result = _analyze(manifest, reports)["updates"][0]
    assert result["reference_squared_l2"] == 30
    assert result["error_squared_l2"] == pytest.approx(0.5625)
    assert result["global_relative_l2"] == pytest.approx(math.sqrt(0.5625 / 30))
    contributions = {row["name"]: row["fraction_of_squared_error"] for row in result["parameter_contributions"]}
    assert contributions == pytest.approx({"replicated": 0.25 / 0.5625, "sharded": 0.3125 / 0.5625})


def test_zero_reference_shard_retains_its_error_contribution():
    manifest = {"tp": 2, "pp": 1, "parameters": {"x": {"axis": 0, "optimizer_fields": ["master"]}}}
    reports = [
        {
            **_metadata(rank, manifest),
            "updates": 1,
            "gradient_summary": [{"relative_l2": 0.25}],
            "parameter_gradient_errors": [_row("x", reference, actual)],
        }
        for rank, (reference, actual) in enumerate([([0], [0.25]), ([1], [1])])
    ]
    result = _analyze(manifest, reports)["updates"][0]
    assert result["reference_squared_l2"] == 1
    assert result["error_squared_l2"] == pytest.approx(0.0625)
    assert result["global_relative_l2"] == pytest.approx(0.25)


def test_zero_total_error_has_no_defined_contribution_percentages():
    manifest, reports = _fixture()
    for report in reports:
        report["gradient_summary"] = [{"relative_l2": 0.0}]
        for row in report["parameter_gradient_errors"]:
            row["relative_l2"] = 0.0
            row["maximum_absolute_error"] = 0.0
    result = _analyze(manifest, reports)["updates"][0]
    assert result["zero_total_error"] is True
    assert all(row["fraction_of_squared_error"] is None for row in result["parameter_contributions"])


def test_parameter_contributions_against_explicit_fp32_vectors():
    def fp32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    rng = random.Random(20261123)
    manifest, _ = _fixture()
    for _ in range(100):
        shared_reference = [fp32(rng.uniform(-3, 3)) for _ in range(4)]
        vectors = []
        for _rank in range(2):
            partition = [fp32(rng.uniform(-3, 3)) for _ in range(4)]
            rank_vectors = {}
            for name, reference in [("replicated", shared_reference), ("sharded", partition)]:
                actual = [fp32(value + rng.uniform(-1, 1)) for value in reference]
                rank_vectors[name] = (reference, actual)
            vectors.append(rank_vectors)

        reference_squared = error_squared = 0.0
        expected_contributions = {"replicated": 0.0, "sharded": 0.0}
        for rank in range(2):
            for name, (reference, actual) in vectors[rank].items():
                divisor = 2 if name == "replicated" else 1
                reference_squared += sum(value * value for value in reference) / divisor
                contribution = sum((a - b) ** 2 for a, b in zip(actual, reference, strict=True)) / divisor
                expected_contributions[name] += contribution
                error_squared += contribution
        reports = []
        for rank in range(2):
            rows = []
            for name, (reference, actual) in vectors[rank].items():
                row = _row(name, reference, actual)
                row["maximum_absolute_error"] = max(abs(fp32(a - b)) for a, b in zip(actual, reference, strict=True))
                rows.append(row)
            reports.append(
                {
                    **_metadata(rank, manifest),
                    "updates": 1,
                    "gradient_summary": [{"relative_l2": math.sqrt(error_squared / reference_squared)}],
                    "parameter_gradient_errors": rows,
                }
            )
        result = _analyze(manifest, reports)["updates"][0]
        assert result["error_squared_l2"] == pytest.approx(error_squared)
        observed = {item["name"]: item["squared_error"] for item in result["parameter_contributions"]}
        assert observed == pytest.approx(expected_contributions)


def test_rejects_incomplete_or_inconsistent_evidence():
    manifest, reports = _fixture()
    missing_norm = copy.deepcopy(reports)
    del missing_norm[0]["parameter_gradient_errors"][0]["reference_squared_l2"]
    with pytest.raises(ValueError, match="reference squared L2"):
        _analyze(manifest, missing_norm)
    incorrect_summary = copy.deepcopy(reports)
    incorrect_summary[1]["gradient_summary"] = [{"relative_l2": 0.1}]
    with pytest.raises(ValueError, match="do not reproduce"):
        _analyze(manifest, incorrect_summary)
    with pytest.raises(ValueError, match="rank 1 update 0"):
        _analyze(manifest, list(reversed(incorrect_summary)))
    missing_rank = reports[:1]
    with pytest.raises(ValueError, match="complete TP-only"):
        _analyze(manifest, missing_rank)
    incomplete_layout = copy.deepcopy(manifest)
    del incomplete_layout["parameters"]["replicated"]["axis"]
    with pytest.raises(ValueError, match="layouts"):
        _analyze(incomplete_layout, reports)
    duplicate_rank = [reports[0], copy.deepcopy(reports[0])]
    with pytest.raises(ValueError, match="rank coverage"):
        _analyze(manifest, duplicate_rank)
    wrong_precision = copy.deepcopy(reports)
    wrong_precision[1]["parameter_gradient_errors"][0]["gradient_dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError, match="non-FP32"):
        _analyze(manifest, wrong_precision)
    wrong_run = copy.deepcopy(reports)
    wrong_run[1]["run_id"] = "e7ad8f26-71e1-423f-9fc7-d3cb46ed1664"
    with pytest.raises(ValueError, match="different run"):
        _analyze(manifest, wrong_run)
    nonfinite = copy.deepcopy(reports)
    nonfinite[0]["parameter_gradient_errors"][0]["nonfinite"] = True
    with pytest.raises(ValueError, match="non-finite"):
        _analyze(manifest, nonfinite)
    small_but_inconsistent = copy.deepcopy(reports)
    for report in small_but_inconsistent:
        report["gradient_summary"] = [{"relative_l2": 1e-15}]
        for row in report["parameter_gradient_errors"]:
            row["relative_l2"] *= 1e-11
    with pytest.raises(ValueError, match="do not reproduce"):
        _analyze(manifest, small_but_inconsistent)


def test_status_binds_all_report_bytes_even_when_numerical_gate_failed():
    manifest, reports = _fixture()
    hashes = ["c" * 64, "d" * 64]
    status = {
        "phase": "tp_observation_completed",
        "source_input_posthash_passed": True,
        "recorded_math_and_default_rng_passed": False,
        "run_id": reports[0]["run_id"],
        "source_manifest_sha256": reports[0]["source_manifest_sha256"],
        "native_status_sha256": reports[0]["native_status_sha256"],
        "bundle_manifest_sha256": _manifest_sha(manifest),
        "report_sha256_by_rank": {"0": hashes[0], "1": hashes[1]},
    }
    verify_completed_status(status, reports, hashes, manifest_sha256=_manifest_sha(manifest))
    with pytest.raises(ValueError, match="bytes do not match"):
        verify_completed_status(status, reports, [hashes[1], hashes[0]], manifest_sha256=_manifest_sha(manifest))
    incomplete = {**status, "source_input_posthash_passed": False}
    with pytest.raises(ValueError, match="completed, posthashed"):
        verify_completed_status(incomplete, reports, hashes, manifest_sha256=_manifest_sha(manifest))


def test_cli_checks_pinned_status_and_report_bytes(tmp_path):
    manifest, reports = _fixture()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    report_paths = []
    hashes = {}
    for report in reports:
        path = tmp_path / f"tp.rank{report['rank']}.json"
        raw = json.dumps(report).encode()
        path.write_bytes(raw)
        report_paths.append(path)
        hashes[str(report["rank"])] = hashlib.sha256(raw).hexdigest()
    status = {
        "phase": "tp_observation_completed",
        "source_input_posthash_passed": True,
        "run_id": reports[0]["run_id"],
        "source_manifest_sha256": reports[0]["source_manifest_sha256"],
        "native_status_sha256": reports[0]["native_status_sha256"],
        "bundle_manifest_sha256": _manifest_sha(manifest),
        "report_sha256_by_rank": hashes,
    }
    status_raw = json.dumps(status).encode()
    status_path = tmp_path / "status.json"
    status_path.write_bytes(status_raw)
    output_path = tmp_path / "result.json"
    tool = Path(__file__).resolve().parents[2] / "tools/tp_pp_gradient_contributions.py"
    command = [
        sys.executable,
        "-B",
        str(tool),
        "--manifest",
        str(manifest_path),
        "--reports",
        *(str(path) for path in report_paths),
        "--status",
        str(status_path),
        "--status-sha256",
        hashlib.sha256(status_raw).hexdigest(),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output_path.read_text())
    assert result["report_sha256_by_rank"] == hashes
    assert result["updates"][0]["global_relative_l2"] == pytest.approx(math.sqrt(0.5625 / 30))
    report_paths[0].write_text(json.dumps(reports[0], indent=2))
    output_path.unlink()
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode != 0
    assert "rank report bytes do not match" in failure.stderr
    assert not output_path.exists()
