# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import verify_tp_pp_async_row_connection_results as audit

ROOT = Path(__file__).resolve().parents[2]


def write_benchmark(root, backend, elapsed):
    root.mkdir(parents=True)
    batches = [
        {
            "microstep": i,
            "source": f"/inputs/{i}.pt",
            "payload_sha256": str(i) * 64,
            "samples": n,
            "image": False,
            "tokens": {"text": 10 + i, "vision": 20 + i, "action": 30 + i},
        }
        for i, n in enumerate((9, 11, 13))
    ]
    contract = {
        "backend": backend,
        "tp": 4 if backend == "tp" else 1,
        "pp": 1,
        "dp": 1 if backend == "tp" else 4,
        "precision": "bfloat16",
        "microsteps_per_update": 3,
        "warmup_updates": 2,
        "measured_updates": 5,
        "per_rank_microbatches": batches,
        "global_sample_count_matched": False,
        "numerical_acceptance": False,
        "checkpoint_writes": False,
    }
    logs = root / "training"
    if backend == "tp":
        logs /= "cosmos3_action/action128/tp4_microbatch"
    logs.mkdir(parents=True)
    for rank in range(4):
        (root / f"contract.rank{rank}.json").write_text(json.dumps(contract))
        rows = [{"iteration": 1201 + i, "update_seconds": elapsed + rank * 0.01 + i * 0.001} for i in range(7)]
        (logs / f"updates.rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return contract


def test_independent_rank_timing_and_fair_sample_counts(tmp_path):
    tp, fsdp = tmp_path / "tp", tmp_path / "fsdp"
    write_benchmark(tp, "tp", 8.0)
    write_benchmark(fsdp, "fsdp", 23.0)
    result = audit.independent_compare(tp, fsdp)
    assert result["tp"]["global_samples_per_update"] == 33
    assert result["fsdp"]["global_samples_per_update"] == 132
    assert result["tp"]["measured_mean_update_seconds"] == pytest.approx(8.034)
    assert result["tp_over_fsdp_sample_throughput"] == pytest.approx((33 / 8.034) / (132 / 23.034))


def test_invalid_rank_logs_or_contract_fail_closed(tmp_path):
    root = tmp_path / "tp"
    contract = write_benchmark(root, "tp", 8.0)
    contract["per_rank_microbatches"][0]["samples"] = 10
    (root / "contract.rank0.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="contracts differ"):
        audit.independent_side(root, "tp")
    write_benchmark(tmp_path / "other", "tp", 8.0)
    log = tmp_path / "other/training/cosmos3_action/action128/tp4_microbatch/updates.rank2.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[4]["update_seconds"] = float("nan")
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="invalid update"):
        audit.independent_side(tmp_path / "other", "tp")


def test_prepared_schema_checks_sample_count_and_tensor_buffer_bounds():
    class Buffer:
        dtype = "torch.int64"

        def numel(self):
            return 12

    schema = {
        "version": 1,
        "tensors": [
            {"dtype": "torch.int64", "shape": [4], "offset": 0},
            {"dtype": "torch.int64", "shape": [6], "offset": 4},
        ],
        "tree": {
            "dict": [
                ["sample_count", 2],
                ["loss_inputs", {"dict": [["is_image_batch", False]]}],
                [
                    "packed_sequence",
                    {
                        "class": "PackedSequence",
                        "fields": {
                            "sample_lens": {"list": [4, 6]},
                            "text_indexes": {"tensor": 0},
                            "vision": {"fields": {"sequence_indexes": {"tensor": 1}}},
                            "action": None,
                        },
                    },
                ],
            ]
        },
    }
    buffers = {"torch.int64": Buffer()}
    result = audit.prepared_metadata(json.dumps(schema).encode(), buffers)
    assert result == {"samples": 2, "image": False, "tokens": {"text": 4, "vision": 6, "action": 0}}
    schema["tree"]["dict"][0][1] = 3
    with pytest.raises(ValueError, match="sample count"):
        audit.prepared_metadata(json.dumps(schema).encode(), buffers)
    schema["tree"]["dict"][0][1] = 2
    schema["tensors"][1]["offset"] = 10
    with pytest.raises(ValueError, match="exceeds"):
        audit.prepared_metadata(json.dumps(schema).encode(), buffers)


def test_comparator_comes_from_fixed_absolute_snapshot(tmp_path, monkeypatch):
    root_decoy = tmp_path / "decoy.py"
    root_decoy.write_text("ORIGIN = 'wrong-root'\n")
    snapshot = tmp_path / "snapshot"
    tool = snapshot / "tools/compare_tp4_fsdp4_microbatch_benchmark.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("ORIGIN = 'fixed-snapshot'\n")
    digest = hashlib.sha256(tool.read_bytes()).hexdigest()
    monkeypatch.syspath_prepend(str(root_decoy.parent))
    module, reported = audit.load_snapshot_comparator(
        snapshot, {"files": {"tools/compare_tp4_fsdp4_microbatch_benchmark.py": digest}}
    )
    assert module.ORIGIN == "fixed-snapshot"
    assert Path(module.__file__).resolve() == tool.resolve()
    assert reported == digest
    tool.write_text("ORIGIN = 'tampered'\n")
    with pytest.raises(ValueError, match="queued SHA"):
        audit.load_snapshot_comparator(snapshot, {"files": {"tools/compare_tp4_fsdp4_microbatch_benchmark.py": digest}})


def test_snapshot_manifest_rejects_changed_or_added_python(tmp_path):
    source = tmp_path / "original.py"
    source.write_text("value = 1\n")
    manifest = {"files": {"original.py": audit.sha256(source)}}
    audit.verify_snapshot_files(tmp_path, manifest)
    added = tmp_path / "added.py"
    added.write_text("value = 2\n")
    with pytest.raises(ValueError, match="file set"):
        audit.verify_snapshot_files(tmp_path, manifest)
    added.unlink()
    source.write_text("value = 3\n")
    with pytest.raises(ValueError, match="file changed"):
        audit.verify_snapshot_files(tmp_path, manifest)


def test_full_offline_audit_detects_tampered_recorded_summary(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    comparator = snapshot / "tools/compare_tp4_fsdp4_microbatch_benchmark.py"
    comparator.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "tools/compare_tp4_fsdp4_microbatch_benchmark.py", comparator)
    root_comparator = tmp_path / "root_comparator.py"
    shutil.copy2(comparator, root_comparator)
    digest = audit.sha256(comparator)
    snap_manifest = {
        "snapshot": str(snapshot.resolve()),
        "files": {"tools/compare_tp4_fsdp4_microbatch_benchmark.py": digest},
    }
    baseline_manifest = {"files": {str(comparator): digest}}
    sequence = tmp_path / "sequence"
    cases, comparisons = {}, {}
    for connection in (1, 8):
        fsdp = sequence / f"conn{connection}_fsdp"
        write_benchmark(fsdp, "fsdp", 23.0)
        cases[fsdp.name] = {"returncode": 0, "started": 200}
        for mode in ("shared", "async"):
            tp = sequence / f"conn{connection}_tp_{mode}"
            write_benchmark(tp, "tp", 8.0)
            cases[tp.name] = {"returncode": 0, "started": 200}
            comparisons[f"conn{connection}_{mode}"] = audit.independent_compare(tp, fsdp)
    summary = {
        "completed": True,
        "failure": None,
        "performance_acceptance": False,
        "full_model_numerical_acceptance": False,
        "cases": cases,
        "comparisons": comparisons,
    }
    (sequence / "summary.json").write_text(json.dumps(summary))
    evidence = {
        "root_comparator": str(root_comparator),
        "root_comparator_sha256": digest,
        "snapshot_comparator": str(comparator),
        "snapshot_comparator_sha256": digest,
        "queue_launched": False,
        "observed_unix": 100,
    }
    monkeypatch.setattr(audit, "verify_prepared_payloads", lambda *args: [{"verified": True}])
    report = audit.audit(sequence, snapshot, snap_manifest, baseline_manifest, evidence)
    assert report["passed"] and report["root_sha_equal_at_both_observations"]
    summary["comparisons"]["conn8_async"]["tp"]["global_samples_per_second"] += 1
    (sequence / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Numeric mismatch"):
        audit.audit(sequence, snapshot, snap_manifest, baseline_manifest, evidence)
