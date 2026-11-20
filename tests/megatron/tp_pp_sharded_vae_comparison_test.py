# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import pytest

from tools.run_tp4_sharded_vae_comparison import (
    SequenceAdapter,
    fingerprint_path,
    peak_memory,
    source_contract,
    verify_logical_state,
)


def test_fingerprint_detects_same_size_content_and_file_coverage_changes(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "one.pt").write_bytes(b"abc")
    initial = fingerprint_path(weights)
    (weights / "one.pt").write_bytes(b"xyz")
    assert fingerprint_path(weights) != initial
    (weights / "one.pt").write_bytes(b"abc")
    assert fingerprint_path(weights) == initial
    (weights / "two.pt").write_bytes(b"x")
    assert fingerprint_path(weights) != initial


def setup_checkpoints(tmp_path):
    source, tp4, tp2 = (tmp_path / name for name in ("source", "tp4", "tp2"))
    for path in (source, tp4, tp2):
        path.mkdir()
    for bundle, topology in ((tp4, (4, 1)), (tp2, (2, 2))):
        (bundle / ".save_complete").touch()
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "cosmos_tp_pp_canonical",
                    "format_version": 1,
                    "complete": True,
                    "source": str(source),
                    "tp": topology[0],
                    "pp": topology[1],
                    "parameters": {"weight": {}},
                }
            )
        )
    for component in ("trainer", "scheduler", "dataloader"):
        for root in (source, tp2 / "logical_state"):
            path = root / component
            path.mkdir(parents=True)
            (path / "part").write_bytes(component.encode())
    return source, tp4, tp2


def test_canonical_source_and_logical_state_contract(tmp_path):
    source, tp4, tp2 = setup_checkpoints(tmp_path)
    assert source_contract(tp4, tp2, source)["tp4_parameters"] == 1
    assert verify_logical_state(tp2, source) == 3
    (tp2 / "logical_state/dataloader/part").write_bytes(b"changed")
    with pytest.raises(ValueError, match="logical state differs"):
        verify_logical_state(tp2, source)
    (tp2 / "logical_state/dataloader/part").write_bytes(b"dataloader")
    (tp2 / "logical_state/dataloader/extra").touch()
    with pytest.raises(ValueError, match="coverage differs"):
        verify_logical_state(tp2, source)
    manifest = json.loads((tp4 / "manifest.json").read_text())
    manifest["source"] = str(tmp_path / "other")
    (tp4 / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible"):
        source_contract(tp4, tp2, source)


def test_tp_and_fsdp_commands_use_same_contract_and_different_entry(tmp_path):
    sequence = SimpleNamespace(TP2="tp2", TP4="tp4", TOML="recipe", SPOOL="spool", ENV={}, PYTHON="python")
    args = ("--backend", "tp", "--updates", 7, "--warmup-updates", 2)
    tp = SequenceAdapter(sequence, tmp_path, tmp_path / "qual", tmp_path / "report", "tp").distributed(
        "tools/run_tp4_fsdp4_raw_microbatch_benchmark.py", *args
    )
    fsdp = SequenceAdapter(sequence, tmp_path, tmp_path / "qual", tmp_path / "report", "fsdp").distributed(
        "tools/run_tp4_fsdp4_raw_microbatch_benchmark.py", *args
    )
    assert tp[-len(args) :] == [str(value) for value in args] == fsdp[-len(args) :]
    assert str(tmp_path / "tools/run_tp4_sharded_vae_raw_benchmark.py") in tp
    assert str(tmp_path / "tools/run_tp4_fsdp4_raw_microbatch_benchmark.py") in fsdp


def test_weight_fingerprint_rejects_symlink(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    target = tmp_path / "actual"
    target.write_bytes(b"x")
    (root / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        fingerprint_path(root)


@pytest.mark.parametrize("backend", ["tp", "fsdp"])
def test_peak_memory_keeps_warmup_and_measurement_separate(tmp_path, backend):
    directory = tmp_path / "training"
    if backend == "tp":
        directory /= "cosmos3_action/action128/tp4_microbatch"
    directory.mkdir(parents=True)
    for rank in range(4):
        values = [1000, 900, 10 + rank, 11 + rank, 12 + rank, 13 + rank, 14 + rank]
        (directory / f"updates.rank{rank}.jsonl").write_text(
            "".join(json.dumps({"peak_cuda_allocated_bytes": value}) + "\n" for value in values)
        )
    result = peak_memory(tmp_path, backend)
    assert result["3"] == {"all_updates_peak_allocated_bytes": 1000, "measured_updates_peak_allocated_bytes": 17}
