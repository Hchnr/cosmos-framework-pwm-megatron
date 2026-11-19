# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sqlite3

import pytest

from tools.run_tp_pp_flash3_benchmark_sequence import count_flash_kernels, require_policy_reports, require_single_gate


def test_kernel_count_only_counts_actual_flash_forward_kernels(tmp_path):
    target = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(target)
    connection.executescript(
        "CREATE TABLE StringIds (id INTEGER, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER, demangledName INTEGER);"
        "INSERT INTO StringIds VALUES (1, 'flash::FlashAttnFwdSm90'), (2, 'flash::FlashAttnBwdSm90');"
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,1),(0,1),(1,1),(1,2);"
    )
    connection.commit()
    connection.close()
    assert count_flash_kernels(target) == {"0": 2, "1": 1}


def test_single_gpu_gate_requires_counts_and_both_gradient_checks():
    report = {
        "passed": True,
        "cases": {
            f"{mode}_{kind}": {"flash3_forward_cuda_kernel_count": count}
            for mode in ("eager", "inductor")
            for kind, count in (("full", 2), ("selective", 1))
        },
        "comparisons": {mode: {"passed": True} for mode in ("eager", "inductor")},
    }
    require_single_gate(report)
    report["cases"]["inductor_selective"]["flash3_forward_cuda_kernel_count"] = 2
    with pytest.raises(ValueError):
        require_single_gate(report)


def test_four_rank_policy_reports_must_show_selected_fa3_path(tmp_path):
    for rank in range(4):
        row = {
            "rank": rank,
            "selected_layers": 8,
            "failure": None,
            "policy_trace_counts": {"forward": 8, "tp_selected_layers": 8},
        }
        (tmp_path / f"flash3_selective_ac.rank{rank}.json").write_text(json.dumps(row))
    assert len(require_policy_reports(tmp_path, "tp", 8)) == 4
    path = tmp_path / "flash3_selective_ac.rank2.json"
    row = json.loads(path.read_text())
    row["policy_trace_counts"]["forward"] = 0
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        require_policy_reports(tmp_path, "tp", 8)


def test_baseline_fingerprint_covers_all_source_trees(tmp_path, monkeypatch):
    from tools import run_tp_pp_flash3_benchmark_sequence as sequence

    for folder in ("tools", "tests", "cosmos_framework"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "module.py").write_text(folder)
    monkeypatch.setattr(sequence, "BASE", tmp_path)
    before = sequence.baseline_fingerprints()
    assert set(before) == {f"{folder}/module.py" for folder in ("tools", "tests", "cosmos_framework")}
    (tmp_path / "cosmos_framework/module.py").write_text("changed")
    after = sequence.baseline_fingerprints()
    assert before["cosmos_framework/module.py"] != after["cosmos_framework/module.py"]
