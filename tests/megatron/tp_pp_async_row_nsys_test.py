# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sqlite3

import pytest

from tools.analyze_tp_pp_async_row_nsys import analyze, intervals


def test_overlap_unions_intervals_and_clips_to_range():
    result = intervals(
        [(0, 12_000, "compute"), (5_000, 15_000, "ncclDevKernel"), (7_000, 11_000, "compute")],
        2_000,
        14_000,
    )
    assert result["seconds"]["compute_only"] == pytest.approx(3e-6)
    assert result["seconds"]["overlap"] == pytest.approx(7e-6)
    assert result["seconds"]["nccl_only"] == pytest.approx(2e-6)


def make_db(path, *, overlap=True, missing=False):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INTEGER, start INTEGER, end INTEGER);"
            "CREATE TABLE StringIds(id INTEGER, value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER, "
            "demangledName INTEGER, deviceId INTEGER);"
        )
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "ncclDevKernel"), (2, "gemm")])
        for device in range(4):
            global_pid = (device + 1) << 24
            for mode_index, mode in enumerate(("baseline", "delayed")):
                if missing and device == 3 and mode == "delayed":
                    continue
                start = (device * 2 + mode_index) * 100_000
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
                    (f"async_row_probe.{mode}.step1", global_pid + 7, start, start + 90_000),
                )
                for item in range(3):
                    left = start + 1_000 + item * 20_000
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                        (left, left + 6_000, global_pid, 2, device),
                    )
                    offset = 3_000 if overlap and mode == "delayed" else 8_000
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                        (left + offset, left + offset + 5_000, global_pid, 1, device),
                    )


def test_analyze_requires_more_overlap_than_baseline(tmp_path):
    path = tmp_path / "profile.sqlite"
    make_db(path)
    report = analyze(path, warmups=1, steps=2)
    assert report["passed"]
    assert len(report["ranges"]) == 8
    assert all(item["mean_overlap_seconds"]["delayed"] > 0 for item in report["per_device"])


def test_analyze_fails_closed_for_no_overlap_or_missing_rank(tmp_path):
    path = tmp_path / "none.sqlite"
    make_db(path, overlap=False)
    assert not analyze(path, warmups=1, steps=2)["passed"]
    path = tmp_path / "missing.sqlite"
    make_db(path, missing=True)
    with pytest.raises(ValueError, match="Missing"):
        analyze(path, warmups=1, steps=2)
