"""Nsight attribution must keep process identity and overlapping CUDA work."""

import sqlite3

import pytest

from tools.analyze_tp_pp_nsys import analyze


def test_process_local_launches_and_memory_ops_do_not_hide_idle_time(tmp_path):
    path = tmp_path / "capture.sqlite"
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
            (start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER, demangledName INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, globalPid INTEGER, bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER, globalPid INTEGER, bytes INTEGER);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        """
    )
    scale = 10**9
    left, right = 101 << 24, 202 << 24
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "gemm"), (2, "ncclKernel_AllReduce")])
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
        [(scale, 2 * scale, left + 1, 1), (3 * scale, 4 * scale, left + 1, 2), (scale, 2 * scale, right + 1, 1)],
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)",
        [
            (10 * scale, 20 * scale, left, 0, 1, 1),
            (40 * scale, 50 * scale, left, 0, 2, 2),
            (100 * scale, 200 * scale, right, 1, 1, 1),
        ],
    )
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?)", (20 * scale, 35 * scale, left, 1024))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?,?,?,?)", (30 * scale, 45 * scale, left, 2048))
    db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)", (0, 5 * scale, "cosmos.tp_pp.forward", None, left + 1))
    db.commit()
    db.close()
    report = analyze(path, phase_prefixes=["cosmos.tp_pp."])
    processes = {row["pid"]: row for row in report["processes"]}
    assert processes[101]["no_kernel_seconds"] == pytest.approx(20)
    assert processes[101]["memory_ops_without_kernel_seconds"] == pytest.approx(20)
    assert processes[101]["no_cuda_activity_seconds"] == 0
    assert processes[101]["memcpy_bytes_for_overlapping_ops"] == 1024
    assert processes[202]["memory_ops_without_kernel_seconds"] == 0
    phase = report["phases"][0]
    assert phase["pid"] == 101
    assert phase["launched_kernel_union_seconds"] == pytest.approx(20)
    assert phase["launched_non_nccl_kernel_union_seconds"] == pytest.approx(10)
    assert phase["launched_nccl_only_kernel_seconds"] == pytest.approx(10)
