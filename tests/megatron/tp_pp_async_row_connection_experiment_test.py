# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from tools.run_tp_pp_async_row_connection_experiment import (
    benchmark_launch,
    require_conn8_overlap,
    require_terminal_queue,
)


class Guards:
    @staticmethod
    def benchmark_args(backend, folder, steps):
        return ["--backend", backend, "--output", str(folder), "--updates", str(steps)]


def test_terminal_prerequisite_allows_expected_no_overlap_failure():
    assert require_terminal_queue({"returncode": 1}) == 1
    with pytest.raises(ValueError, match="terminal"):
        require_terminal_queue({"completed": False})


def test_no_benchmark_without_connection8_measured_overlap():
    with pytest.raises(ValueError, match="did not demonstrate"):
        require_conn8_overlap({"cases": {"conn8_profile": {"overlap": {"passed": False}}}})
    require_conn8_overlap({"cases": {"conn8_profile": {"overlap": {"passed": True}}}})


@pytest.mark.parametrize("mode,backend", [("tp_shared", "tp"), ("tp_async", "tp"), ("fsdp", "fsdp")])
def test_benchmark_retains_four_ranks_seven_updates_and_matching_backend(mode, backend, tmp_path):
    actual, command = benchmark_launch(Guards, mode, tmp_path)
    assert actual == backend
    assert "--nproc-per-node=4" in command
    assert command[command.index("--backend") + 1] == backend
    assert command[command.index("--updates") + 1] == "7"
    assert ("--shared-input-reduction" in command) == (mode == "tp_async")
    assert any("async_row_reduction.py" in part for part in command) == (mode == "tp_async")


def test_unknown_benchmark_mode_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Unexpected"):
        benchmark_launch(Guards, "unknown", tmp_path)
