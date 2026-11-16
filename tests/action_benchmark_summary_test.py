"""Measured-window compilation checks include the warmup boundary."""

import json

import pytest

from tools.summarize_action_benchmark import summarize


@pytest.mark.parametrize("graphs,stable", [([2, 2, 2], True), ([1, 2, 2], False), ([1, 1, 2], False)])
def test_compile_stability_includes_first_measured_update(tmp_path, graphs, stable):
    for rank in range(8):
        rows = [
            dict(
                iteration=i,
                measured=i > 0,
                seconds=2.0,
                global_samples=32,
                peak_allocated_bytes=1024,
                peak_reserved_bytes=2048,
                compile_unique_graphs=count,
                compile_graph_breaks=0,
            )
            for i, count in enumerate(graphs)
        ]
        (tmp_path / f"rank_{rank}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = summarize(tmp_path)
    assert result["compile_stable_in_measured_window"] is stable
    assert result["total_samples"] == 64
    assert result["samples_per_second"] == 16
