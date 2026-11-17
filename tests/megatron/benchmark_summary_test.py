"""Prevent live throughput claims from mismatched or incomplete input traces."""

import json

import pytest

from tools.summarize_megatron_benchmarks import summarize


@pytest.fixture
def live_runs(tmp_path):
    paths = []
    for backend, times in (("pytorch", [3, 2, 2]), ("megatron", [2, 1, 1])):
        path = tmp_path / f"live_{backend}.json"
        path.write_text(
            json.dumps(
                dict(
                    backend=backend,
                    scope="test live",
                    excludes=["checkpoint save I/O"],
                    token_capacities=[512, 8192, 128],
                    gradient_accumulation=3,
                    seed_reset_after_initialization=42,
                    live_data=True,
                    checkpoint="same",
                    fresh_optimizer_state=True,
                    warmup_steps=1,
                    world_size=2,
                    content_trace_enabled=True,
                    step_seconds=times,
                    median_seconds=times[-1],
                    peak_allocated_bytes=100,
                    update_including_first_dataload_seconds=times,
                    update_data_wait_seconds=[0.1, 0.1, 0.1],
                    mean_final_microbatch_losses=[1.0, 0.9, 0.8],
                )
            )
        )
        for rank in range(2):
            path.with_name(f"{path.stem}_inputs_rank_{rank}.json").write_text(
                json.dumps(
                    [
                        dict(
                            iteration=micro // 3,
                            content_sha256=f"rank{rank}_batch{micro}",
                            resume_rng_sha256=f"rng{micro}",
                            global_id=micro,
                        )
                        for micro in range(9)
                    ]
                )
            )
        paths.append(path)
    return paths


def test_live_summary_includes_first_dataload_and_all_observations(live_runs):
    result = summarize(live_runs)
    assert result["whole_update_mean_throughput_gain_fraction"] == 1.0
    assert result["input_trace_microbatches_per_rank"] == 9
    assert result["live_input_traces_bitwise_equal"]
    assert result["backends"]["pytorch"]["whole_update_including_first_dataload"]["observations"] == 2


@pytest.mark.parametrize("changed", ["content_sha256", "resume_rng_sha256", "global_id"])
def test_live_summary_rejects_changed_data_or_worker_continuation(live_runs, changed):
    path = live_runs[1].with_name(f"{live_runs[1].stem}_inputs_rank_1.json")
    trace = json.loads(path.read_text())
    trace[-1][changed] = "changed"
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="traces differ"):
        summarize(live_runs)


def test_live_summary_rejects_missing_microbatch(live_runs):
    path = live_runs[1].with_name(f"{live_runs[1].stem}_inputs_rank_1.json")
    path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
    with pytest.raises(ValueError, match="Incomplete live input trace"):
        summarize(live_runs)
