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
                    passed=True,
                    optimizer_steps=3,
                    sharding_strategy="optim_grads",
                    torch_version="test-torch",
                    cuda_version="test-cuda",
                    megatron_core_version="0.20.0",
                    gpu="test-gpu",
                    short_und_capacity=0,
                    checkpoint_load_mode="warm_start",
                    initial_iteration=0,
                    dataloader_max_samples_per_batch=1,
                    dataloader_packing_pool_size=1,
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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("key", ["step_seconds", "mean_final_microbatch_losses", "update_data_wait_seconds"])
def test_summary_rejects_nonfinite_observations_including_warmup(live_runs, invalid, key):
    path = live_runs[1]
    run = json.loads(path.read_text())
    run[key][0] = invalid
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Invalid"):
        summarize(live_runs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("passed", False),
        ("throughput_qualification", False),
        ("optimizer_steps", 4),
        ("warmup_steps", 3),
        ("warmup_steps", -1),
        ("step_seconds", [3, 2, 0]),
        ("mean_final_microbatch_losses", [1, 0.9]),
        ("update_including_first_dataload_seconds", [3, 2]),
        ("update_data_wait_seconds", [0.1, 0.1, -1]),
        ("world_size", 1),
        ("backend", "unknown"),
        ("torch_version", "different-torch"),
        ("cuda_version", "different-cuda"),
        ("megatron_core_version", "different-megatron"),
        ("gpu", "different-gpu"),
        ("short_und_capacity", 5120),
        ("checkpoint_load_mode", "extend"),
        ("preserve_source_normalization", True),
        ("state_normalization_paths", {"root": "snapshot.json"}),
        ("initial_iteration", 1200),
        ("dataloader_max_samples_per_batch", 92),
        ("dataloader_packing_pool_size", 16),
    ],
)
def test_summary_rejects_failed_incomplete_or_incompatible_runs(live_runs, key, value):
    path = live_runs[1]
    run = json.loads(path.read_text())
    run[key] = value
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        summarize(live_runs)


def test_summary_accepts_zero_data_wait(live_runs):
    for path in live_runs:
        run = json.loads(path.read_text())
        run["update_data_wait_seconds"] = [0, 0, 0]
        path.write_text(json.dumps(run))
    assert summarize(live_runs)["backends"]["megatron"]["data_wait"]["mean_seconds"] == 0


def test_summary_rejects_mixed_megatron_strategies(live_runs):
    candidate = live_runs[1]
    path = candidate.with_name("another_megatron.json")
    run = json.loads(candidate.read_text())
    run["sharding_strategy"] = "optim_grads_params"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Incompatible Megatron sharding strategies"):
        summarize([*live_runs, path])


def test_pytorch_ignores_megatron_strategy_metadata(live_runs):
    path = live_runs[0]
    run = json.loads(path.read_text())
    run["sharding_strategy"] = "unused_by_pytorch"
    path.write_text(json.dumps(run))
    assert summarize(live_runs)["live_input_traces_bitwise_equal"]


@pytest.mark.parametrize("changed", ["loss", "seconds", "rank"])
def test_vlm_summary_rejects_invalid_rank_observations(tmp_path, changed):
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "artifacts/megatron/vlm_2b_1024_1_pytorch.json"
    paths = []
    for backend in ("pytorch", "megatron"):
        run = json.loads(source.read_text())
        run["backend"] = backend
        if backend == "megatron":
            if changed == "rank":
                run["ranks"][-1]["rank"] = 0
            else:
                run["ranks"][-1]["observations"][0][changed] = float("nan")
        path = tmp_path / f"{backend}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    with pytest.raises(ValueError):
        summarize(paths)
