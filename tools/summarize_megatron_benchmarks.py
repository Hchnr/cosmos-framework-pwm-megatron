"""Summarize paired replay runs without extrapolating to production topology."""

import argparse
import json
import math
import statistics
from pathlib import Path


def _finite_values(values, name, *, positive=False):
    if not values or any(not math.isfinite(value) or (positive and value <= 0) for value in values):
        raise ValueError(f"Invalid {name}: expected nonempty finite{' positive' if positive else ''} observations")


def _latencies(observations, *, positive=True):
    _finite_values(observations, "latencies", positive=positive)
    return {
        "observations": len(observations),
        "median_seconds": statistics.median(observations),
        "mean_seconds": statistics.mean(observations),
        "p95_seconds": (
            statistics.quantiles(observations, n=100, method="inclusive")[94]
            if len(observations) > 1
            else observations[0]
        ),
        "max_seconds": max(observations),
    }


def summarize_vlm(paths, runs):
    reference = runs[0]
    settings = (
        "model_path",
        "compile",
        "full_activation_checkpointing",
        "gradient_accumulation",
        "optimizer",
        "learning_rate",
        "world_size",
        "replicas",
        "sequence_tokens",
        "image_grid_thw",
        "warmup_steps",
        "attention_implementation",
        "compute_dtype",
        "master_dtype",
        "gradient_dtype",
        "gradient_communication_dtype",
        "megatron_sharding_strategy",
        "frozen_vision_encoder",
        "frozen_mm_projector",
    )
    for run in runs:
        for key in settings:
            if run[key] != reference[key]:
                raise ValueError(f"Incompatible VLM benchmark setting: {key}")
        if len(run["ranks"]) != reference["world_size"]:
            raise ValueError("Missing rank observations")
        if sorted(rank["rank"] for rank in run["ranks"]) != list(range(run["world_size"])):
            raise ValueError("Missing or duplicate VLM ranks")
        for actual, expected in zip(run["ranks"], reference["ranks"], strict=True):
            if (actual["rank"], actual["input_sha256"]) != (expected["rank"], expected["input_sha256"]):
                raise ValueError("VLM input hashes differ")
            if len(actual["observations"]) != len(expected["observations"]):
                raise ValueError("VLM step counts differ")
            if not 0 <= run["warmup_steps"] < len(actual["observations"]):
                raise ValueError("No steady VLM observations")
            _finite_values([row["seconds"] for row in actual["observations"]], "VLM latencies", positive=True)
            _finite_values([row["loss"] for row in actual["observations"]], "VLM losses")
    result = dict(
        scope=reference["scope"],
        runs=[str(path) for path in paths],
        settings={key: reference[key] for key in settings},
        inputs_bitwise_equal=True,
        includes_all_steady_observations=True,
        backends={},
    )
    for backend in ("pytorch", "megatron"):
        selected = [run for run in runs if run["backend"] == backend]
        if not selected:
            raise ValueError("Need both VLM backends")
        latencies = [
            row["seconds"] for run in selected for row in run["ranks"][0]["observations"][run["warmup_steps"] :]
        ]
        result["backends"][backend] = dict(
            **_latencies(latencies),
            per_run_median_seconds=[run["median_seconds"] for run in selected],
            peak_allocated_bytes=max(rank["peak_allocated_bytes"] for run in selected for rank in run["ranks"]),
        )
    result["median_throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["median_seconds"] / result["backends"]["megatron"]["median_seconds"] - 1
    )
    result["mean_throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["mean_seconds"] / result["backends"]["megatron"]["mean_seconds"] - 1
    )
    result["maximum_loss_trace_relative_difference"] = max(
        abs(actual["loss"] - expected["loss"]) / max(abs(expected["loss"]), 1e-12)
        for run in runs
        for rank, baseline in zip(run["ranks"], reference["ranks"], strict=True)
        for actual, expected in zip(rank["observations"], baseline["observations"], strict=True)
    )
    return result


def summarize(paths):
    runs = [json.loads(path.read_text()) for path in paths]
    if not runs:
        raise ValueError("Need observations for both backends")
    if any(run["backend"] not in ("pytorch", "megatron") for run in runs):
        raise ValueError("Unknown benchmark backend")
    if "sequence_tokens" in runs[0]:
        return summarize_vlm(paths, runs)
    for run in runs:
        count = len(run["step_seconds"])
        if run.get("passed") is not True:
            raise ValueError("Benchmark run did not pass")
        if run["optimizer_steps"] != count or not 0 <= run["warmup_steps"] < count:
            raise ValueError("Incomplete benchmark optimizer-step observations")
        for key in ("step_seconds", "mean_final_microbatch_losses"):
            if len(run[key]) != count:
                raise ValueError(f"Incomplete benchmark observations: {key}")
            _finite_values(run[key], key, positive=key == "step_seconds")
        if run["live_data"]:
            for key in ("update_including_first_dataload_seconds", "update_data_wait_seconds"):
                if len(run[key]) != count:
                    raise ValueError(f"Incomplete benchmark observations: {key}")
                _finite_values(run[key], key, positive=key == "update_including_first_dataload_seconds")
                if any(value < 0 for value in run[key]):
                    raise ValueError(f"Negative benchmark observations: {key}")
    grouped = {backend: [run for run in runs if run["backend"] == backend] for backend in ("pytorch", "megatron")}
    if any(not values for values in grouped.values()):
        raise ValueError("Need observations for both backends")
    strategies = {run.get("sharding_strategy") for run in grouped["megatron"]}
    if len(strategies) != 1:
        raise ValueError("Incompatible Megatron sharding strategies")
    for key in (
        "token_capacities",
        "gradient_accumulation",
        "seed_reset_after_initialization",
        "live_data",
        "checkpoint",
        "fresh_optimizer_state",
        "warmup_steps",
        "excludes",
        "content_trace_enabled",
        "world_size",
        "torch_version",
        "cuda_version",
        "megatron_core_version",
        "gpu",
    ):
        if any(run.get(key) != runs[0].get(key) for run in runs):
            raise ValueError(f"Incompatible benchmark setting: {key}")
    result = {
        "scope": runs[0]["scope"],
        "excludes": runs[0]["excludes"],
        "token_capacities": runs[0]["token_capacities"],
        "gradient_accumulation": runs[0]["gradient_accumulation"],
        "runs": [str(path) for path in paths],
        "backends": {},
    }
    if runs[0]["live_data"]:
        if any(run["world_size"] != runs[0]["world_size"] for run in runs):
            raise ValueError("Incompatible live benchmark world_size")
        reference_inputs = None
        for path, run in zip(paths, runs, strict=True):
            if not run.get("content_trace_enabled"):
                raise ValueError("Live comparison requires complete content traces")
            inputs = []
            for rank in range(run["world_size"]):
                trace = json.loads(path.with_name(f"{path.stem}_inputs_rank_{rank}.json").read_text())
                if len(trace) != len(run["step_seconds"]) * run["gradient_accumulation"]:
                    raise ValueError(f"Incomplete live input trace: {path}, rank {rank}")
                inputs.append(trace)
            if reference_inputs is None:
                reference_inputs = inputs
            elif inputs != reference_inputs:
                raise ValueError(f"Live content/cursor/RNG traces differ: {path}")
        result["live_input_traces_bitwise_equal"] = True
        result["input_trace_microbatches_per_rank"] = len(reference_inputs[0])
        result["includes_content_hashing_overhead"] = True
    for backend, values in grouped.items():
        observations = [value for run in values for value in run["step_seconds"][run["warmup_steps"] :]]
        result["backends"][backend] = {
            **_latencies(observations),
            "per_run_median_seconds": [run["median_seconds"] for run in values],
            "peak_allocated_bytes": max(run["peak_allocated_bytes"] for run in values),
        }
        if runs[0]["live_data"]:
            result["backends"][backend]["whole_update_including_first_dataload"] = _latencies(
                [
                    value
                    for run in values
                    for value in run["update_including_first_dataload_seconds"][run["warmup_steps"] :]
                ]
            )
            result["backends"][backend]["data_wait"] = _latencies(
                [value for run in values for value in run["update_data_wait_seconds"][run["warmup_steps"] :]],
                positive=False,
            )
    result["throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["median_seconds"] / result["backends"]["megatron"]["median_seconds"] - 1
    )
    result["mean_throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["mean_seconds"] / result["backends"]["megatron"]["mean_seconds"] - 1
    )
    if runs[0]["live_data"]:
        result["whole_update_mean_throughput_gain_fraction"] = (
            result["backends"]["pytorch"]["whole_update_including_first_dataload"]["mean_seconds"]
            / result["backends"]["megatron"]["whole_update_including_first_dataload"]["mean_seconds"]
            - 1
        )
    reference = grouped["pytorch"][0]["mean_final_microbatch_losses"]
    if any(len(run["mean_final_microbatch_losses"]) != len(reference) for run in runs):
        raise ValueError("Cannot compare loss traces with different step counts")
    result["max_loss_trace_absolute_difference"] = max(
        abs(actual - expected)
        for run in runs
        for actual, expected in zip(run["mean_final_microbatch_losses"], reference, strict=True)
    )
    result["max_loss_trace_relative_difference"] = max(
        abs(actual - expected) / max(abs(expected), 1e-12)
        for run in runs
        for actual, expected in zip(run["mean_final_microbatch_losses"], reference, strict=True)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
