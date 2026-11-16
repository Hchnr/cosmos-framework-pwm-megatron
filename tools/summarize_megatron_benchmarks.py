"""Summarize paired replay runs without extrapolating to production topology."""

import argparse
import json
import statistics
from pathlib import Path


def _latencies(observations):
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
        for actual, expected in zip(run["ranks"], reference["ranks"], strict=True):
            if (actual["rank"], actual["input_sha256"]) != (expected["rank"], expected["input_sha256"]):
                raise ValueError("VLM input hashes differ")
            if len(actual["observations"]) != len(expected["observations"]):
                raise ValueError("VLM step counts differ")
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
    if "sequence_tokens" in runs[0]:
        return summarize_vlm(paths, runs)
    grouped = {backend: [run for run in runs if run["backend"] == backend] for backend in ("pytorch", "megatron")}
    if any(not values for values in grouped.values()):
        raise ValueError("Need observations for both backends")
    for key in ("token_capacities", "gradient_accumulation", "seed_reset_after_initialization", "live_data"):
        if any(run[key] != runs[0][key] for run in runs):
            raise ValueError(f"Incompatible benchmark setting: {key}")
    result = {
        "scope": runs[0]["scope"],
        "excludes": runs[0]["excludes"],
        "token_capacities": runs[0]["token_capacities"],
        "gradient_accumulation": runs[0]["gradient_accumulation"],
        "runs": [str(path) for path in paths],
        "backends": {},
    }
    for backend, values in grouped.items():
        observations = [value for run in values for value in run["step_seconds"][run["warmup_steps"] :]]
        result["backends"][backend] = {
            **_latencies(observations),
            "per_run_median_seconds": [run["median_seconds"] for run in values],
            "peak_allocated_bytes": max(run["peak_allocated_bytes"] for run in values),
        }
    result["throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["median_seconds"] / result["backends"]["megatron"]["median_seconds"] - 1
    )
    result["mean_throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["mean_seconds"] / result["backends"]["megatron"]["mean_seconds"] - 1
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
