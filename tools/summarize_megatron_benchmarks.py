"""Summarize paired replay runs without extrapolating to production topology."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths):
    runs = [json.loads(path.read_text()) for path in paths]
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
            "observations": len(observations),
            "median_seconds": statistics.median(observations),
            "per_run_median_seconds": [run["median_seconds"] for run in values],
            "peak_allocated_bytes": max(run["peak_allocated_bytes"] for run in values),
        }
    result["throughput_gain_fraction"] = (
        result["backends"]["pytorch"]["median_seconds"] / result["backends"]["megatron"]["median_seconds"] - 1
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
