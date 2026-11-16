# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize per-rank load-balance JSONL traces.

Run with::

    python -m cosmos_framework.tools.load_balance_report <load_balance_trace_dir>

The report first collapses duplicate timings inside each context-parallel group
with ``max`` (the CP critical path), then compares independent CP groups in the
same slot. Timings can include collective wait, so the reported balance ratio
is an observed skew proxy rather than a guaranteed scheduler speedup.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

_TIMING_FIELDS = (
    "batch_cuda_ms",
    "forward_cuda_ms",
    "backward_cuda_ms",
    "batch_host_ms",
    "forward_host_ms",
    "backward_host_ms",
    "dataloader_host_ms",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Iterable[float | int]) -> dict[str, float] | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return {
        "min": min(numbers),
        "mean": fmean(numbers),
        "p10": _percentile(numbers, 0.10),
        "p50": _percentile(numbers, 0.50),
        "p90": _percentile(numbers, 0.90),
        "p95": _percentile(numbers, 0.95),
        "max": max(numbers),
    }


def load_trace_records(trace_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load all rank JSONL files and add their source location."""
    paths = sorted(trace_dir.glob("rank_*.jsonl"))
    if not paths:
        raise ValueError(f"No rank_*.jsonl files found in {trace_dir}.")

    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
                record["_source_file"] = path.name
                record["_source_line"] = line_number
                records.append(record)
    return records, paths


def _unit_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["iteration"]),
        int(record["cp_window_index"]),
        int(record["cp_window_slot"]),
        int(record["cp_group_id"]),
    )


def _sync_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["iteration"]),
        int(record["cp_window_index"]),
        int(record["cp_window_slot"]),
    )


def _vae_pixel_volume(output: dict[str, Any]) -> int:
    total = 0
    for shape in output.get("vae_pixel_shapes", []):
        if isinstance(shape, list) and all(isinstance(value, int) for value in shape):
            volume = 1
            for value in shape:
                volume *= value
            total += volume
    return total


def _attention_area(output: dict[str, Any]) -> int:
    total = 0
    for length, mode in zip(output.get("split_lens", []), output.get("attn_modes", []), strict=False):
        if not isinstance(length, int):
            continue
        total += length * (length + 1) // 2 if mode == "causal" else length * length
    return total


def _owner_features(owner: dict[str, Any]) -> dict[str, float]:
    output = owner.get("output", {})
    plans = owner.get("data", {}).get("sequence_plans", [])
    conditioning = Counter(plan.get("vision_conditioning") for plan in plans)
    pixel_shapes = output.get("vae_pixel_shapes", [])
    video_frames = sum(shape[0] for shape in pixel_shapes if isinstance(shape, list) and shape)
    und_tokens = int(output.get("und_token_length", 0))
    gen_tokens = int(output.get("gen_token_length", 0))
    return {
        "sample_count": float(output.get("batch_size", 0)),
        "und_token_length": float(und_tokens),
        "gen_token_length": float(gen_tokens),
        "total_token_length": float(und_tokens + gen_tokens),
        "attention_area": float(_attention_area(output)),
        "vae_pixel_volume": float(_vae_pixel_volume(output)),
        "video_frames": float(video_frames),
        "t2v_count": float(conditioning["t2v"]),
        "i2v_count": float(conditioning["i2v"]),
        "v2v_count": float(conditioning["v2v"]),
        "vision_condition_frames": float(sum(int(plan.get("vision_condition_frames", 0)) for plan in plans)),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = sum(value * value for value in left_delta) * sum(value * value for value in right_delta)
    if denominator == 0:
        return None
    return sum(x * y for x, y in zip(left_delta, right_delta, strict=True)) / denominator**0.5


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def _correlation_report(units: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for timing_field in _TIMING_FIELDS:
        observations = [unit for unit in units if unit["owner"] is not None and timing_field in unit]
        if not observations:
            continue
        feature_rows = [_owner_features(unit["owner"]) for unit in observations]
        timings = [float(unit[timing_field]) for unit in observations]
        features: dict[str, Any] = {}
        for feature_name in feature_rows[0]:
            values = [row[feature_name] for row in feature_rows]
            features[feature_name] = {
                "pearson": _pearson(values, timings),
                "spearman": _pearson(_ranks(values), _ranks(timings)),
            }
        result[timing_field] = {"observations": len(observations), "features": features}
    return result


def _feature_report(units: list[dict[str, Any]]) -> dict[str, Any]:
    owners = [unit["owner"] for unit in units if unit["owner"] is not None]
    conditioning: Counter[str] = Counter()
    sequence_lengths: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    frame_bucket_lengths: defaultdict[int, defaultdict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for owner in owners:
        plans = owner.get("data", {}).get("sequence_plans", [])
        video_shapes = owner.get("data", {}).get("video_shapes", [])
        output = owner.get("output", {})
        split_lens = output.get("split_lens", [])
        attn_modes = output.get("attn_modes", [])
        sample_splits: list[tuple[int, int]] = []
        split_index = 0
        while split_index + 1 < len(split_lens) and split_index + 1 < len(attn_modes):
            und_length = split_lens[split_index]
            gen_length = split_lens[split_index + 1]
            if (
                attn_modes[split_index] == "causal"
                and attn_modes[split_index + 1] == "full"
                and isinstance(und_length, int)
                and isinstance(gen_length, int)
            ):
                sample_splits.append((und_length, gen_length))
                split_index += 2
            else:
                split_index += 1

        for plan_index, plan in enumerate(plans):
            name = plan.get("vision_conditioning")
            if isinstance(name, str):
                conditioning[name] += 1
                if plan_index < len(sample_splits):
                    und_length, gen_length = sample_splits[plan_index]
                    sequence_lengths[name]["und_token_length"].append(und_length)
                    sequence_lengths[name]["gen_token_length"].append(gen_length)
                    sequence_lengths[name]["total_token_length"].append(und_length + gen_length)
                    condition_frames = plan.get("vision_condition_frames", 0)
                    if isinstance(condition_frames, int):
                        sequence_lengths[name]["vision_condition_frames"].append(condition_frames)

        for sample_index, sample_shapes in enumerate(video_shapes):
            frame_count = None
            for shape in sample_shapes if isinstance(sample_shapes, list) else []:
                if isinstance(shape, list) and len(shape) >= 4 and all(isinstance(value, int) for value in shape):
                    # CP slot 0 records CTHW while a rotated slot can retain a
                    # singleton batch dimension (BCTHW). T is third from last.
                    frame_count = shape[-3]
                    break
            if frame_count is not None and sample_index < len(sample_splits):
                und_length, gen_length = sample_splits[sample_index]
                frame_bucket_lengths[frame_count]["und_token_length"].append(und_length)
                frame_bucket_lengths[frame_count]["gen_token_length"].append(gen_length)
                frame_bucket_lengths[frame_count]["total_token_length"].append(und_length + gen_length)

    def output_numbers(field: str) -> list[int | float]:
        values: list[int | float] = []
        for owner in owners:
            value = owner.get("output", {}).get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(value)
        return values

    total_tokens = [
        int(owner.get("output", {}).get("und_token_length", 0))
        + int(owner.get("output", {}).get("gen_token_length", 0))
        for owner in owners
    ]
    conditioning_total = sum(conditioning.values())
    conditioning_sequence_lengths = {
        name: {
            "sample_count": conditioning[name],
            "observed_ratio": 0.0 if conditioning_total == 0 else conditioning[name] / conditioning_total,
            **{field: _numeric_summary(values) for field, values in sorted(fields.items())},
        }
        for name, fields in sorted(sequence_lengths.items())
    }
    frame_bucket_total = sum(len(fields["total_token_length"]) for fields in frame_bucket_lengths.values())
    frame_bucket_sequence_lengths = {
        str(frame_count): {
            "sample_count": len(fields["total_token_length"]),
            "observed_ratio": 0.0
            if frame_bucket_total == 0
            else len(fields["total_token_length"]) / frame_bucket_total,
            **{field: _numeric_summary(values) for field, values in sorted(fields.items())},
        }
        for frame_count, fields in sorted(frame_bucket_lengths.items())
    }
    return {
        "owner_records": len(owners),
        "conditioning_counts": dict(sorted(conditioning.items())),
        "conditioning_sequence_lengths": conditioning_sequence_lengths,
        "frame_bucket_sequence_lengths": frame_bucket_sequence_lengths,
        "sample_count": _numeric_summary(output_numbers("batch_size")),
        "und_token_length": _numeric_summary(output_numbers("und_token_length")),
        "gen_token_length": _numeric_summary(output_numbers("gen_token_length")),
        "total_token_length": _numeric_summary(total_tokens),
        "vae_pixel_volume": _numeric_summary(_vae_pixel_volume(owner.get("output", {})) for owner in owners),
        "attention_area": _numeric_summary(_attention_area(owner.get("output", {})) for owner in owners),
    }


def _timing_report(units: list[dict[str, Any]]) -> dict[str, Any]:
    units_by_sync: defaultdict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        units_by_sync[unit["sync_key"]].append(unit)

    result: dict[str, Any] = {}
    for field in _TIMING_FIELDS:
        windows: list[dict[str, float]] = []
        peer_spreads: list[float] = []
        present_unit_count = 0
        for sync_units in units_by_sync.values():
            values = [unit[field] for unit in sync_units if field in unit]
            present_unit_count += len(values)
            if not values:
                continue
            mean_value = fmean(values)
            max_value = max(values)
            windows.append(
                {
                    "min_ms": min(values),
                    "mean_ms": mean_value,
                    "max_ms": max_value,
                    "bubble_proxy": 0.0 if max_value == 0 else 1 - mean_value / max_value,
                }
            )
            peer_spreads.extend(unit[f"{field}_cp_peer_spread"] for unit in sync_units if field in unit)
        if not windows:
            continue

        sum_mean = sum(window["mean_ms"] for window in windows)
        sum_max = sum(window["max_ms"] for window in windows)
        bubble_values = [window["bubble_proxy"] for window in windows]
        result[field] = {
            "unit_coverage": present_unit_count / len(units),
            "window_count": len(windows),
            "total_mean_work_ms": sum_mean,
            "total_critical_path_ms": sum_max,
            "weighted_bubble_proxy": 0.0 if sum_max == 0 else 1 - sum_mean / sum_max,
            "zero_overhead_balance_ratio_proxy": None if sum_mean == 0 else sum_max / sum_mean,
            "bubble_proxy": _numeric_summary(bubble_values),
            "cp_peer_relative_spread": _numeric_summary(peer_spreads),
        }
    return result


def build_report(records: list[dict[str, Any]], file_count: int | None = None) -> dict[str, Any]:
    """Build an integrity, feature, and timing report from trace records."""
    starts = [record for record in records if record.get("record_type") == "trace_start"]
    ends = [record for record in records if record.get("record_type") == "trace_end"]
    microbatches = [record for record in records if record.get("record_type") == "microbatch"]
    if not microbatches:
        raise ValueError("Trace contains no microbatch records.")

    rank_ids = sorted({int(record["rank"]) for record in starts or microbatches})
    sync_keys = sorted({_sync_key(record) for record in microbatches})
    cp_group_ids = sorted({int(record["cp_group_id"]) for record in microbatches})
    grouped: defaultdict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in microbatches:
        grouped[_unit_key(record)].append(record)

    units: list[dict[str, Any]] = []
    valid_owner_units = 0
    valid_cp_rank_units = 0
    consistent_output_units = 0
    for key, members in sorted(grouped.items()):
        cp_size = int(members[0].get("cp_size", 1))
        owners = [member for member in members if member.get("is_active_payload_owner") is True]
        owner = owners[0] if len(owners) == 1 else None
        valid_owner_units += owner is not None
        valid_cp_rank_units += len({int(member["rank"]) for member in members}) == cp_size
        serialized_outputs = {json.dumps(member.get("output", {}), sort_keys=True) for member in members}
        consistent_output_units += len(serialized_outputs) == 1
        unit: dict[str, Any] = {
            "key": key,
            "sync_key": key[:3],
            "owner": owner,
        }
        for field in _TIMING_FIELDS:
            values = [float(member[field]) for member in members if isinstance(member.get(field), (int, float))]
            if values:
                unit[field] = max(values)
                maximum = max(values)
                unit[f"{field}_cp_peer_spread"] = 0.0 if maximum == 0 else (maximum - min(values)) / maximum
        units.append(unit)

    expected_rank_records = len(sync_keys) * len(rank_ids)
    expected_units = len(sync_keys) * len(cp_group_ids)
    unit_count = len(units)
    declared_world_sizes = sorted({int(record["world_size"]) for record in starts if "world_size" in record})
    cp_slots = sorted({unit["sync_key"][2] for unit in units})
    return {
        "schema_version": 1,
        "integrity": {
            "file_count": file_count,
            "rank_count": len(rank_ids),
            "rank_ids": rank_ids,
            "declared_world_sizes": declared_world_sizes,
            "trace_start_ranks": sorted(int(record["rank"]) for record in starts),
            "trace_end_ranks": sorted(int(record["rank"]) for record in ends),
            "microbatch_records": len(microbatches),
            "sync_windows": len(sync_keys),
            "cp_work_units": unit_count,
            "rank_record_coverage": 0.0 if expected_rank_records == 0 else len(microbatches) / expected_rank_records,
            "cp_work_unit_coverage": 0.0 if expected_units == 0 else unit_count / expected_units,
            "cp_rank_coverage": 0.0 if unit_count == 0 else valid_cp_rank_units / unit_count,
            "payload_owner_coverage": 0.0 if unit_count == 0 else valid_owner_units / unit_count,
            "cp_output_consistency": 0.0 if unit_count == 0 else consistent_output_units / unit_count,
        },
        "features": _feature_report(units),
        "features_by_cp_slot": {
            str(slot): _feature_report([unit for unit in units if unit["sync_key"][2] == slot]) for slot in cp_slots
        },
        "timings": _timing_report(units),
        "timings_by_cp_slot": {
            str(slot): _timing_report([unit for unit in units if unit["sync_key"][2] == slot]) for slot in cp_slots
        },
        "correlations": _correlation_report(units),
        "correlations_by_cp_slot": {
            str(slot): _correlation_report([unit for unit in units if unit["sync_key"][2] == slot]) for slot in cp_slots
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    integrity = report["integrity"]
    lines = [
        "# Load-balance trace report",
        "",
        (
            f"Ranks: {integrity['rank_count']} | microbatch records: {integrity['microbatch_records']} | "
            f"sync windows: {integrity['sync_windows']} | CP work units: {integrity['cp_work_units']}"
        ),
        "",
        (
            "Coverage: "
            f"rank records {integrity['rank_record_coverage']:.3%}, "
            f"CP ranks {integrity['cp_rank_coverage']:.3%}, "
            f"payload owners {integrity['payload_owner_coverage']:.3%}, "
            f"CP output consistency {integrity['cp_output_consistency']:.3%}."
        ),
        "",
        "| Timing | Windows | Weighted bubble proxy | P95 bubble | Max bubble | Balance ratio proxy | CP-peer P95 spread |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, summary in report["timings"].items():
        bubble = summary["bubble_proxy"]
        peer = summary["cp_peer_relative_spread"]
        ratio = summary["zero_overhead_balance_ratio_proxy"]
        lines.append(
            f"| {field} | {summary['window_count']} | {summary['weighted_bubble_proxy']:.2%} | "
            f"{bubble['p95']:.2%} | {bubble['max']:.2%} | {ratio:.4f} | {peer['p95']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Sequence length by vision conditioning",
            "",
            "| Conditioning | Samples | Mix | UND P50 (P10–P90) | GEN P50 (P10–P90) | Total P50 (P10–P90) | Conditioning frames P50 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, summary in report["features"]["conditioning_sequence_lengths"].items():
        und = summary["und_token_length"]
        gen = summary["gen_token_length"]
        total = summary["total_token_length"]
        condition_frames = summary["vision_condition_frames"]
        lines.append(
            f"| {name} | {summary['sample_count']} | {summary['observed_ratio']:.2%} | "
            f"{und['p50']:.0f} ({und['p10']:.0f}–{und['p90']:.0f}) | "
            f"{gen['p50']:.0f} ({gen['p10']:.0f}–{gen['p90']:.0f}) | "
            f"{total['p50']:.0f} ({total['p10']:.0f}–{total['p90']:.0f}) | "
            f"{condition_frames['p50']:.0f} |"
        )
    lines.extend(
        [
            "",
            "UND/GEN are the complete causal/full Transformer segments. GEN includes both clean conditioning "
            "and noised target vision tokens; the trace does not relabel clean GEN tokens as UND.",
        ]
    )
    lines.extend(
        [
            "",
            "## Sequence length by pixel-frame bucket",
            "",
            "| Pixel frames | Samples | Mix | UND P50 (P10–P90) | GEN P50 (P10–P90) | Total P50 (P10–P90) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for frame_count, summary in report["features"]["frame_bucket_sequence_lengths"].items():
        und = summary["und_token_length"]
        gen = summary["gen_token_length"]
        total = summary["total_token_length"]
        lines.append(
            f"| {frame_count} | {summary['sample_count']} | {summary['observed_ratio']:.2%} | "
            f"{und['p50']:.0f} ({und['p10']:.0f}–{und['p90']:.0f}) | "
            f"{gen['p50']:.0f} ({gen['p10']:.0f}–{gen['p90']:.0f}) | "
            f"{total['p50']:.0f} ({total['p10']:.0f}–{total['p90']:.0f}) |"
        )
    lines.extend(
        [
            "",
            "## CUDA timing by CP slot",
            "",
            "| CP slot | Timing | Windows | Weighted bubble proxy | P95 bubble | Max bubble | Balance ratio proxy |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slot, timings in report["timings_by_cp_slot"].items():
        for field in ("batch_cuda_ms", "forward_cuda_ms", "backward_cuda_ms"):
            if field not in timings:
                continue
            summary = timings[field]
            bubble = summary["bubble_proxy"]
            ratio = summary["zero_overhead_balance_ratio_proxy"]
            lines.append(
                f"| {slot} | {field} | {summary['window_count']} | {summary['weighted_bubble_proxy']:.2%} | "
                f"{bubble['p95']:.2%} | {bubble['max']:.2%} | {ratio:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Batch CUDA correlation by CP slot",
            "",
            "| CP slot | Feature | Observations | Spearman | Pearson |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for slot, correlations in report["correlations_by_cp_slot"].items():
        batch = correlations.get("batch_cuda_ms")
        if batch is None:
            continue
        ranked_features = sorted(
            ((feature, values) for feature, values in batch["features"].items() if values["spearman"] is not None),
            key=lambda item: abs(item[1]["spearman"]),
            reverse=True,
        )
        for feature, values in ranked_features[:5]:
            lines.append(
                f"| {slot} | {feature} | {batch['observations']} | {values['spearman']:.3f} | {values['pearson']:.3f} |"
            )
    features = report["features"]
    lines.extend(
        [
            "",
            "The balance ratio is derived from observed elapsed time and may include collective wait; it is not a promised speedup.",
            "Feature correlations are observational screening signals, not causal attribution.",
            "",
            f"Conditioning counts (payload owners only): `{json.dumps(features['conditioning_counts'], sort_keys=True)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, help="Directory containing rank_*.jsonl files.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Optional output file; stdout is used by default.")
    parser.add_argument("--min-iteration", type=int, help="Keep microbatches at or after this iteration.")
    parser.add_argument("--max-iteration", type=int, help="Keep microbatches before this iteration.")
    args = parser.parse_args()

    records, paths = load_trace_records(args.trace_dir)
    if args.min_iteration is not None or args.max_iteration is not None:
        records = [
            record
            for record in records
            if record.get("record_type") != "microbatch"
            or (
                (args.min_iteration is None or int(record["iteration"]) >= args.min_iteration)
                and (args.max_iteration is None or int(record["iteration"]) < args.max_iteration)
            )
        ]
    report = build_report(records, file_count=len(paths))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else render_markdown(report)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
