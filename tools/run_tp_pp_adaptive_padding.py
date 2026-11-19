# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimentally reduce padded UND and action tokens in both TP and FSDP.

This changes execution shapes only, never the packed samples or valid tokens.
Both sides of a throughput comparison must install identical bucket policies.
GPU mathematics, compiler-cache costs and full-model parity remain unqualified.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def parse_buckets(value):
    try:
        numbers = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError("Bucket limits must be comma-separated positive integers") from error
    if not numbers or numbers != tuple(sorted(set(numbers))) or numbers[0] <= 0:
        raise ValueError("Bucket limits must be strictly increasing positive integers")
    return numbers


def bucket_for(length, maximum, buckets):
    if maximum <= 0 or not 0 <= length <= maximum:
        raise ValueError("Actual length must fit the original fixed capacity")
    for bucket in (*buckets, maximum):
        if bucket <= maximum and bucket >= length:
            return bucket
    raise ValueError("No safe bucket for the original capacity")


def adaptive_capacities(pack, fixed_lengths, und_short_tokens, und_buckets, action_buckets):
    if fixed_lengths is None:
        raise ValueError("The experiment requires fixed capacities")
    if len(fixed_lengths) != 3:
        raise ValueError("Expected UND, VIDEO and ACTION capacities")
    und_max, video_max, action_max = map(int, fixed_lengths)
    und_actual = int(pack["_num_causal_tokens"])
    action_actual = len(pack["_action_indices"])
    # Preserve the existing full-UND warmup. Zero also means short-UND is
    # disabled, so in that configuration we cannot safely distinguish the two
    # cases and leave UND at full capacity. Action can still use smaller bins.
    und = bucket_for(und_actual, und_max, und_buckets) if und_short_tokens else und_max
    action = bucket_for(action_actual, action_max, action_buckets)
    if video_max <= 0:
        raise ValueError("VIDEO capacity must be positive")
    return (und, video_max, action), (und_actual, action_actual)


def install(und_buckets, action_buckets, observations):
    from cosmos_framework.model.generator.megatron_tp_pp import io as tp_io
    from cosmos_framework.model.generator.mot import small_action_tower

    original = small_action_tower.compact_pack
    if tp_io.compact_pack is not original:
        raise RuntimeError("Unexpected TP compact_pack binding; refuse ambiguous installation")

    def compact(pack, positions, position_ids, shape, fixed_lengths=None, attention_backend="sdpa", und_short_tokens=0):
        if fixed_lengths is None:
            raise ValueError("Adaptive padding is restricted to fixed-capacity training")
        chosen, actual = adaptive_capacities(pack, fixed_lengths, und_short_tokens, und_buckets, action_buckets)
        result = original(
            pack,
            positions,
            position_ids,
            shape,
            fixed_lengths=chosen,
            attention_backend=attention_backend,
            # The capacity above already includes the short-UND decision.
            und_short_tokens=0,
        )
        observations.append(
            {
                "original_capacity": list(fixed_lengths),
                "actual_und_action": list(actual),
                "chosen_capacity": list(chosen),
            }
        )
        return result

    # Native dynamically imports from small_action_tower; TP holds an imported
    # binding in its I/O module. Both must use the identical policy.
    small_action_tower.compact_pack = compact
    tp_io.compact_pack = compact


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--und-buckets", default="2048,4096,5120,8192,11776")
    parser.add_argument("--action-buckets", default="768,1536,2944")
    parser.add_argument("--shared-input-reduction", action="store_true")
    # An explicit separator protects target options with the same names from
    # being consumed by this wrapper.
    arguments = sys.argv[1:]
    if "--" in arguments:
        separator = arguments.index("--")
        args, remainder = parser.parse_args(arguments[:separator]), arguments[separator + 1 :]
    else:
        args, remainder = parser.parse_known_args(arguments)
    und_buckets, action_buckets = parse_buckets(args.und_buckets), parse_buckets(args.action_buckets)
    rank = os.environ.get("RANK")
    if rank is None or not rank.isdecimal():
        raise ValueError("A numeric distributed RANK is required")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    target = args.report_dir / f"adaptive_padding.rank{rank}.json"
    if target.exists():
        raise FileExistsError(target)
    rows = []
    error = None
    try:
        if args.shared_input_reduction:
            from tools.tp_pp_shared_input_reduction import install as install_shared

            install_shared()
        install(und_buckets, action_buckets, rows)
        sys.path.insert(0, str(args.entry.resolve().parent))
        sys.argv = [str(args.entry), *remainder]
        runpy.run_path(str(args.entry), run_name="__main__")
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        # Only data-parallel/first-stage owners build packs. Empty rows are
        # reported explicitly for non-owning pipeline ranks.
        with target.open("x") as handle:
            json.dump(
                {
                    "scope": "shape-only opt-in; full-model and throughput acceptance pending",
                    "rank": int(rank),
                    "und_buckets": und_buckets,
                    "action_buckets": action_buckets,
                    "rows": rows,
                    "failure": error,
                },
                handle,
                indent=2,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
