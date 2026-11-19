# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize equal-per-microstep TP4/FSDP4 training-compute measurements."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root, backend):
    contracts = [json.loads((root / f"contract.rank{rank}.json").read_text()) for rank in range(4)]
    first = contracts[0]
    if any(contract != first for contract in contracts):
        raise ValueError("Rank benchmark contracts differ")
    if first["backend"] != backend or first["microsteps_per_update"] != 3:
        raise ValueError("Unexpected benchmark topology or accumulation")
    log_root = root / "training"
    if backend == "tp":
        log_root = log_root / "cosmos3_action/action128/tp4_microbatch"
    ranks = []
    for rank in range(4):
        rows = [json.loads(line) for line in (log_root / f"updates.rank{rank}.jsonl").read_text().splitlines()]
        if len(rows) != first["warmup_updates"] + first["measured_updates"]:
            raise ValueError("Incomplete measurement window")
        if [row["iteration"] for row in rows] != list(range(1201, 1201 + len(rows))):
            raise ValueError("Unexpected or duplicated iterations")
        ranks.append(rows)
    critical = [max(rank[index]["update_seconds"] for rank in ranks) for index in range(len(ranks[0]))]
    measured = critical[first["warmup_updates"] :]
    mean = statistics.mean(measured)
    samples = sum(batch["samples"] for batch in first["per_rank_microbatches"])
    return {
        "contract": first,
        "slowest_rank_update_seconds": critical,
        "measured_mean_update_seconds": mean,
        "measured_median_update_seconds": statistics.median(measured),
        "amortized_microstep_seconds": mean / 3,
        "global_samples_per_update": samples * first["dp"],
        "global_samples_per_second": samples * first["dp"] / mean,
        "real_tokens_per_second": {
            key: sum(batch["tokens"][key] for batch in first["per_rank_microbatches"]) * first["dp"] / mean
            for key in ("text", "vision", "action")
        },
    }


def compare(tp_root, fsdp_root):
    tp, fsdp = summarize(tp_root, "tp"), summarize(fsdp_root, "fsdp")
    for key in ("per_rank_microbatches", "microsteps_per_update", "warmup_updates", "measured_updates", "precision"):
        if tp["contract"][key] != fsdp["contract"][key]:
            raise ValueError(f"Per-microstep comparison mismatch: {key}")
    if (tp["contract"]["tp"], tp["contract"]["pp"], tp["contract"]["dp"], fsdp["contract"]["dp"]) != (4, 1, 1, 4):
        raise ValueError("Require TP4/PP1/DP1 versus FSDP4")
    payload_kinds = {
        batch.get("payload_kind", "prepared") for batch in tp["contract"]["per_rank_microbatches"]
    }
    if payload_kinds == {"prepared"}:
        scope = "real immutable prepared input replay; training compute, communication and optimizer; media/VAE excluded"
    elif payload_kinds == {"raw_before_vae_and_noising"}:
        scope = (
            "real immutable raw input replay; fixture copy, H2D, native VAE/noising, training, communication "
            "and optimizer; media decode and checkpoint writes excluded"
        )
    else:
        raise ValueError(f"Unknown or mixed benchmark payload kinds: {payload_kinds}")
    return {
        "measurement_completed": True,
        "scope": scope,
        "per_microstep_batches_exact": True,
        "global_batch_matched": False,
        "numerical_acceptance": False,
        "tp": tp,
        "fsdp": fsdp,
        "tp_over_fsdp_sample_throughput": tp["global_samples_per_second"] / fsdp["global_samples_per_second"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-root", type=Path, required=True)
    parser.add_argument("--fsdp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.tp_root, args.fsdp_root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
