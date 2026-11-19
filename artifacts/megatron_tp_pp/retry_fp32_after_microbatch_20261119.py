# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Retry FP32 math after restoring the old node's /share/mixed path alias."""

import importlib.util
import json
import time
from pathlib import Path


def main():
    path = Path(__file__).with_name("run_fp32_and_microbatch_sequence_20261119.py")
    spec = importlib.util.spec_from_file_location("_isolated_sequence", path)
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
    sequence.event(fp32_retry_waiting_for_performance=True, restored_path_alias="/share/mixed -> /share-mixed")
    while True:
        events = [json.loads(line) for line in (sequence.OUTPUT / "sequence_events.jsonl").read_text().splitlines()]
        if any(event.get("sequence_finished") for event in events):
            break
        time.sleep(15)
    sequence.event(fp32_retry_started=True)
    common = [
        "--objective",
        "native_rf",
        "--retain-full-text",
        "--deterministic",
        "--bundle",
        str(sequence.TP2),
        "--toml",
        str(sequence.TOML),
        "--prepared-dir",
        str(sequence.OUTPUT / "recovered_math_inputs"),
        "--reference-dir",
        str(sequence.REFERENCE),
    ]
    entry = "tools/run_tp_pp_traced_fp32_math.py"
    helper = "tests/megatron/tp_pp_full_source_math.py"
    code = sequence.run(
        "fp32_native_ac_reference_retry",
        [
            sequence.PYTHON,
            entry,
            "--run-path",
            helper,
            "--trace-dir",
            str(sequence.OUTPUT / "math_traces/native"),
            "--mode",
            "native",
            "--gpu-leaf-reference",
            "--preserve-saved-layout",
            "--native-device",
            "cuda",
            "--native-optimizer",
            "native_cuda",
            *common,
            "--output",
            str(sequence.OUTPUT / "fp32_native_ac_reference.json"),
        ],
        single_gpu=True,
    )
    if code:
        sequence.event(fp32_retry_reference_failed=True, returncode=code, numerical_acceptance=False)
        return
    for label, extra, tp, bundle in [
        ("fp32_tp2_standard", [], 2, sequence.TP2),
        ("fp32_native_block_control", ["--native-block-control"], 2, sequence.TP2),
        ("fp32_tp4_standard", [], 4, sequence.TP4),
    ]:
        case = list(common)
        case[case.index("--bundle") + 1] = str(bundle)
        sequence.run(
            label,
            sequence.distributed(
                entry,
                "--run-path",
                helper,
                "--trace-dir",
                sequence.OUTPUT / "math_traces" / label,
                *extra,
                "--mode",
                "tp",
                "--tp",
                tp,
                "--native-adam-arithmetic",
                "--activation-cpu-offload",
                "--observe-all-updates",
                *case,
                "--output",
                sequence.OUTPUT / f"{label}.json",
            ),
        )
    sequence.event(fp32_retry_sequence_finished=True, numerical_acceptance=False, full_capacity_validation_pending=True)


if __name__ == "__main__":
    main()
