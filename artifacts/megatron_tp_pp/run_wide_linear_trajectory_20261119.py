# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Test the explicit wide-accumulator candidate after current GPU measurements."""

import importlib.util
import json
import time
from pathlib import Path

import psutil


def main():
    path = Path(__file__).with_name("run_fp32_and_microbatch_sequence_20261119.py")
    spec = importlib.util.spec_from_file_location("_sequence", path)
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
    identity = json.loads((sequence.OUTPUT / "post_fp32_raw_performance_command.json").read_text())
    while True:
        events = [json.loads(line) for line in (sequence.OUTPUT / "sequence_events.jsonl").read_text().splitlines()]
        if any(event.get("post_fp32_raw_performance_finished") for event in events):
            break
        try:
            process = psutil.Process(identity["pid"])
            if abs(process.create_time() - identity["create_time"]) > 0.01 or process.status() == psutil.STATUS_ZOMBIE:
                raise psutil.NoSuchProcess(identity["pid"])
        except psutil.NoSuchProcess:
            sequence.event(wide_candidate_wait_failed=True)
            return
        time.sleep(15)
    for rank in range(4):
        assert json.loads((sequence.OUTPUT / f"wide_linear_network_qualification.rank{rank}.json").read_text())[
            "passed"
        ]
    code = sequence.run(
        "native_norm_qualification",
        [
            sequence.PYTHON,
            "tools/qualify_tp_pp_native_norm.py",
            "--reference",
            str(sequence.REFERENCE),
            "--output",
            str(sequence.OUTPUT / "native_norm_qualification.json"),
        ],
        single_gpu=True,
    )
    assert code == 0 and json.loads((sequence.OUTPUT / "native_norm_qualification.json").read_text())["passed"]
    sequence.run(
        "native_block_native_norm_control",
        sequence.distributed(
            "tools/run_tp_pp_traced_fp32_math.py",
            "--run-path",
            "tests/megatron/tp_pp_full_source_math_cuda_compare.py",
            "--trace-dir",
            sequence.OUTPUT / "math_traces/native_block_native_norm_control",
            "--native-block-control",
            "--native-norm-order",
            sequence.OUTPUT / "native_parameter_order_from_meta.json",
            "--mode",
            "tp",
            "--tp",
            2,
            "--bundle",
            sequence.TP2,
            "--native-adam-arithmetic",
            "--activation-cpu-offload",
            "--observe-all-updates",
            "--objective",
            "native_rf",
            "--retain-full-text",
            "--deterministic",
            "--toml",
            sequence.TOML,
            "--prepared-dir",
            sequence.OUTPUT / "recovered_math_inputs",
            "--reference-dir",
            sequence.REFERENCE,
            "--output",
            sequence.OUTPUT / "native_block_native_norm_control.json",
        ),
    )
    entry = "tools/run_tp_pp_wide_linear_math.py"
    helper = "tests/megatron/tp_pp_full_source_math_cuda_compare.py"
    for case, inputs in [
        ("new", sequence.OUTPUT / "recovered_math_inputs"),
        ("legacy", sequence.OUTPUT / "legacy_prepared_inputs"),
    ]:
        if case == "legacy":
            assert json.loads((inputs / "manifest.json").read_text())["passed"]
        reference = Path(f"/opt/cosmos-tp-pp-wide-native-{case}-20261119")
        common = [
            "--objective",
            "native_rf",
            "--retain-full-text",
            "--deterministic",
            "--toml",
            str(sequence.TOML),
            "--prepared-dir",
            str(inputs),
            "--reference-dir",
            str(reference),
        ]
        code = sequence.run(
            f"wide_native_{case}",
            [
                sequence.PYTHON,
                entry,
                "--helper",
                helper,
                "--trace-dir",
                str(sequence.OUTPUT / f"wide_traces/native_{case}"),
                "--mode",
                "native",
                "--gpu-leaf-reference",
                "--preserve-saved-layout",
                "--native-device",
                "cuda",
                "--native-optimizer",
                "native_cuda",
                "--bundle",
                str(sequence.TP2),
                *common,
                "--output",
                str(sequence.OUTPUT / f"wide_native_{case}.json"),
            ],
            single_gpu=True,
        )
        if code:
            sequence.event(wide_native_reference_failed=True, case=case, returncode=code)
            continue
        for tp, bundle in [(2, sequence.TP2), (4, sequence.TP4)]:
            label = f"wide_tp{tp}_{case}"
            sequence.run(
                label,
                sequence.distributed(
                    entry,
                    "--helper",
                    helper,
                    "--trace-dir",
                    sequence.OUTPUT / "wide_traces" / label,
                    "--mode",
                    "tp",
                    "--tp",
                    tp,
                    "--bundle",
                    bundle,
                    "--native-norm-order",
                    sequence.OUTPUT / "native_parameter_order_from_meta.json",
                    "--native-adam-arithmetic",
                    "--activation-cpu-offload",
                    "--observe-all-updates",
                    *common,
                    "--output",
                    sequence.OUTPUT / f"{label}.json",
                ),
            )
    sequence.event(
        wide_candidate_sequence_finished=True,
        fp32_storage_fp64_linear_accumulation=True,
        standard_fp32_acceptance=False,
    )


if __name__ == "__main__":
    main()
