# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Continue complete trajectories with the qualified CUDA FP64 validator."""

import importlib.util
import json
from pathlib import Path


def main():
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("_sequence", root / "run_fp32_and_microbatch_sequence_20261119.py")
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
    assert json.loads((sequence.OUTPUT / "gpu_comparison_qualification.json").read_text())["passed"]
    helper = "tests/megatron/tp_pp_full_source_math_cuda_compare.py"
    common = [
        "--objective",
        "native_rf",
        "--retain-full-text",
        "--deterministic",
        "--toml",
        str(sequence.TOML),
        "--prepared-dir",
        str(sequence.OUTPUT / "recovered_math_inputs"),
        "--reference-dir",
        str(sequence.REFERENCE),
    ]
    sequence.event(cuda_validator_sequence_started=True, training_algorithm_changed=False)
    for label, extra, tp, bundle in [
        ("fp32_tp2_standard_gpu", [], 2, sequence.TP2),
        ("fp32_native_block_control_gpu", ["--native-block-control"], 2, sequence.TP2),
        ("fp32_tp4_standard_gpu", [], 4, sequence.TP4),
    ]:
        sequence.run(
            label,
            sequence.distributed(
                "tools/run_tp_pp_traced_fp32_math.py",
                "--run-path",
                helper,
                "--trace-dir",
                sequence.OUTPUT / "math_traces" / label,
                *extra,
                "--mode",
                "tp",
                "--tp",
                tp,
                "--bundle",
                bundle,
                "--native-adam-arithmetic",
                "--activation-cpu-offload",
                "--observe-all-updates",
                *common,
                "--output",
                sequence.OUTPUT / f"{label}.json",
            ),
        )
        if label == "fp32_tp2_standard_gpu":
            checks = []
            for rank in range(4):
                cpu = json.loads(
                    (sequence.OUTPUT / f"fp32_tp2_standard.update0.observation.rank{rank}.json").read_text()
                )
                cuda = json.loads((sequence.OUTPUT / f"{label}.update0.observation.rank{rank}.json").read_text())
                assert cpu["passed"] == cuda["passed"]
                for key in ("relative_l2", "cosine"):
                    assert abs(cpu["gradient_summary"][key] - cuda["gradient_summary"][key]) < 1e-12
                assert {row["name"] for row in cpu["failed_parameter_shards"]} == {
                    row["name"] for row in cuda["failed_parameter_shards"]
                }
                checks.append(dict(rank=rank, passed=True, cpu=cpu["gradient_summary"], cuda=cuda["gradient_summary"]))
            (sequence.OUTPUT / "full_source_cpu_cuda_first_gradient_comparison.json").write_text(
                json.dumps(dict(passed=True, checks=checks), indent=2) + "\n"
            )
    sequence.event(cuda_validator_sequence_finished=True, full_capacity_validation_pending=True)


if __name__ == "__main__":
    main()
